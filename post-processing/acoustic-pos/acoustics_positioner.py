from logging import config
import acoustic_local_functions as alf
import json
from pathlib import Path
import numpy as np

#MICROPHONE_LABEL = "D06"  # Set the exact microphone label for the acoustic waveform.
DATASET_PATH = None  # Set this to a specific .nc file when you do not want the newest match.
CSI_DATASET_PATH = None  # Optional: provide a specific CSI dataset path if needed.

# Get the directory where this script is located
script_dir = Path(__file__).parent
config_path = script_dir / 'config.json'

with open(config_path) as json_file:
    config = json.load(json_file)

n_selected_ans = config['number_of_selected_anchors']
anchor_selection_method = config['anchor_selection_method']
sumrate_threshold = config['sumrate_threshold']

v_sound = 20 * np.sqrt(273 + config["temperature"])
print('\nSpeed of sound: ', v_sound, ' m/s\n')

if config['use_all_mics']:
    MICROPHONE_LABEL = config['all_mics']
else:
    MICROPHONE_LABEL = config['used_mics']

microphone_positions = alf.load_microphone_positions()

# Get positions for selected microphones as a reusable dictionary
selected_mic_positions = alf.get_selected_mic_positions(microphone_positions, MICROPHONE_LABEL)

# read original Chirp
fs_source, chirp_orig, duration_chirp = alf.read_transmit_chirp()  # fs_source is the original sampling frequency of the chirp, e.g., 500000 Hz
print("Duration of the transmitted chirp: %.2f s" % duration_chirp)
print("Sample rate of the source : %.2f Hz" % fs_source)

fs_mic = config['fs_mic']  # sampling frequency at the microphone
print("Sample rate of the microphone : %.2f Hz" % fs_mic)

# resample if fs source is not equal to fs receive
chirp_orig_resampl = alf.resample_chirp(fs_source, fs_mic, chirp_orig)*10

if __name__ == "__main__":

    EXPERIMENT_ID = "EXP008"
    CYCLE_ID = 20  # Set the exact acoustic cycle you want to inspect.

    # Read the GT position for that measurement
    position = alf.get_rover_position(EXPERIMENT_ID, CYCLE_ID, CSI_DATASET_PATH)
    if position["position_available"]:
        print(
            f"Rover position for cycle {CYCLE_ID}: "
            f"x={position['rover_x']:.2f}, y={position['rover_y']:.2f}, z={position['rover_z']:.2f}"
        )
    else:
        print(f"No rover position available for cycle {CYCLE_ID}")

    # Read the dataset once (shared across all microphones)
    ds, dataset_path = alf.open_acoustic_dataset(EXPERIMENT_ID, DATASET_PATH)
    print(f"Processing {len(selected_mic_positions)} microphones...")

    # Collect anchor candidates for selection
    anchor_candidates = []

    # Loop over all selected microphones
    for mic_label, mic_position in selected_mic_positions.items():
        # Get the data itself
        waveform, waveform_values, sample_index = alf.get_acoustic_waveform(EXPERIMENT_ID, CYCLE_ID, mic_label, DATASET_PATH)

        # Do some High pass filtering to remove low frequency noise (e.g., from fans, server, etc.) before anchor selection 
        waveform_filtered = alf.butter_highpass_filter(waveform_values, 15000, fs_mic, 10)

        # Plot the waveform
        # alf.plot_received_signal_mic(np.arange(len(chirp_orig_resampl)), chirp_orig_resampl, EXPERIMENT_ID, CYCLE_ID, mic_label)
        # alf.plot_received_signal_comb(sample_index, waveform_values, waveform_filtered, EXPERIMENT_ID, CYCLE_ID, mic_label)

        # Collect anchor metadata for selection based on signal strength
        anchor_info = alf.select_anchors(waveform_filtered, mic_label, EXPERIMENT_ID, CYCLE_ID, n_selected_ans, sumrate_threshold)
        anchor_candidates.append(anchor_info)


    # Select the top anchors based on the chosen method
    selected_anchors_dict, sort_key = alf.select_top_anchors(anchor_candidates, anchor_selection_method, n_selected_ans)

    # Print selected anchors information
    alf.print_selected_anchors_info(selected_anchors_dict, sort_key, anchor_selection_method)

    # Get waveform_values from selected_anchors_dict and apply gain_equaliser to all at once
    signals = np.array([anchor['waveform_values'] for anchor in selected_anchors_dict.values()])
    adjusted_signals = alf.gain_equaliser(chirp_orig_resampl, signals, 8000, 10000)
    
    # Save the adjusted waveforms back in selected_anchors_dict
    for i, mic_label in enumerate(selected_anchors_dict.keys()):
        selected_anchors_dict[mic_label]['adjusted_waveform'] = adjusted_signals[i]

    # Plot all adjusted waveforms with the original chirp
    alf.plot_adjusted_waveforms_with_chirp(selected_anchors_dict, chirp_orig_resampl, EXPERIMENT_ID, CYCLE_ID)

    true_distances = None
    if position["position_available"]:
        rover_xyz = np.array([position['rover_x'], position['rover_y'], position['rover_z']], dtype=float)
        true_distances = {}
        for mic_label in selected_anchors_dict.keys():
            mic_xyz = np.asarray(selected_mic_positions[mic_label], dtype=float)
            true_distances[mic_label] = float(np.linalg.norm(mic_xyz - rover_xyz))

    # Do Pulse Compression
    pulse_compr_all, LPF_all, corr_index_array = alf.get_corr_with_LPF_curve(chirp_orig_resampl, selected_anchors_dict)

    alf.plot_pulsecomp_and_lpf(
        pulse_compr_all,
        LPF_all,
        corr_index_array,
        fs_mic,
        selected_anchors_dict,
        chirp_orig_resampl,
        true_distances=true_distances,
        plot_full_pulse_compression=config["plot_full_pulse_compression"],
    )

    # Perform MB positioning -----------------------------------------------------------------------------------------------
    # Do ranging based on peak prominence

    n_wake_up_samples = config["wake_up_duration"] * fs_mic
    n_wake_up_samples_eff = int(min(n_wake_up_samples, np.size(chirp_orig_resampl)))
    wake_up_at_sample = int(np.size(chirp_orig_resampl) - n_wake_up_samples_eff)
    eff_start_samp_chirp = (((corr_index_array + 1) - n_wake_up_samples_eff)[np.newaxis]).T
    delta_sample = wake_up_at_sample - eff_start_samp_chirp
    distances_meas = (delta_sample / fs_mic) * v_sound

    print("Measured distances (m):", distances_meas)
    if true_distances is not None:
        print("Theoretical distances (m):")
        for idx, mic_label in enumerate(selected_anchors_dict.keys()):
            meas_val = float(distances_meas[idx, 0])
            gt_val = float(true_distances[mic_label])
            print(f"  {mic_label}: measured={meas_val:.3f}, theoretical={gt_val:.3f}, error={meas_val - gt_val:+.3f}")

    # Do multilateriation based on LS

    # Perform GNN positioning ------------------------------------------------------------------------------------------------
