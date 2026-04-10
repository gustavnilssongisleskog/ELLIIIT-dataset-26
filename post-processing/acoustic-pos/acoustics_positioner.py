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
chirp_orig_resampl = alf.resample_chirp(fs_source, fs_mic, chirp_orig)

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

    # Loop over all selected microphones
    for mic_label, mic_position in selected_mic_positions.items():
        print(f"\nProcessing microphone {mic_label}...")

        # Get the data itself
        waveform, waveform_values, sample_index = alf.get_acoustic_waveform(EXPERIMENT_ID, CYCLE_ID, mic_label, DATASET_PATH)

        # Plot the waveform
        alf.plot_received_signal_mic(sample_index, waveform_values, EXPERIMENT_ID, CYCLE_ID, mic_label)

        # Do anchor selection for microphones (choose 6 from for instance 15)

        # Do Pulse Compression

        # Add LPF

        # Perform MB positioning
            # Do ranging based on peak prominence

            # Do multilateriation based on LS

        # Perform GNN positioning