import acoustic_local_functions as alf
import numpy as np
from pathlib import Path

DATASET_PATH = None      # Set to a specific .nc file to override the newest match.
CSI_DATASET_PATH = None  # Optional: provide a specific CSI dataset path if needed.

if __name__ == "__main__":

    EXPERIMENT_ID = "EXP008"
    CYCLE_ID = 70  # Set the exact acoustic cycle you want to inspect.
    PATH_ID = 0

    # Load all setup values (config, microphone positions, chirp)
    setup = alf.load_experiment_setup()
    print('\nSpeed of sound: ', setup['v_sound'], ' m/s\n')
    print("Duration of the transmitted chirp: %.2f s" % setup['duration_chirp'])
    print("Sample rate of the source : %.2f Hz" % setup['fs_source'])
    print("Sample rate of the microphone : %.2f Hz" % setup['fs_mic'])
    print(f"Processing {len(setup['selected_mic_positions'])} microphones...")

    # Read the GT position for that measurement
    position = alf.get_rover_position(EXPERIMENT_ID, CYCLE_ID, CSI_DATASET_PATH)
    if position["position_available"]:
        print(
            f"Rover position for cycle {CYCLE_ID}: "
            f"x={position['rover_x']:.2f}, y={position['rover_y']:.2f}, z={position['rover_z']:.2f}"
        )
    else:
        print(f"No rover position available for cycle {CYCLE_ID}")

    # Collect anchor candidates from all selected microphones
    anchor_candidates = alf.collect_anchor_candidates(setup['selected_mic_positions'], EXPERIMENT_ID, CYCLE_ID, DATASET_PATH, setup['chirp_orig_resampl'], setup['fs_mic'], setup['n_selected_ans'], setup['sumrate_threshold'])

    # Select the top anchors based on the chosen method
    selected_anchors_dict, sort_key = alf.select_top_anchors(anchor_candidates, setup['anchor_selection_method'], setup['n_selected_ans'])
    alf.print_selected_anchors_info(selected_anchors_dict, sort_key, setup['anchor_selection_method'])

    # Apply gain equalisation and plot adjusted waveforms
    selected_anchors_dict = alf.apply_gain_equalisation(selected_anchors_dict, setup['chirp_orig_resampl'])
    alf.plot_adjusted_waveforms_with_chirp(selected_anchors_dict, setup['chirp_orig_resampl'], EXPERIMENT_ID, CYCLE_ID)

    # Compute ground-truth distances (None if position unavailable)
    true_distances = alf.compute_true_distances(position, selected_anchors_dict, setup['selected_mic_positions'])

    # Pulse compression
    pulse_compr_all, LPF_all, corr_index_array = alf.get_corr_with_LPF_curve(setup['chirp_orig_resampl'], selected_anchors_dict)
    alf.plot_pulsecomp_and_lpf(pulse_compr_all, LPF_all, corr_index_array, setup['fs_mic'], selected_anchors_dict, setup['chirp_orig_resampl'], true_distances=true_distances, plot_full_pulse_compression=setup['plot_full_pulse_compression'])

    # Ranging
    distances_meas = alf.compute_ranging(corr_index_array, setup['chirp_orig_resampl'], setup['fs_mic'], setup['v_sound'])
    alf.print_ranging_errors(distances_meas, true_distances, selected_anchors_dict)

    # Determine position estimate based on the ToA ranges (LS)
    # Pass original data structures; LS_positioning performs the required conversion internally.
    position_estimate = alf.LS_positioning(selected_anchors_dict, distances_meas, np.array([4.0, 2.0, 1.5]), selected_mic_positions=setup['selected_mic_positions'])
    print('\n ToA Estimated position: \n', position_estimate)

    # Compute and persist error records using helper functions
    position_metrics = alf.compute_position_error_metrics(position, position_estimate)
    alf.print_position_error_report(position_metrics)

    position_record = alf.build_position_record(EXPERIMENT_ID, CYCLE_ID, PATH_ID, position_metrics)
    ranging_record = alf.build_ranging_record(EXPERIMENT_ID, CYCLE_ID, PATH_ID, position_metrics, distances_meas, true_distances, selected_anchors_dict)

    position_error_file, position_records_count, ranging_error_file, ranging_records_count = alf.save_position_and_ranging_records(Path(__file__).parent, position_record, ranging_record)
    print(f"Saved position record to {position_error_file} (total entries: {position_records_count})")
    print(f"Saved ranging record to {ranging_error_file} (total entries: {ranging_records_count})")

    # Perform GNN positioning

