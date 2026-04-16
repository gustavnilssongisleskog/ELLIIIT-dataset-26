import sys
from pathlib import Path
import acoustic_local_functions as alf
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DATASET_PATH = None      # Set to a specific .nc file to override the newest match.
CSI_DATASET_PATH = None  # Optional: provide a specific CSI dataset path if needed.
WALKS_PICKLE_PATH = PROJECT_ROOT / "walks" / "test.pickle"

# Auto-detect all available path IDs from the pickle file
PATH_IDS_TO_PROCESS = alf.get_all_path_ids(WALKS_PICKLE_PATH)
# Or manually override: PATH_IDS_TO_PROCESS = [0, 1, 2]  # specific paths


if __name__ == "__main__":

    runs_to_process = alf.load_runs_to_process(PATH_IDS_TO_PROCESS, walks_pickle_path=WALKS_PICKLE_PATH, csi_dataset_path=CSI_DATASET_PATH)

    # Load all setup values (config, microphone positions, chirp)
    setup = alf.load_experiment_setup()
    print('\nSpeed of sound: ', setup['v_sound'], ' m/s\n')
    print(f"Processing {len(setup['selected_mic_positions'])} microphones...")
    workers_msg = setup.get('mic_processing_workers')
    if workers_msg is None:
        print("Microphone processing workers: auto")
    else:
        print(f"Microphone processing workers: {workers_msg}")

    for run_idx, (PATH_ID, EXPERIMENT_ID, CYCLE_ID) in enumerate(runs_to_process, start=1):
        print("\n" + "=" * 80)
        print(
            f"Run {run_idx}/{len(runs_to_process)} | "
            f"Path: {PATH_ID} | Experiment: {EXPERIMENT_ID} | Cycle: {CYCLE_ID}"
        )
        print("=" * 80)

        # Read the GT position for this measurement
        position = alf.get_rover_position(EXPERIMENT_ID, CYCLE_ID, CSI_DATASET_PATH)
        if position["position_available"]:
            print(
                f"Rover position for cycle {CYCLE_ID}: "
                f"x={position['rover_x']:.2f}, y={position['rover_y']:.2f}, z={position['rover_z']:.2f}"
            )
        else:
            print(f"No rover position available for cycle {CYCLE_ID}")

        # Collect anchor candidates from all selected microphones
        anchor_candidates = alf.collect_anchor_candidates(
            setup['selected_mic_positions'],
            EXPERIMENT_ID,
            CYCLE_ID,
            DATASET_PATH,
            setup['chirp_orig_resampl'],
            setup['fs_mic'],
            setup['n_selected_ans'],
            setup['sumrate_threshold'],
            n_workers=setup.get('mic_processing_workers')
        )

        # Select the top anchors based on the chosen method
        selected_anchors_dict, sort_key = alf.select_top_anchors(
            anchor_candidates,
            setup['anchor_selection_method'],
            setup['n_selected_ans']
        )
        # alf.print_selected_anchors_info(selected_anchors_dict, sort_key, setup['anchor_selection_method'])

        # Apply gain equalisation and plot adjusted waveforms
        selected_anchors_dict = alf.apply_gain_equalisation(selected_anchors_dict, setup['chirp_orig_resampl'])
        alf.plot_adjusted_waveforms_with_chirp(selected_anchors_dict, setup['chirp_orig_resampl'], EXPERIMENT_ID, CYCLE_ID)

        # Compute ground-truth distances (None if position unavailable)
        true_distances = alf.compute_true_distances(position, selected_anchors_dict, setup['selected_mic_positions'])

        # Pulse compression
        pulse_compr_all, LPF_all, corr_index_array = alf.get_corr_with_LPF_curve(setup['chirp_orig_resampl'], selected_anchors_dict)
        alf.plot_pulsecomp_and_lpf(
            pulse_compr_all,
            LPF_all,
            corr_index_array,
            setup['fs_mic'],
            selected_anchors_dict,
            setup['chirp_orig_resampl'],
            true_distances=true_distances,
            plot_full_pulse_compression=setup['plot_full_pulse_compression']
        )

        # Ranging
        distances_meas = alf.compute_ranging(corr_index_array, setup['chirp_orig_resampl'], setup['fs_mic'], setup['v_sound'])
        alf.print_ranging_errors(distances_meas, true_distances, selected_anchors_dict)

        # Determine position estimate based on the ToA ranges (LS)
        # Pass original data structures; LS_positioning performs the required conversion internally.
        position_estimate = alf.LS_positioning(
            selected_anchors_dict,
            distances_meas,
            np.array([4.0, 2.0, 1.5]),
            selected_mic_positions=setup['selected_mic_positions']
        )
        
        # Format position output
        if np.all(np.isnan(position_estimate)):
            print('\nToA Estimated position: [NaN, NaN, NaN] (INVALID - insufficient data)')
        else:
            print(f'\nToA Estimated position: [{position_estimate[0]:.4f}, {position_estimate[1]:.4f}, {position_estimate[2]:.4f}]')

        # Compute and persist error records using helper functions
        position_metrics = alf.compute_position_error_metrics(position, position_estimate)
        alf.print_position_error_report(position_metrics)

        position_record = alf.build_position_record(EXPERIMENT_ID, CYCLE_ID, PATH_ID, position_metrics)
        ranging_record = alf.build_ranging_record(EXPERIMENT_ID, CYCLE_ID, PATH_ID, position_metrics, distances_meas, true_distances, selected_anchors_dict)

        position_error_file, position_records_count, ranging_error_file, ranging_records_count = alf.save_position_and_ranging_records(
            Path(__file__).parent,
            position_record,
            ranging_record
        )
        print(f"Saved position record to {position_error_file} (total entries: {position_records_count})")
        print(f"Saved ranging record to {ranging_error_file} (total entries: {ranging_records_count})")

    # ===== Compute calibration offset from all measurements =====
    alf.compute_calibration_offset(SCRIPT_DIR)

    # Perform GNN positioning
    #TODO later train on train.Pickle 'resting' flagged values, test on test.pickle 'resting' flagged values, and compare to with MB method on ceveral test set paths
