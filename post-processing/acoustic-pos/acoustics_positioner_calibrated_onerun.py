import argparse
import json
import sys
from pathlib import Path

import numpy as np

import acoustic_local_functions as alf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DATASET_PATH = None      # Set to a specific .nc file to override the newest match.
CSI_DATASET_PATH = None  # Optional: provide a specific CSI dataset path if needed.
WALKS_PICKLE_PATH = PROJECT_ROOT / "walks" / "test.pickle"
CONFIG_PATH = SCRIPT_DIR / "config.json"
OUTPUT_FILE = SCRIPT_DIR / "MB_positions_onerun.npy"

# Manual fallback when no CLI paths are passed.
PATH_IDS_TO_PROCESS = [0]


def _parse_path_ids() -> list[int]:
    parser = argparse.ArgumentParser(
        description="Estimate calibrated MB positions for one or more path IDs and save to MB_positions_onerun.npy"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=int,
        help="Path IDs to process, e.g. 0 3 7",
    )
    args = parser.parse_args()

    if args.paths:
        return args.paths
    if PATH_IDS_TO_PROCESS:
        return PATH_IDS_TO_PROCESS

    raise ValueError(
        "No path IDs provided. Pass them as CLI args (e.g. '...py 0 1 2') "
        "or set PATH_IDS_TO_PROCESS in this file."
    )


def _load_bias_vector() -> np.ndarray:
    if not CONFIG_PATH.exists():
        print(f"Config not found at {CONFIG_PATH}. Using zero bias.")
        return np.zeros(3, dtype=float)

    with CONFIG_PATH.open() as f:
        config = json.load(f)

    raw_bias = config.get("bias_vector_xyz", [0.0, 0.0, 0.0])
    bias = np.asarray(raw_bias, dtype=float).reshape(3)

    print(
        "Using calibration bias vector from config:\n"
        f"  dx = {bias[0]:+.4f} m\n"
        f"  dy = {bias[1]:+.4f} m\n"
        f"  dz = {bias[2]:+.4f} m\n"
        f"  |bias| = {np.linalg.norm(bias):.4f} m"
    )
    return bias


def _apply_bias(position_estimate: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply calibration by subtracting bias from the raw LS estimate."""
    est = np.asarray(position_estimate, dtype=float).reshape(-1)
    if est.size != 3:
        return est
    if np.all(np.isnan(est)):
        return est
    return est - bias


def _run_one_position_estimation(path_id: int, experiment_id: str, cycle_id: int, setup: dict, bias: np.ndarray) -> dict:
    print("\n" + "=" * 80)
    print(f"Path: {path_id} | Experiment: {experiment_id} | Cycle: {cycle_id}")
    print("=" * 80)

    position = alf.get_rover_position(experiment_id, cycle_id, CSI_DATASET_PATH)
    if position["position_available"]:
        print(
            f"Rover position for cycle {cycle_id}: "
            f"x={position['rover_x']:.2f}, y={position['rover_y']:.2f}, z={position['rover_z']:.2f}"
        )
    else:
        print(f"No rover position available for cycle {cycle_id}")

    anchor_candidates = alf.collect_anchor_candidates(
        setup["selected_mic_positions"],
        experiment_id,
        cycle_id,
        DATASET_PATH,
        setup["chirp_orig_resampl"],
        setup["fs_mic"],
        setup["n_selected_ans"],
        setup["sumrate_threshold"],
        n_workers=setup.get("mic_processing_workers"),
    )

    selected_anchors_dict, sort_key = alf.select_top_anchors(
        anchor_candidates,
        setup["anchor_selection_method"],
        setup["n_selected_ans"],
    )

    selected_anchors_dict = alf.apply_gain_equalisation(selected_anchors_dict, setup["chirp_orig_resampl"])
    alf.plot_adjusted_waveforms_with_chirp(selected_anchors_dict, setup["chirp_orig_resampl"], experiment_id, cycle_id)

    true_distances = alf.compute_true_distances(position, selected_anchors_dict, setup["selected_mic_positions"])

    pulse_compr_all, LPF_all, corr_index_array = alf.get_corr_with_LPF_curve(setup["chirp_orig_resampl"], selected_anchors_dict)
    alf.plot_pulsecomp_and_lpf(
        pulse_compr_all,
        LPF_all,
        corr_index_array,
        setup["fs_mic"],
        selected_anchors_dict,
        setup["chirp_orig_resampl"],
        true_distances=true_distances,
        plot_full_pulse_compression=setup["plot_full_pulse_compression"],
    )

    distances_meas = alf.compute_ranging(corr_index_array, setup["chirp_orig_resampl"], setup["fs_mic"], setup["v_sound"])
    alf.print_ranging_errors(distances_meas, true_distances, selected_anchors_dict)

    raw_position_estimate = alf.LS_positioning(
        selected_anchors_dict,
        distances_meas,
        np.array([4.0, 2.0, 1.5]),
        selected_mic_positions=setup["selected_mic_positions"],
    )

    calibrated_position_estimate = _apply_bias(raw_position_estimate, bias)

    if np.all(np.isnan(raw_position_estimate)):
        print("\nRaw ToA estimate       : [NaN, NaN, NaN] (INVALID - insufficient data)")
        print("Calibrated ToA estimate: [NaN, NaN, NaN]")
    else:
        print(
            f"\nRaw ToA estimate       : "
            f"[{raw_position_estimate[0]:.4f}, {raw_position_estimate[1]:.4f}, {raw_position_estimate[2]:.4f}]"
        )
        print(
            f"Calibrated ToA estimate: "
            f"[{calibrated_position_estimate[0]:.4f}, {calibrated_position_estimate[1]:.4f}, {calibrated_position_estimate[2]:.4f}]"
        )

    position_metrics = alf.compute_position_error_metrics(position, calibrated_position_estimate)
    alf.print_position_error_report(position_metrics)

    position_record = alf.build_position_record(experiment_id, cycle_id, path_id, position_metrics)
    position_record["estimated_position_xyz_uncalibrated"] = np.asarray(raw_position_estimate, dtype=float).tolist()
    position_record["applied_bias_vector_xyz"] = bias.tolist()

    return position_record


if __name__ == "__main__":
    path_ids_to_process = _parse_path_ids()
    runs_to_process = alf.load_runs_to_process(
        path_ids_to_process,
        walks_pickle_path=WALKS_PICKLE_PATH,
        csi_dataset_path=CSI_DATASET_PATH,
    )

    setup = alf.load_experiment_setup()
    bias = _load_bias_vector()

    print(f"\nSpeed of sound: {setup['v_sound']} m/s")
    print(f"Processing {len(setup['selected_mic_positions'])} microphones...")
    workers_msg = setup.get("mic_processing_workers")
    if workers_msg is None:
        print("Microphone processing workers: auto")
    else:
        print(f"Microphone processing workers: {workers_msg}")

    one_run_records = []
    for run_idx, (path_id, experiment_id, cycle_id) in enumerate(runs_to_process, start=1):
        print(f"\nRun {run_idx}/{len(runs_to_process)}")
        record = _run_one_position_estimation(path_id, experiment_id, cycle_id, setup, bias)
        one_run_records.append(record)

    np.save(OUTPUT_FILE, np.array(one_run_records, dtype=object), allow_pickle=True)
    print(f"\nSaved {len(one_run_records)} calibrated position records to: {OUTPUT_FILE}")
