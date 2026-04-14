import json
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import acoustic_local_functions as alf


DATASET_PATH = None  # Optional: set a specific acoustic .nc path
CSI_DATASET_PATH = None  # Optional: set a specific CSI .nc path

# Positions to evaluate together (all errors are aggregated into one final graph).
# Format examples:
#   {"experiment_id": "EXP008", "cycle_ids": [20, 21, 22]}
#   {"experiment_id": "EXP010", "cycle_ids": "all"}
EVAL_TARGETS = [
    {"experiment_id": "EXP008", "cycle_ids": "all"},
]

# Sweep settings
FACTOR_MIN = 0.15
FACTOR_MAX = 0.40
FACTOR_STEP = 0.01

# Speed settings
MAX_RANDOM_POSITIONS_PER_EXPERIMENT = 100
RANDOM_SEED = 42
USE_MULTIPROCESSING = True
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)

# Data validity settings
# Keep only cycle IDs with valid rover position in CSI dataset.
# If set, keep only the first N valid cycle IDs per experiment (ordered by cycle_id).
MAX_VALID_CYCLE_IDS_PER_EXPERIMENT = 200


script_dir = Path(__file__).parent
config_path = script_dir / "config.json"

with open(config_path) as json_file:
    config = json.load(json_file)

v_sound = 20 * np.sqrt(273 + config["temperature"])
fs_mic = config["fs_mic"]
n_selected_ans = config["number_of_selected_anchors"]
anchor_selection_method = config["anchor_selection_method"]
sumrate_threshold = config["sumrate_threshold"]

if config["use_all_mics"]:
    microphone_labels = config["all_mics"]
else:
    microphone_labels = config["used_mics"]


def build_selected_anchors(experiment_id, cycle_id, show_progress=True):
    microphone_positions = alf.load_microphone_positions()
    selected_mic_positions = alf.get_selected_mic_positions(microphone_positions, microphone_labels)

    fs_source, chirp_orig, _ = alf.read_transmit_chirp()
    chirp_orig_resampl = alf.resample_chirp(fs_source, fs_mic, chirp_orig) * 10

    position = alf.get_rover_position(experiment_id, cycle_id, CSI_DATASET_PATH)
    if not position["position_available"]:
        raise ValueError(
            f"No rover position available for {experiment_id} cycle {cycle_id}. "
            "Ground truth is required for tuning."
        )

    _, dataset_path = alf.open_acoustic_dataset(experiment_id, DATASET_PATH)
    print(f"Using acoustic dataset for {experiment_id}: {dataset_path}")

    anchor_candidates = []
    mic_iter = selected_mic_positions.items()
    if show_progress:
        mic_iter = tqdm(
            mic_iter,
            total=len(selected_mic_positions),
            desc=f"Collecting anchor candidates ({experiment_id} cycle {cycle_id})",
            unit="mic",
        )
    for mic_label, _ in mic_iter:
        _, waveform_values, _ = alf.get_acoustic_waveform(
            experiment_id, cycle_id, mic_label, DATASET_PATH
        )
        waveform_filtered = alf.butter_highpass_filter(waveform_values, 15000, fs_mic, 10)
        anchor_info = alf.select_anchors(
            waveform_filtered,
            mic_label,
            experiment_id,
            cycle_id,
            n_selected_ans,
            sumrate_threshold,
        )
        anchor_candidates.append(anchor_info)

    selected_anchors_dict, _ = alf.select_top_anchors(
        anchor_candidates, anchor_selection_method, n_selected_ans
    )

    signals = np.array([anchor["waveform_values"] for anchor in selected_anchors_dict.values()])
    adjusted_signals = alf.gain_equaliser(chirp_orig_resampl, signals, 8000, 10000)

    for i, mic_label in enumerate(selected_anchors_dict.keys()):
        selected_anchors_dict[mic_label]["adjusted_waveform"] = adjusted_signals[i]

    rover_xyz = np.array(
        [position["rover_x"], position["rover_y"], position["rover_z"]], dtype=float
    )
    true_distances = {}
    for mic_label in selected_anchors_dict.keys():
        mic_xyz = np.asarray(selected_mic_positions[mic_label], dtype=float)
        true_distances[mic_label] = float(np.linalg.norm(mic_xyz - rover_xyz))

    return chirp_orig_resampl, selected_anchors_dict, true_distances


def precompute_corr_and_lpf(chirp_tx, selected_anchors_dict, show_progress=True):
    rx_audio_full = np.array(
        [anchor["adjusted_waveform"] for anchor in selected_anchors_dict.values()]
    )
    rx_audio_amp, n_wake_up_samples_eff, wake_up_at_sample = alf.get_wake_up_part(
        rx_audio_full, chirp_tx, fs_mic
    )

    corr_vals = []
    lpf_vals = []
    rx_iter = range(rx_audio_amp.shape[0])
    if show_progress:
        rx_iter = tqdm(rx_iter, desc="Precomputing pulse compression", unit="mic")
    for rx_idx in rx_iter:
        corr_val = alf.norm_correlate(chirp_tx, rx_audio_amp, rx_idx)
        lpf_val = alf.LPF(corr_val, "lowpass", 10, fs_mic / 26, fs_mic)
        corr_vals.append(corr_val)
        lpf_vals.append(lpf_val)

    return corr_vals, lpf_vals, n_wake_up_samples_eff, wake_up_at_sample


def evaluate_factor(
    factor,
    corr_vals,
    lpf_vals,
    anchor_labels,
    true_distances,
    n_wake_up_samples_eff,
    wake_up_at_sample,
):
    alf.config["peak_prominence_factor"] = float(factor)

    abs_errors = []
    failed_mics = 0

    for mic_label, corr_val, lpf_val in zip(anchor_labels, corr_vals, lpf_vals):
        try:
            corr_idx = float(alf.get_peak_prom_index(lpf_val, corr_val))
        except Exception:
            failed_mics += 1
            continue

        eff_start_samp_chirp = (corr_idx + 1) - n_wake_up_samples_eff
        delta_sample = wake_up_at_sample - eff_start_samp_chirp
        measured_distance = (delta_sample / fs_mic) * v_sound

        gt_distance = true_distances[mic_label]
        abs_errors.append(abs(measured_distance - gt_distance))

    return abs_errors, failed_mics


def flatten_targets(eval_targets):
    flat = []
    for target in eval_targets:
        exp_id = target["experiment_id"]
        for cycle_id in target["cycle_ids"]:
            flat.append((exp_id, int(cycle_id)))
    return flat


def _list_cycle_ids_for_experiment(experiment_id):
    csi_ds, csi_path = alf.open_csi_dataset(experiment_id, CSI_DATASET_PATH)
    try:
        exp_slice = csi_ds.sel(experiment_id=experiment_id)
        if "cycle_id" in exp_slice.coords:
            cycle_ids = np.asarray(exp_slice.coords["cycle_id"].values).astype(int)
        elif "cycle_id" in exp_slice:
            cycle_ids = np.asarray(exp_slice["cycle_id"].values).astype(int)
        else:
            raise KeyError("cycle_id coordinate/variable not found in CSI dataset")

        if "position_available" not in exp_slice:
            raise KeyError("position_available variable not found in CSI dataset")
        valid_mask = np.asarray(exp_slice["position_available"].values > 0, dtype=bool)

        # Guard against NaN coordinates being flagged as available.
        for coord_name in ("rover_x", "rover_y", "rover_z"):
            if coord_name in exp_slice:
                valid_mask &= np.isfinite(np.asarray(exp_slice[coord_name].values, dtype=float))

        valid_cycle_ids = sorted(int(c) for c, ok in zip(cycle_ids.tolist(), valid_mask.tolist()) if ok)
        total_cycle_ids = sorted({int(c) for c in cycle_ids.tolist()})

        if MAX_VALID_CYCLE_IDS_PER_EXPERIMENT is not None and MAX_VALID_CYCLE_IDS_PER_EXPERIMENT > 0:
            valid_cycle_ids = valid_cycle_ids[:MAX_VALID_CYCLE_IDS_PER_EXPERIMENT]

        print(
            f"{experiment_id}: total cycles={len(total_cycle_ids)}, "
            f"valid cycles={int(np.sum(valid_mask))}, used valid cycles={len(valid_cycle_ids)} "
            f"from {csi_path.name}"
        )
        return valid_cycle_ids
    finally:
        csi_ds.close()


def expand_eval_targets(eval_targets):
    expanded = []
    for target in eval_targets:
        experiment_id = target["experiment_id"]
        cycle_ids = target["cycle_ids"]

        if isinstance(cycle_ids, str) and cycle_ids.lower() == "all":
            cycle_ids = _list_cycle_ids_for_experiment(experiment_id)

        for cycle_id in cycle_ids:
            expanded.append((experiment_id, int(cycle_id)))

    # de-duplicate while preserving order
    return list(dict.fromkeys(expanded))


def sample_targets(flat_targets, max_per_experiment, seed):
    if max_per_experiment is None or max_per_experiment <= 0:
        return flat_targets

    grouped = defaultdict(list)
    for exp_id, cycle_id in flat_targets:
        grouped[exp_id].append(int(cycle_id))

    rng = np.random.default_rng(seed)
    sampled = []
    for exp_id, cycles in grouped.items():
        unique_cycles = sorted(set(cycles))
        n_take = min(max_per_experiment, len(unique_cycles))
        chosen = rng.choice(unique_cycles, size=n_take, replace=False)
        for cycle_id in sorted(int(c) for c in chosen.tolist()):
            sampled.append((exp_id, cycle_id))

    return sampled


def _prepare_single_target(experiment_id, cycle_id):
    chirp_tx, selected_anchors_dict, true_distances = build_selected_anchors(
        experiment_id, cycle_id, show_progress=False
    )
    corr_vals, lpf_vals, n_wake_up_samples_eff, wake_up_at_sample = precompute_corr_and_lpf(
        chirp_tx, selected_anchors_dict, show_progress=False
    )
    return {
        "experiment_id": experiment_id,
        "cycle_id": cycle_id,
        "anchor_labels": list(selected_anchors_dict.keys()),
        "true_distances": true_distances,
        "corr_vals": corr_vals,
        "lpf_vals": lpf_vals,
        "n_wake_up_samples_eff": n_wake_up_samples_eff,
        "wake_up_at_sample": wake_up_at_sample,
    }


def main():
    flat_targets = expand_eval_targets(EVAL_TARGETS)
    if not flat_targets:
        raise ValueError("EVAL_TARGETS is empty. Add at least one experiment/cycle target.")

    full_target_count = len(flat_targets)
    flat_targets = sample_targets(
        flat_targets,
        max_per_experiment=MAX_RANDOM_POSITIONS_PER_EXPERIMENT,
        seed=RANDOM_SEED,
    )

    print(
        f"Preparing {len(flat_targets)} sampled target(s) "
        f"(from {full_target_count} total) for aggregate sweep..."
    )

    prepared_targets = []
    skipped_targets = []

    if USE_MULTIPROCESSING and len(flat_targets) > 1:
        print(f"Using multiprocessing with {N_WORKERS} workers for target preparation...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            future_map = {
                executor.submit(_prepare_single_target, exp_id, cycle_id): (exp_id, cycle_id)
                for exp_id, cycle_id in flat_targets
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Preparing targets", unit="target"):
                exp_id, cycle_id = future_map[future]
                try:
                    prepared_targets.append(future.result())
                except Exception as exc:
                    skipped_targets.append((exp_id, cycle_id, str(exc)))
    else:
        for experiment_id, cycle_id in tqdm(flat_targets, desc="Preparing targets", unit="target"):
            try:
                prepared_targets.append(_prepare_single_target(experiment_id, cycle_id))
            except Exception as exc:
                skipped_targets.append((experiment_id, cycle_id, str(exc)))

    if not prepared_targets:
        raise RuntimeError("No valid targets could be prepared. Check EVAL_TARGETS and dataset availability.")

    if skipped_targets:
        print("\nSkipped targets:")
        for exp_id, cyc_id, reason in skipped_targets:
            print(f"  - {exp_id} cycle {cyc_id}: {reason}")

    print(f"\nPrepared targets successfully: {len(prepared_targets)}")
    print(f"Random sampling per experiment: up to {MAX_RANDOM_POSITIONS_PER_EXPERIMENT} positions")
    print(f"Random seed: {RANDOM_SEED}")
    if prepared_targets:
        print("First prepared targets:")
        for prepared in prepared_targets[:10]:
            print(f"  - {prepared['experiment_id']} cycle {prepared['cycle_id']}")

    factors = np.arange(FACTOR_MIN, FACTOR_MAX + 1e-12, FACTOR_STEP)
    mean_errors = []
    p95_errors = []
    failed_counts = []
    valid_counts = []

    for factor in tqdm(factors, desc="Sweeping peak_prominence_factor", unit="val"):
        factor_errors = []
        total_failed_mics = 0

        for prepared in prepared_targets:
            abs_errors, failed_mics = evaluate_factor(
                factor,
                prepared["corr_vals"],
                prepared["lpf_vals"],
                prepared["anchor_labels"],
                prepared["true_distances"],
                prepared["n_wake_up_samples_eff"],
                prepared["wake_up_at_sample"],
            )
            factor_errors.extend(abs_errors)
            total_failed_mics += failed_mics

        if factor_errors:
            mean_errors.append(float(np.mean(factor_errors)))
            p95_errors.append(float(np.percentile(factor_errors, 95)))
        else:
            mean_errors.append(np.nan)
            p95_errors.append(np.nan)
        failed_counts.append(total_failed_mics)
        valid_counts.append(len(factor_errors))

    mean_errors = np.array(mean_errors, dtype=float)
    p95_errors = np.array(p95_errors, dtype=float)
    failed_counts = np.array(failed_counts, dtype=int)
    valid_counts = np.array(valid_counts, dtype=int)

    valid = np.isfinite(mean_errors) & np.isfinite(p95_errors)
    if not np.any(valid):
        raise RuntimeError(
            "All factor evaluations failed to find valid peaks. "
            "Try lowering FACTOR_MIN or improving SNR/filter settings."
        )

    # Primary criterion: lowest P95; tie-breaker: lowest mean error.
    valid_idx = np.where(valid)[0]
    best_idx = min(valid_idx, key=lambda i: (p95_errors[i], mean_errors[i]))

    best_factor = float(factors[best_idx])
    print("\nBest factor (min P95, tie-break by mean):", f"{best_factor:.3f}")
    print(f"Prepared targets used: {len(prepared_targets)}")
    print(f"Mean absolute ranging error at best factor (m): {mean_errors[best_idx]:.3f}")
    print(f"P95 absolute ranging error at best factor (m): {p95_errors[best_idx]:.3f}")
    print(f"Valid microphone measurements at best factor: {int(valid_counts[best_idx])}")
    print(f"Failed microphone measurements at best factor: {int(failed_counts[best_idx])}")

    figs_dir = script_dir / "Figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    targets_label = "multi_targets"
    plot_path = figs_dir / f"peak_prominence_sweep_{targets_label}.png"

    plt.figure(figsize=(10, 5.5))
    plt.plot(factors, mean_errors, marker="o", linewidth=1.5, label="Mean abs error")
    plt.plot(factors, p95_errors, marker="s", linewidth=1.5, label="P95 abs error")
    plt.axvline(best_factor, color="black", linestyle="--", linewidth=1.0, label=f"Best={best_factor:.3f}")
    plt.xlabel("peak_prominence_factor")
    plt.ylabel("Absolute ranging error (m)")
    plt.title(f"Peak prominence sweep - aggregate over {len(prepared_targets)} targets")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {plot_path}")

    csv_path = figs_dir / f"peak_prominence_sweep_{targets_label}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("factor,mean_abs_error_m,p95_abs_error_m,valid_measurements,failed_mics\n")
        for factor, mean_err, p95_err, valid_n, failed in zip(
            factors, mean_errors, p95_errors, valid_counts, failed_counts
        ):
            f.write(f"{factor:.5f},{mean_err:.8f},{p95_err:.8f},{int(valid_n)},{int(failed)}\n")

    print(f"Saved sweep data to: {csv_path}")

    targets_path = figs_dir / f"peak_prominence_sweep_{targets_label}_targets.txt"
    with open(targets_path, "w", encoding="utf-8") as f:
        f.write("Prepared targets used for aggregation:\n")
        for prepared in prepared_targets:
            f.write(f"{prepared['experiment_id']},cycle={prepared['cycle_id']}\n")
    print(f"Saved target list to: {targets_path}")


if __name__ == "__main__":
    main()
