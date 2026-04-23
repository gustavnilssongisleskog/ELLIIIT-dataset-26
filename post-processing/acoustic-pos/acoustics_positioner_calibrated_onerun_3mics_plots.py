import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go
from scipy.signal import find_peaks, peak_prominences

import acoustic_local_functions as alf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WALKS_PICKLE_PATH = PROJECT_ROOT / "walks" / "test.pickle"
DATASET_PATH = None
CSI_DATASET_PATH = None
N_MICROPHONES = 3

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one-run acoustic diagnostic plots for 3 microphones."
    )
    parser.add_argument(
        "path_id",
        nargs="?",
        type=int,
        default=0,
        help="Path ID from walks/test.pickle (default: 0).",
    )
    parser.add_argument(
        "--quality-examples",
        action="store_true",
        help="Generate 3 LPF examples (close/ok/poor) based on prediction vs theoretical distance error.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="Maximum number of resting-position runs to score when --quality-examples is used (default: 40).",
    )
    return parser.parse_args()


def _collect_raw_and_filtered_rx(
    selected_anchors_dict: dict,
    experiment_id: str,
    cycle_id: int,
    fs_mic: float,
) -> dict[str, dict[str, np.ndarray]]:
    rx_signals: dict[str, dict[str, np.ndarray]] = {}
    for mic_label in selected_anchors_dict.keys():
        _, rx_raw, _ = alf.get_acoustic_waveform(experiment_id, cycle_id, mic_label, DATASET_PATH)
        rx_filtered = alf.butter_highpass_filter(rx_raw, 15000, fs_mic, 10)
        rx_signals[mic_label] = {
            "raw": np.asarray(rx_raw, dtype=float),
            "filtered": np.asarray(rx_filtered, dtype=float),
            "adjusted": np.asarray(selected_anchors_dict[mic_label]["adjusted_waveform"], dtype=float),
        }
    return rx_signals


def _build_received_and_transmitted_plot(
    tx_raw: np.ndarray,
    tx_filtered: np.ndarray,
    rx_signals: dict[str, dict[str, np.ndarray]],
    experiment_id: str,
    cycle_id: int,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(tx_raw.size),
            y=tx_raw,
            mode="lines",
            name="TX raw",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(tx_filtered.size),
            y=tx_filtered,
            mode="lines",
            name="TX filtered",
            line=dict(color="#17becf", width=2, dash="dash"),
        )
    )

    mic_label = next(iter(rx_signals.keys()))
    signals = rx_signals[mic_label]
    fig.add_trace(
        go.Scatter(
            x=np.arange(signals["raw"].size),
            y=signals["raw"],
            mode="lines",
            name=f"RX raw {mic_label}",
            line=dict(color="#d62728", width=1.5),
            opacity=0.55,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(signals["adjusted"].size),
            y=signals["adjusted"],
            mode="lines",
            name=f"RX filtered+adjusted {mic_label}",
            line=dict(color="#2ca02c", width=2),
            opacity=0.9,
        )
    )

    fig.update_layout(
        title=(
            f"Received and transmitted acoustic data before/after filtering - "
            f"{experiment_id} cycle {cycle_id}"
        ),
        xaxis_title="Sample index",
        yaxis_title="Amplitude",
        legend_title="Signals",
        template="plotly_white",
    )
    return fig


def _compute_x_signal_meters(x_len: int, tx: np.ndarray, fs: float) -> np.ndarray:
    # Use the exact same conversion as the main ranging pipeline.
    temp = alf.config["temperature"]
    v_sound = 20 * np.sqrt(273 + temp)
    corr_idx_array = np.arange(0, x_len, 1, dtype=float)
    x_signal = alf.compute_ranging(corr_idx_array, tx, fs, v_sound).reshape(-1)
    return x_signal


def _build_pulse_compression_plot(
    pulse_compr_all: np.ndarray,
    lpf_all: np.ndarray,
    corr_index_array: np.ndarray,
    selected_anchors_dict: dict,
    tx: np.ndarray,
    fs: float,
    true_distances: dict[str, float] | None,
) -> go.Figure:
    x_signal = _compute_x_signal_meters(pulse_compr_all.shape[1], tx, fs)

    fig = go.Figure()
    colors = plotly.colors.qualitative.Plotly
    prominence_threshold = float(alf.config["peak_prominence_factor"])

    for i, mic_label in enumerate(selected_anchors_dict.keys()):
        color = colors[i % len(colors)]
        pulse = pulse_compr_all[i, :]
        lpf = lpf_all[i, :]

        fig.add_trace(
            go.Scatter(
                x=x_signal,
                y=pulse,
                mode="lines",
                name=f"Pulse comp {mic_label}",
                line=dict(color=color, width=1),
                opacity=0.45,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_signal,
                y=lpf,
                mode="lines",
                name=f"LPF {mic_label}",
                line=dict(color=color, width=2, dash="dash"),
            )
        )

        peaks, _ = find_peaks(lpf)
        if peaks.size > 0:
            prom_vals = peak_prominences(lpf, peaks)[0]
            above_mask = prom_vals > prominence_threshold
            if np.any(above_mask):
                fig.add_trace(
                    go.Scatter(
                        x=x_signal[peaks[above_mask]],
                        y=lpf[peaks[above_mask]],
                        mode="markers",
                        name=f"Peaks>{prominence_threshold:.2f} {mic_label}",
                        showlegend=False,
                        marker=dict(color=color, size=7, symbol="circle-open"),
                        hovertemplate=(
                            "Mic="
                            + mic_label
                            + "<br>Distance=%{x:.3f} m"
                            + "<br>LPF=%{y:.3f}"
                            + "<br>Prominence=%{customdata:.4f}<extra></extra>"
                        ),
                        customdata=prom_vals[above_mask],
                    )
                )

        selected_idx_float = corr_index_array[i]
        if np.isnan(selected_idx_float):
            continue

        selected_idx = int(selected_idx_float)
        if selected_idx < 0 or selected_idx >= pulse.size:
            continue

        nearest_prom = np.nan
        if peaks.size > 0:
            nearest_peak_idx = int(np.argmin(np.abs(peaks - selected_idx)))
            nearest_prom = float(peak_prominences(lpf, peaks)[0][nearest_peak_idx])

        fig.add_trace(
            go.Scatter(
                x=[x_signal[selected_idx]],
                y=[lpf[selected_idx]],
                mode="markers",
                name=f"Selected peak {mic_label}",
                marker=dict(color="red", size=10, symbol="diamond"),
            )
        )
        label_y = min(1.28, float(lpf[selected_idx]) + 0.12)
        fig.add_annotation(
            x=float(x_signal[selected_idx]),
            y=label_y,
            text=f"{mic_label} prom={nearest_prom:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-18,
            font=dict(color="red"),
            bgcolor="rgba(255,255,255,0.7)",
        )
        fig.add_vline(
            x=float(x_signal[selected_idx]),
            line_color="red",
            line_width=1,
            line_dash="dot",
        )

        if true_distances is not None and mic_label in true_distances:
            dist_theory = float(true_distances[mic_label])
            if dist_theory >= 0:
                fig.add_vline(
                    x=dist_theory,
                    line_color=color,
                    line_width=1,
                    line_dash="dash",
                    annotation_text=f"GT {mic_label}",
                    annotation_position="top left",
                )

    fig.update_layout(
        title="Pulse compression and LPF values with selected peaks",
        xaxis_title="Ranging distance (m)",
        yaxis_title="Normalized correlation value",
        yaxis=dict(range=[0, 1.3]),
        template="plotly_white",
    )
    fig.update_xaxes(range=[0, float(np.max(x_signal))])
    return fig


def _build_single_mic_lpf_plot(
    pulse_compr: np.ndarray,
    lpf: np.ndarray,
    selected_idx: int | None,
    mic_label: str,
    tx: np.ndarray,
    fs: float,
    true_distance: float | None,
    quality_label: str,
    quality_error: float | None,
) -> go.Figure:
    """Build a single-microphone LPF quality-example plot.

    selected_idx must come from alf.get_corr_with_LPF_curve (original peak prominence logic).
    """
    x_signal = _compute_x_signal_meters(pulse_compr.size, tx, fs)
    color = "#1f77b4"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_signal, y=pulse_compr,
            mode="lines", name="Pulse compression",
            line=dict(color=color, width=1), opacity=0.4,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_signal, y=lpf,
            mode="lines", name="LPF",
            line=dict(color=color, width=2.5),
        )
    )

    if selected_idx is not None and 0 <= selected_idx < lpf.size:
        dist_meas = float(x_signal[selected_idx])
        corr_val = float(lpf[selected_idx])
        fig.add_trace(
            go.Scatter(
                x=[dist_meas], y=[lpf[selected_idx]],
                mode="markers",
                name="Selected peak",
                marker=dict(color="red", size=11, symbol="diamond"),
            )
        )
        label_y = min(1.30, corr_val + 0.12)
        fig.add_annotation(
            x=dist_meas,
            y=label_y,
            text=f"Selected: {dist_meas:.3f} m (corr={corr_val:.3f})",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-18,
            font=dict(color="red"),
            bgcolor="rgba(255,255,255,0.7)",
        )
        fig.add_vline(
            x=dist_meas, line_color="red", line_width=1.5, line_dash="dot",
            annotation_text="Predicted",
            annotation_position="top right",
        )

    if true_distance is not None and true_distance >= 0:
        fig.add_vline(
            x=true_distance, line_color="green", line_width=1.5, line_dash="dash",
            annotation_text="GT",
            annotation_position="top left",
        )

    error_str = f"  |error|={quality_error:.3f} m" if quality_error is not None else ""
    fig.update_layout(
        title=f"LPF example ({quality_label}) — mic {mic_label}{error_str}",
        xaxis_title="Ranging distance (m)",
        yaxis_title="Normalized correlation value",
        yaxis=dict(range=[0, 1.35]),
        template="plotly_white",
        legend_title="Signals",
    )
    fig.update_xaxes(range=[0, float(np.max(x_signal))])
    return fig


def _process_run(
    path_id: int,
    experiment_id: str,
    cycle_id: int,
    setup: dict,
    all_mics_for_scoring: bool = False,
) -> tuple[dict, list[dict]] | None:
    """Process one position and return (run_meta, per_mic_candidates).

    run_meta contains the full-3-mic figure for the normal single-run output.
    per_mic_candidates is a list of scored individual-mic entries used by quality-examples mode.
    When all_mics_for_scoring=True all available microphones are processed so the quality-examples
    pool is as large as possible (91 mics instead of top-3).
    When all_mics_for_scoring=True ALL microphones are scored directly based on
    range estimation quality — signal-strength ranking is NOT used for scoring.
    """
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
    # For normal single-run we always pick top-N_MICROPHONES for the figure.
    # For quality-examples we also build the top-3 figure, but score all mics.
    selected_anchors_dict, _ = alf.select_top_anchors(
        anchor_candidates,
        setup["anchor_selection_method"],
        N_MICROPHONES,
    )
    if len(selected_anchors_dict) < N_MICROPHONES:
        return None

    # For quality-examples scoring: use ALL candidates directly, no signal-strength filter.
    if all_mics_for_scoring:
        scoring_anchors_dict = {c["microphone_label"]: c for c in anchor_candidates}
    else:
        scoring_anchors_dict = selected_anchors_dict

    selected_anchors_dict = alf.apply_gain_equalisation(
        selected_anchors_dict,
        setup["chirp_orig_resampl"],
    )

    position = alf.get_rover_position(experiment_id, cycle_id, CSI_DATASET_PATH)
    # true_distances needs all mics in the scoring dict; use the full mic-positions map.
    true_distances = alf.compute_true_distances(
        position,
        scoring_anchors_dict,
        setup["selected_mic_positions"],
    )
    if true_distances is None:
        return None

    tx_raw = np.asarray(setup["chirp_orig_resampl"], dtype=float)

    # Compute pulse compression for the top-3 selected mics (used for the figure)
    # Compute pulse compression for the top-3 selected mics (used for the 3-mic figure).
    pulse_compr_all, lpf_all, corr_index_array = alf.get_corr_with_LPF_curve(
        tx_raw,
        selected_anchors_dict,
    )

    # Compute pulse compression for all scoring mics (may be same as above when not quality mode)
    # Compute pulse compression for the scoring pool.
    if all_mics_for_scoring:
        # Apply gain equalisation to ALL mics independently before computing correlations.
        scoring_anchors_dict = alf.apply_gain_equalisation(
            scoring_anchors_dict, setup["chirp_orig_resampl"]
        )
        pulse_compr_score, lpf_score, corr_idx_score = alf.get_corr_with_LPF_curve(
            tx_raw, scoring_anchors_dict
        )
    else:
        pulse_compr_score, lpf_score, corr_idx_score = pulse_compr_all, lpf_all, corr_index_array

    x_signal = _compute_x_signal_meters(pulse_compr_score.shape[1], tx_raw, setup["fs_mic"])
    mic_labels = list(scoring_anchors_dict.keys())

    # Score per microphone individually
    per_mic_candidates: list[dict] = []
    mean_errors: list[float] = []
    for i, mic_label in enumerate(mic_labels):
        idx = corr_idx_score[i]
        if np.isnan(idx):
            continue
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= x_signal.size:
            continue
        if mic_label not in true_distances:
            continue
        pred_distance = float(x_signal[idx_i])
        error = abs(pred_distance - float(true_distances[mic_label]))
        mean_errors.append(error)
        per_mic_candidates.append({
            "path_id": path_id,
            "experiment_id": experiment_id,
            "cycle_id": cycle_id,
            "mic_label": mic_label,
            "mic_idx": i,
            "quality_error": error,
            "pulse_compr": pulse_compr_score[i, :],
            "lpf": lpf_score[i, :],
            "selected_idx": idx_i,
            "true_distance": float(true_distances[mic_label]),
            "tx_raw": tx_raw,
            "fs": setup["fs_mic"],
        })

    if not per_mic_candidates:
        return None

    quality_error = float(np.mean(mean_errors))
    fig_pc = _build_pulse_compression_plot(
        pulse_compr_all,
        lpf_all,
        corr_index_array,
        selected_anchors_dict,
        tx_raw,
        setup["fs_mic"],
        true_distances,
    )
    fig_pc.update_layout(
        title=(
            f"Pulse compression and LPF - {experiment_id} cycle {cycle_id} "
            f"(path {path_id}, mean |error|={quality_error:.3f} m)"
        )
    )

    run_meta = {
        "path_id": path_id,
        "experiment_id": experiment_id,
        "cycle_id": cycle_id,
        "quality_error": quality_error,
        "figure": fig_pc,
    }
    return run_meta, per_mic_candidates


def _pick_quality_examples(scored_mics: list[dict]) -> dict[str, dict]:
    """Pick close/ok/poor from per-mic scored candidates."""
    scored_sorted = sorted(scored_mics, key=lambda r: r["quality_error"])
    n = len(scored_sorted)
    if n == 0:
        raise ValueError("No valid per-mic candidates could be scored for quality examples.")
    if n == 1:
        return {"close": scored_sorted[0], "ok": scored_sorted[0], "poor": scored_sorted[0]}
    if n == 2:
        return {"close": scored_sorted[0], "ok": scored_sorted[1], "poor": scored_sorted[1]}

    close = scored_sorted[0]

    # Prefer a different microphone for the median-quality example.
    ok = scored_sorted[n // 2]
    if ok["mic_label"] == close["mic_label"]:
        for candidate in scored_sorted:
            if candidate["mic_label"] != close["mic_label"]:
                ok = candidate
                break

    # Prefer a different microphone for the poor example as well.
    poor = scored_sorted[-1]
    if poor["mic_label"] in {close["mic_label"], ok["mic_label"]}:
        for candidate in reversed(scored_sorted):
            if candidate["mic_label"] not in {close["mic_label"], ok["mic_label"]}:
                poor = candidate
                break

    return {
        "close": close,
        "ok": ok,
        "poor": poor,
    }


def _run_quality_examples(setup: dict, max_candidates: int) -> None:
    # Avoid unstable multiprocessing fan-out during broad quality sweeps.
    quality_setup = dict(setup)
    quality_setup["mic_processing_workers"] = 1

    # Force scoring over all 91 microphones, independent of config use_all_mics/used_mics.
    microphone_positions = alf.load_microphone_positions()
    all_mic_labels = list(alf.config["all_mics"])
    quality_setup["selected_mic_positions"] = alf.get_selected_mic_positions(
        microphone_positions,
        all_mic_labels,
    )
    quality_setup["n_selected_ans"] = len(quality_setup["selected_mic_positions"])
    print(f"Quality mode microphone pool: {quality_setup['n_selected_ans']} mics")

    all_path_ids = alf.get_all_path_ids(WALKS_PICKLE_PATH)
    # Limit early to keep the run responsive and avoid loading/formatting all walks.
    candidate_path_ids = all_path_ids[: max(1, int(max_candidates))]
    all_runs = alf.load_runs_to_process(
        candidate_path_ids,
        walks_pickle_path=WALKS_PICKLE_PATH,
        csi_dataset_path=CSI_DATASET_PATH,
    )
    if len(all_runs) == 0:
        raise ValueError("No resting-position runs were found for quality examples.")

    candidate_runs = all_runs
    print(f"Scoring {len(candidate_runs)} candidate runs (per microphone) for LPF quality examples...")

    all_mic_candidates: list[dict] = []
    for path_id, experiment_id, cycle_id in candidate_runs:
        try:
            result = _process_run(path_id, experiment_id, cycle_id, quality_setup, all_mics_for_scoring=True)
            if result is not None:
                _, per_mic = result
                all_mic_candidates.extend(per_mic)
        except Exception as exc:
            print(f"Skipping run path={path_id}, exp={experiment_id}, cycle={cycle_id}: {exc}")

    if len(all_mic_candidates) == 0:
        raise ValueError("No valid scored microphone candidates found. Try increasing --max-candidates.")

    print(f"Scored {len(all_mic_candidates)} individual mic measurements across {len(candidate_runs)} runs.")
    quality_errors = np.array([float(rec["quality_error"]) for rec in all_mic_candidates], dtype=float)
    print(
        "Quality stats (|pred - GT| in meters): "
        f"min={np.min(quality_errors):.4f}, "
        f"p25={np.percentile(quality_errors, 25):.4f}, "
        f"median={np.median(quality_errors):.4f}, "
        f"p75={np.percentile(quality_errors, 75):.4f}, "
        f"max={np.max(quality_errors):.4f}"
    )
    chosen = _pick_quality_examples(all_mic_candidates)

    figs_dir = SCRIPT_DIR / "Figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for label in ("close", "ok", "poor"):
        rec = chosen[label]
        fig = _build_single_mic_lpf_plot(
            pulse_compr=rec["pulse_compr"],
            lpf=rec["lpf"],
            selected_idx=rec["selected_idx"],
            mic_label=rec["mic_label"],
            tx=rec["tx_raw"],
            fs=rec["fs"],
            true_distance=rec["true_distance"],
            quality_label=label,
            quality_error=rec["quality_error"],
        )
        out_path = figs_dir / (
            f"pulse_compression_lpf_{label}_"
            f"path{rec['path_id']}_exp{rec['experiment_id']}_cycle{rec['cycle_id']}_mic{rec['mic_label']}.html"
        )
        fig.write_html(str(out_path))
        # Also keep stable filenames to avoid confusion about which file to open.
        fig.write_html(str(figs_dir / f"pulse_compression_lpf_{label}.html"))
        print(
            f"{label.upper():5s} -> mic {rec['mic_label']} | path {rec['path_id']} | "
            f"exp {rec['experiment_id']} | cycle {rec['cycle_id']} | "
            f"|error|={rec['quality_error']:.4f} m | saved {out_path}"
        )


def main() -> None:
    args = _parse_args()
    setup = alf.load_experiment_setup()

    if args.quality_examples:
        _run_quality_examples(setup, args.max_candidates)
        return

    runs_to_process = alf.load_runs_to_process(
        [args.path_id],
        walks_pickle_path=WALKS_PICKLE_PATH,
        csi_dataset_path=CSI_DATASET_PATH,
    )
    if not runs_to_process:
        raise ValueError(f"No run found for path_id={args.path_id}")

    path_id, experiment_id, cycle_id = runs_to_process[0]
    process_result = _process_run(path_id, experiment_id, cycle_id, setup)
    if process_result is None:
        raise ValueError(
            "Could not compute a valid run (missing position/theoretical distances or invalid selected peaks)."
        )
    run_result, _ = process_result

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
        N_MICROPHONES,
    )
    alf.print_selected_anchors_info(
        selected_anchors_dict,
        sort_key,
        setup["anchor_selection_method"],
    )

    selected_anchors_dict = alf.apply_gain_equalisation(
        selected_anchors_dict,
        setup["chirp_orig_resampl"],
    )

    position = alf.get_rover_position(experiment_id, cycle_id, CSI_DATASET_PATH)
    true_distances = alf.compute_true_distances(
        position,
        selected_anchors_dict,
        setup["selected_mic_positions"],
    )

    tx_raw = np.asarray(setup["chirp_orig_resampl"], dtype=float)
    tx_filtered = alf.butter_highpass_filter(tx_raw, 15000, setup["fs_mic"], 10)
    rx_signals = _collect_raw_and_filtered_rx(
        selected_anchors_dict,
        experiment_id,
        cycle_id,
        setup["fs_mic"],
    )

    pulse_compr_all, lpf_all, corr_index_array = alf.get_corr_with_LPF_curve(
        tx_raw,
        selected_anchors_dict,
    )

    figs_dir = SCRIPT_DIR / "Figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_rx = _build_received_and_transmitted_plot(
        tx_raw,
        tx_filtered,
        rx_signals,
        experiment_id,
        cycle_id,
    )
    first_plot_path = figs_dir / "plot_received_and_iltered_acoustic_signal.html"
    fig_rx.write_html(str(first_plot_path))

    fig_pc = run_result["figure"]  # already built by _process_run
    second_plot_path = figs_dir / "pulse_compression_lpf_plot.html"
    fig_pc.write_html(str(second_plot_path))

    print("\nDone.")
    print(f"Path ID: {path_id} | Experiment: {experiment_id} | Cycle: {cycle_id}")
    print(f"Saved: {first_plot_path}")
    print(f"Saved: {second_plot_path}")


if __name__ == "__main__":
    main()