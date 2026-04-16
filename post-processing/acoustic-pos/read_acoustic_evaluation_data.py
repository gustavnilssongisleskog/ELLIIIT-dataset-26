import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import acoustic_local_functions as alf


def _path_sort_key(path_id):
	path_str = str(path_id)
	if path_str.isdigit():
		return (0, int(path_str))
	return (1, path_str)


def _print_error_summary(label: str, errors_m: np.ndarray) -> None:
	if errors_m.size == 0:
		print(f"{label}: no samples")
		return

	mean_m = float(np.mean(errors_m))
	rmse_m = float(np.sqrt(np.mean(np.square(errors_m))))
	p0_m, p50_m, p90_m, p95_m = np.percentile(errors_m, [0, 50, 90, 95])

	print(
		f"{label}: "
		f"mean={mean_m:.4f} m, "
		f"RMSE={rmse_m:.4f} m, "
		f"P0={p0_m:.4f} m, "
		f"P50={p50_m:.4f} m, "
		f"P90={p90_m:.4f} m, "
		f"P95={p95_m:.4f} m"
	)


def _print_mean_error_vector(label: str, estimated: np.ndarray, ground_truth: np.ndarray, axes: tuple[str, ...]) -> None:
	if estimated.size == 0 or ground_truth.size == 0:
		print(f"{label} mean error vector: no samples")
		return

	valid_rows = np.isfinite(estimated).all(axis=1) & np.isfinite(ground_truth).all(axis=1)
	if not np.any(valid_rows):
		print(f"{label} mean error vector: no finite samples")
		return

	error_vectors = estimated[valid_rows] - ground_truth[valid_rows]
	mean_vector = np.mean(error_vectors, axis=0)
	mean_vector_norm = float(np.linalg.norm(mean_vector))
	components = ", ".join(f"{ax}={val:.4f} m" for ax, val in zip(axes, mean_vector))
	print(f"{label} mean error vector (est-gt): [{components}], |mean vector|={mean_vector_norm:.4f} m")


def _get_figs_dir() -> Path:
	figs_dir = Path(__file__).resolve().parent / "Figs"
	figs_dir.mkdir(parents=True, exist_ok=True)
	return figs_dir


def _save_figure(fig: plt.Figure, stem: str) -> None:
	figs_dir = _get_figs_dir()
	png_path = figs_dir / f"{stem}.png"
	svg_path = figs_dir / f"{stem}.svg"
	tex_path = figs_dir / f"{stem}.tex"

	fig.savefig(png_path, dpi=300, bbox_inches="tight")
	fig.savefig(svg_path, bbox_inches="tight")
	try:
		tikzplotlib.save(str(tex_path), figure=fig)
		print(f"Saved figure: {tex_path}")
	except Exception as exc:
		print(f"Skipping TikZ export for {stem}: {exc}")

	print(f"Saved figure: {png_path}")
	print(f"Saved figure: {svg_path}")


def _cdf_label_with_stats(base_label: str, errors_m: np.ndarray) -> str:
	if errors_m.size == 0:
		return f"{base_label} (no samples)"

	mean_m = float(np.mean(errors_m))
	p50_m, p90_m, p95_m = np.percentile(errors_m, [50, 90, 95])
	return (
		f"{base_label} | "
		f"mean={mean_m:.3f} m, "
		f"P50={p50_m:.3f} m, "
		f"P90={p90_m:.3f} m, "
		f"P95={p95_m:.3f} m"
	)


def _annotate_cdf_stats(
	ax: plt.Axes,
	errors_m: np.ndarray,
	curve_color,
	line_style: str,
	label_prefix: str,
	text_y: float,
) -> None:
	if errors_m.size == 0:
		return
	mean_m = float(np.mean(errors_m))
	p50_m, p90_m, p95_m = np.percentile(errors_m, [50, 90, 95])
	stats = [
		("mean", mean_m),
		("P50", float(p50_m)),
		("P90", float(p90_m)),
		("P95", float(p95_m)),
	]

	# mean vs P50: the smaller value's label goes left, the larger goes right
	# P90 and P95 always go to the right
	mean_val = next(v for n, v in stats if n == "mean")
	p50_val = next(v for n, v in stats if n == "P50")
	for name, value in stats:
		ax.axvline(value, color=curve_color, linestyle=line_style, linewidth=1.0, alpha=0.35)
		if name == "mean":
			left = mean_val < p50_val
		elif name == "P50":
			left = p50_val < mean_val
		else:
			left = False
		if left:
			x_text, ha = value - 0.005, "right"
		else:
			x_text, ha = value + 0.005, "left"
		ax.text(
			x_text,
			text_y,
			f"{label_prefix} {name}={value:.3f} m",
			color=curve_color,
			fontsize=8,
			rotation=90,
			va="bottom",
			ha=ha,
			alpha=0.9,
		)


def _plot_combined_cdfs(error_2d_m: np.ndarray, error_3d_m: np.ndarray, ranging_abs_error_m: np.ndarray) -> None:
	x2, y2 = alf.empirical_cdf(error_2d_m)
	x3, y3 = alf.empirical_cdf(error_3d_m)
	xr, yr = alf.empirical_cdf(ranging_abs_error_m)

	fig, ax = plt.subplots(figsize=(10, 6))
	curve_specs = []
	if x2.size:
		line2, = ax.plot(
			x2,
			y2,
			linewidth=2,
			label="2D Position Error CDF (all paths)",
		)
		curve_specs.append((line2.get_color(), "-", "2D", error_2d_m, 0.7))
	if x3.size:
		line3, = ax.plot(
			x3,
			y3,
			linewidth=2,
			linestyle="--",
			label="3D Position Error CDF (all paths)",
		)
		curve_specs.append((line3.get_color(), "--", "3D", error_3d_m, 0.0))
	if xr.size:
		liner, = ax.plot(
			xr,
			yr,
			linewidth=2,
			linestyle=":",
			label="Ranging Error CDF (all paths)",
		)
		curve_specs.append((liner.get_color(), ":", "RNG", ranging_abs_error_m, 0.2))

	for color, linestyle, label_prefix, errors, text_y in curve_specs:
		_annotate_cdf_stats(
			ax=ax,
			errors_m=errors,
			curve_color=color,
			line_style=linestyle,
			label_prefix=label_prefix,
			text_y=text_y,
		)

	ax.set_title("Combined CDFs Across All Paths")
	ax.set_xlabel("Error (m)")
	ax.set_ylabel("Empirical CDF")
	ax.set_xlim(0, 1)
	ax.grid(True, alpha=0.3)
	if x2.size or x3.size or xr.size:
		ax.legend(fontsize=8)
	plt.tight_layout()
	_save_figure(fig, "combined_cdfs")
	plt.close(fig)


def _plot_all_paths_in_one_2d(path_data_by_id: dict, room_size_xy: tuple[float, float] = (8.56, 4.0)) -> None:
	fig, ax = plt.subplots(figsize=(8, 8))

	room_x, room_y = room_size_xy
	room_outline_x = [0.0, room_x, room_x, 0.0, 0.0]
	room_outline_y = [0.0, 0.0, room_y, room_y, 0.0]
	ax.plot(room_outline_x, room_outline_y, "k-", linewidth=2.0, label="Room boundary")

	cmap = plt.cm.get_cmap("tab20", max(1, len(path_data_by_id)))

	for idx, path_id in enumerate(sorted(path_data_by_id.keys(), key=_path_sort_key)):
		pos_data = path_data_by_id[path_id]
		est = pos_data["estimated_xy"]
		gt = pos_data["ground_truth_xy"]
		if est.size == 0:
			continue

		color = cmap(idx)
		ax.plot(est[:, 0], est[:, 1], "o-", color=color, linewidth=1.4, markersize=3, label=f"Path {path_id} est")

		valid_gt = ~np.isnan(gt).any(axis=1)
		if np.any(valid_gt):
			gt_valid = gt[valid_gt]
			ax.plot(gt_valid[:, 0], gt_valid[:, 1], "--", color=color, linewidth=1.3, alpha=0.8, label=f"Path {path_id} gt")

	ax.set_title("Estimated and GT 2D Trajectories - All Paths")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(-0.2, room_x + 0.2)
	ax.set_ylim(-0.2, room_y + 0.2)
	ax.grid(True, alpha=0.3)

	handles, labels = ax.get_legend_handles_labels()
	if labels:
		unique = dict(zip(labels, handles))
		ax.legend(unique.values(), unique.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

	plt.tight_layout()
	_save_figure(fig, "all_paths_2d")
	plt.close(fig)


def _plot_path_subfigures(path_data_by_id: dict, room_size_xy: tuple[float, float] = (8.56, 4.0)) -> None:
	path_ids = sorted(path_data_by_id.keys(), key=_path_sort_key)
	n_paths = len(path_ids)
	if n_paths == 0:
		print("No path data available for subplot figure.")
		return

	n_cols = 3
	n_rows = 4
	paths_per_figure = n_cols * n_rows
	n_figures = int(math.ceil(n_paths / paths_per_figure))

	room_x, room_y = room_size_xy
	room_outline_x = [0.0, room_x, room_x, 0.0, 0.0]
	room_outline_y = [0.0, 0.0, room_y, room_y, 0.0]

	for fig_idx in range(n_figures):
		start = fig_idx * paths_per_figure
		end = min(start + paths_per_figure, n_paths)
		page_path_ids = path_ids[start:end]

		fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

		for idx, path_id in enumerate(page_path_ids):
			r = idx // n_cols
			c = idx % n_cols
			ax = axes[r][c]
			pos_data = path_data_by_id[path_id]
			est = pos_data["estimated_xy"]
			gt = pos_data["ground_truth_xy"]

			ax.plot(room_outline_x, room_outline_y, "k-", linewidth=1.6, label="Room")
			if est.size:
				ax.plot(est[:, 0], est[:, 1], "o-", linewidth=1.5, markersize=3, label="Estimated")

			valid_gt = ~np.isnan(gt).any(axis=1)
			if np.any(valid_gt):
				gt_valid = gt[valid_gt]
				ax.plot(gt_valid[:, 0], gt_valid[:, 1], "s--", linewidth=1.3, markersize=3, label="GT")

			ax.set_title(f"Path {path_id}")
			ax.set_xlabel("X (m)")
			ax.set_ylabel("Y (m)")
			ax.set_aspect("equal", adjustable="box")
			ax.set_xlim(-0.2, room_x + 0.2)
			ax.set_ylim(-0.2, room_y + 0.2)
			ax.grid(True, alpha=0.3)
			ax.legend(fontsize=8)

		total_axes = n_rows * n_cols
		for idx in range(len(page_path_ids), total_axes):
			r = idx // n_cols
			c = idx % n_cols
			axes[r][c].axis("off")

		fig.suptitle(
			f"Estimated vs GT 2D Trajectories by Path (Figure {fig_idx + 1}/{n_figures})",
			fontsize=14,
		)
		plt.tight_layout(rect=[0, 0, 1, 0.98])
		_save_figure(fig, f"path_subfigures_page_{fig_idx + 1:02d}")
		plt.close(fig)


if __name__ == "__main__":
	pos_records_all, rng_records_all = alf.load_mb_logs()

	all_path_ids = sorted({rec.get("path_id", "NA") for rec in pos_records_all}, key=_path_sort_key)
	if not all_path_ids:
		all_path_ids = sorted({rec.get("path_id", "NA") for rec in rng_records_all}, key=_path_sort_key)

	print(f"Loaded records -> position: {len(pos_records_all)}, ranging: {len(rng_records_all)}")
	print(f"Detected path IDs: {all_path_ids}")

	path_data_by_id = {}
	all_error_2d = []
	all_error_3d = []
	all_ranging_abs = []
	all_est_xy = []
	all_gt_xy = []
	all_est_xyz = []
	all_gt_xyz = []

	for path_id in all_path_ids:
		pos_records, rng_records = alf.filter_records_by_path_id(pos_records_all, rng_records_all, path_id)
		pos_data = alf.prepare_2d_position_data(pos_records, only_with_gt=True)
		pos_data_3d = alf.prepare_3d_position_data(pos_records, only_with_gt=True)
		rng_data = alf.prepare_ranging_error_data(rng_records)

		path_data_by_id[path_id] = pos_data

		if pos_data["error_2d_m"].size:
			all_error_2d.append(pos_data["error_2d_m"])
			all_est_xy.append(pos_data["estimated_xy"])
			all_gt_xy.append(pos_data["ground_truth_xy"])
		if pos_data_3d["error_3d_m"].size:
			all_error_3d.append(pos_data_3d["error_3d_m"])
			all_est_xyz.append(pos_data_3d["estimated_xyz"])
			all_gt_xyz.append(pos_data_3d["ground_truth_xyz"])
		if rng_data["per_anchor_abs_error_m"].size:
			all_ranging_abs.append(rng_data["per_anchor_abs_error_m"])

		print(
			f"Path {path_id}: "
			f"position records={len(pos_records)}, ranging records={len(rng_records)}, "
			f"2D error samples={pos_data['error_2d_m'].size}, "
			f"3D error samples={pos_data_3d['error_3d_m'].size}, "
			f"ranging samples={rng_data['per_anchor_abs_error_m'].size}"
		)

	error_2d_all = np.concatenate(all_error_2d) if all_error_2d else np.empty((0,), dtype=float)
	error_3d_all = np.concatenate(all_error_3d) if all_error_3d else np.empty((0,), dtype=float)
	ranging_abs_all = np.concatenate(all_ranging_abs) if all_ranging_abs else np.empty((0,), dtype=float)
	est_xy_all = np.concatenate(all_est_xy, axis=0) if all_est_xy else np.empty((0, 2), dtype=float)
	gt_xy_all = np.concatenate(all_gt_xy, axis=0) if all_gt_xy else np.empty((0, 2), dtype=float)
	est_xyz_all = np.concatenate(all_est_xyz, axis=0) if all_est_xyz else np.empty((0, 3), dtype=float)
	gt_xyz_all = np.concatenate(all_gt_xyz, axis=0) if all_gt_xyz else np.empty((0, 3), dtype=float)

	print("\nCombined samples across all paths")
	print(f"2D position errors: {error_2d_all.size}")
	print(f"3D position errors: {error_3d_all.size}")
	print(f"Ranging absolute errors: {ranging_abs_all.size}")
	print("\nCombined evaluation metrics across all paths")
	_print_error_summary("2D position error", error_2d_all)
	_print_error_summary("3D position error", error_3d_all)
	_print_error_summary("Ranging absolute error", ranging_abs_all)
	_print_mean_error_vector("2D position error", est_xy_all, gt_xy_all, ("dx", "dy"))
	_print_mean_error_vector("3D position error", est_xyz_all, gt_xyz_all, ("dx", "dy", "dz"))

	_plot_combined_cdfs(error_2d_all, error_3d_all, ranging_abs_all)
	_plot_all_paths_in_one_2d(path_data_by_id, room_size_xy=(8.56, 4.0))
	_plot_path_subfigures(path_data_by_id, room_size_xy=(8.56, 4.0))

