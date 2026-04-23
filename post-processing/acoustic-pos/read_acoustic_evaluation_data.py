import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import acoustic_local_functions as alf


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	base_dir = Path(__file__).parent
	calibrated_file = base_dir / "MB_position_errors_calibrated.npy"
	ranging_file = base_dir / "MB_ranging_errors.npy"

	if calibrated_file.exists():
		print(f"Loading calibrated position records from: {calibrated_file}")
		pos_records_all = alf._load_dict_list_from_npy(calibrated_file)
	else:
		print(f"Calibrated file not found, falling back to: {base_dir / 'MB_position_errors.npy'}")
		pos_records_all, _ = alf.load_mb_logs(base_dir)

	rng_records_all = alf._load_dict_list_from_npy(ranging_file) if ranging_file.exists() else []

	all_path_ids = sorted({rec.get("path_id", "NA") for rec in pos_records_all}, key=alf.path_sort_key)
	if not all_path_ids:
		all_path_ids = sorted({rec.get("path_id", "NA") for rec in rng_records_all}, key=alf.path_sort_key)

	print(f"Loaded records -> position: {len(pos_records_all)}, ranging: {len(rng_records_all)}")
	print(f"Detected path IDs: {all_path_ids}")

	path_data_by_id = {}
	all_error_2d, all_error_3d, all_ranging_abs = [], [], []
	all_est_xy, all_gt_xy = [], []
	all_est_xyz, all_gt_xyz = [], []

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
	alf.print_error_summary("2D position error", error_2d_all)
	alf.print_error_summary("3D position error", error_3d_all)
	alf.print_error_summary("Ranging absolute error", ranging_abs_all)
	alf.print_mean_error_vector("2D position error", est_xy_all, gt_xy_all, ("dx", "dy"))
	alf.print_mean_error_vector("3D position error", est_xyz_all, gt_xyz_all, ("dx", "dy", "dz"))

	alf.plot_combined_cdfs(error_2d_all, error_3d_all, ranging_abs_all, save_fn=_save_figure)
	alf.plot_all_paths_in_one_2d(path_data_by_id, room_size_xy=(8.56, 4.0), save_fn=_save_figure)
	alf.plot_path_subfigures(path_data_by_id, room_size_xy=(8.56, 4.0), save_fn=_save_figure)
