"""
Calibrate MB position estimates by removing the mean 3D bias vector.

The mean error vector is computed as:
    mean(estimated_xyz - ground_truth_xyz)
over all records that have a valid ground-truth position.  That bias is then
subtracted from *every* estimated_position_xyz (with and without GT), and the
corrected records are written to MB_position_errors_calibrated.npy next to the
original file.

Also updates the derived fields that depend on the estimated position:
    estimated_position_xy   (first two components of calibrated xyz)
    position_error_2d_m     (recomputed if GT is available)
    position_error_3d_m     (recomputed if GT is available)
    estimation_error_vector (recomputed if GT is available)
"""

import copy
import json
from pathlib import Path

import numpy as np

import acoustic_local_functions as alf

CONFIG_FILE = Path(__file__).parent / "config.json"
CONFIG_KEY = "bias_vector_xyz"


def _read_bias_from_config() -> np.ndarray | None:
	"""Return the bias vector stored in config.json, or None if not set / all zeros."""
	if not CONFIG_FILE.exists():
		return None
	with CONFIG_FILE.open() as f:
		cfg = json.load(f)
	val = cfg.get(CONFIG_KEY)
	if val is None:
		return None
	arr = np.asarray(val, dtype=float)
	if np.allclose(arr, 0.0):
		return None
	return arr


def _write_bias_to_config(bias: np.ndarray) -> None:
	"""Persist the computed bias vector back into config.json."""
	if CONFIG_FILE.exists():
		with CONFIG_FILE.open() as f:
			cfg = json.load(f)
	else:
		cfg = {}
	cfg[CONFIG_KEY] = [round(float(v), 6) for v in bias]
	with CONFIG_FILE.open("w") as f:
		json.dump(cfg, f, indent=2)
	print(f"Bias vector written to config: {CONFIG_FILE}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _compute_bias(position_records: list[dict]) -> np.ndarray:
    """Return mean(estimated - GT) over records with a valid GT (shape (3,))."""
    error_vectors = []
    for rec in position_records:
        gt = rec.get("ground_truth_position_xyz")
        est = rec.get("estimated_position_xyz")
        if gt is None or est is None:
            continue
        gt_arr = np.asarray(gt, dtype=float)
        est_arr = np.asarray(est, dtype=float)
        if not (np.isfinite(gt_arr).all() and np.isfinite(est_arr).all()):
            continue
        error_vectors.append(est_arr - gt_arr)

    if not error_vectors:
        raise ValueError("No records with valid ground-truth found – cannot compute bias.")

    bias = np.mean(error_vectors, axis=0)
    return bias


def _apply_calibration(position_records: list[dict], bias: np.ndarray) -> list[dict]:
    """
    Return a deep-copied list of records with the bias subtracted from every
    estimated position, and dependent fields recomputed.
    """
    calibrated = []
    for rec in position_records:
        rec_cal = copy.deepcopy(rec)

        est = rec_cal.get("estimated_position_xyz")
        if est is None:
            calibrated.append(rec_cal)
            continue

        est_arr = np.asarray(est, dtype=float)
        est_cal = est_arr - bias

        rec_cal["estimated_position_xyz"] = est_cal.tolist()
        rec_cal["estimated_position_xy"] = est_cal[:2].tolist()

        gt = rec_cal.get("ground_truth_position_xyz")
        if gt is not None:
            gt_arr = np.asarray(gt, dtype=float)
            if np.isfinite(gt_arr).all() and np.isfinite(est_cal).all():
                err_vec = est_cal - gt_arr
                rec_cal["estimation_error_vector"] = err_vec.tolist()
                rec_cal["position_error_3d_m"] = float(np.linalg.norm(err_vec))
                rec_cal["position_error_2d_m"] = float(np.linalg.norm(err_vec[:2]))

        calibrated.append(rec_cal)
    return calibrated


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    source_file = base_dir / "MB_position_errors.npy"
    output_file = base_dir / "MB_position_errors_calibrated.npy"

    print(f"Loading position records from: {source_file}")
    position_records, _ = alf.load_mb_logs(base_dir)
    print(f"  Loaded {len(position_records)} records.")

    bias = _read_bias_from_config()
    if bias is not None:
        print(
            f"\nUsing bias vector read from config ({CONFIG_FILE}):\n"
            f"  dx = {bias[0]:+.4f} m\n"
            f"  dy = {bias[1]:+.4f} m\n"
            f"  dz = {bias[2]:+.4f} m\n"
            f"  |bias| = {np.linalg.norm(bias):.4f} m"
        )
    else:
        bias = _compute_bias(position_records)
        print(
            f"\nBias vector computed from data (mean estimated - GT):\n"
            f"  dx = {bias[0]:+.4f} m\n"
            f"  dy = {bias[1]:+.4f} m\n"
            f"  dz = {bias[2]:+.4f} m\n"
            f"  |bias| = {np.linalg.norm(bias):.4f} m"
        )
        _write_bias_to_config(bias)

    calibrated_records = _apply_calibration(position_records, bias)

    # -----------------------------------------------------------------------
    # Quick sanity check: residual mean error after calibration
    # -----------------------------------------------------------------------
    residuals = []
    for rec in calibrated_records:
        gt = rec.get("ground_truth_position_xyz")
        est = rec.get("estimated_position_xyz")
        if gt is None or est is None:
            continue
        gt_arr = np.asarray(gt, dtype=float)
        est_arr = np.asarray(est, dtype=float)
        if np.isfinite(gt_arr).all() and np.isfinite(est_arr).all():
            residuals.append(est_arr - gt_arr)

    if residuals:
        residual_mean = np.mean(residuals, axis=0)
        errors_3d = np.array([np.linalg.norm(r) for r in residuals])
        errors_2d = np.array([np.linalg.norm(r[:2]) for r in residuals])
        print(
            f"\nPost-calibration residual mean error vector:\n"
            f"  dx = {residual_mean[0]:+.6f} m\n"
            f"  dy = {residual_mean[1]:+.6f} m\n"
            f"  dz = {residual_mean[2]:+.6f} m"
        )
        print(
            f"\nPost-calibration error summary over {len(errors_3d)} samples:\n"
            f"  3D  mean={np.mean(errors_3d):.4f} m  "
            f"P50={np.percentile(errors_3d, 50):.4f} m  "
            f"P90={np.percentile(errors_3d, 90):.4f} m  "
            f"P95={np.percentile(errors_3d, 95):.4f} m\n"
            f"  2D  mean={np.mean(errors_2d):.4f} m  "
            f"P50={np.percentile(errors_2d, 50):.4f} m  "
            f"P90={np.percentile(errors_2d, 90):.4f} m  "
            f"P95={np.percentile(errors_2d, 95):.4f} m"
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    np.save(output_file, np.array(calibrated_records, dtype=object), allow_pickle=True)
    print(f"\nCalibrated records saved to: {output_file}")
