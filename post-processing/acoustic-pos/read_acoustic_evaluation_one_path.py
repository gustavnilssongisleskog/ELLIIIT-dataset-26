from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import acoustic_local_functions as alf


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    onerun_file = base_dir / "MB_positions_onerun.npy"

    if not onerun_file.exists():
        raise FileNotFoundError(
            f"One-run file not found: {onerun_file}\n"
            "Run acoustics_positioner_calibrated_onerun.py first to generate it."
        )

    print(f"Loading one-run position records from: {onerun_file}")
    pos_records = alf._load_dict_list_from_npy(onerun_file)
    print(f"  Loaded {len(pos_records)} records.")

    # Detect path ID(s) present in this run (expected: 1)
    all_path_ids = sorted({rec.get("path_id", "NA") for rec in pos_records}, key=alf.path_sort_key)
    print(f"  Detected path IDs: {all_path_ids}")

    path_data_by_id = {}
    all_error_2d, all_error_3d = [], []
    all_est_xyz, all_gt_xyz = [], []

    for path_id in all_path_ids:
        pos_path_records, _ = alf.filter_records_by_path_id(pos_records, [], path_id)
        pos_data = alf.prepare_2d_position_data(pos_path_records, only_with_gt=True)
        pos_data_3d = alf.prepare_3d_position_data(pos_path_records, only_with_gt=True)

        path_data_by_id[path_id] = pos_data

        if pos_data["error_2d_m"].size:
            all_error_2d.append(pos_data["error_2d_m"])
        if pos_data_3d["error_3d_m"].size:
            all_error_3d.append(pos_data_3d["error_3d_m"])
            all_est_xyz.append(pos_data_3d["estimated_xyz"])
            all_gt_xyz.append(pos_data_3d["ground_truth_xyz"])

        print(
            f"  Path {path_id}: "
            f"{len(pos_path_records)} records, "
            f"2D error samples={pos_data['error_2d_m'].size}, "
            f"3D error samples={pos_data_3d['error_3d_m'].size}"
        )

    error_2d_all = np.concatenate(all_error_2d) if all_error_2d else np.empty((0,), dtype=float)
    error_3d_all = np.concatenate(all_error_3d) if all_error_3d else np.empty((0,), dtype=float)
    est_xyz_all = np.concatenate(all_est_xyz, axis=0) if all_est_xyz else np.empty((0, 3), dtype=float)
    gt_xyz_all = np.concatenate(all_gt_xyz, axis=0) if all_gt_xyz else np.empty((0, 3), dtype=float)

    print("\nError summary for this run:")
    alf.print_error_summary("2D position error", error_2d_all)
    alf.print_error_summary("3D position error", error_3d_all)
    alf.print_mean_error_vector("3D position error", est_xyz_all, gt_xyz_all, ("dx", "dy", "dz"))

    # CDF figure (2D + 3D, no ranging since this file has no ranging records)
    alf.plot_combined_cdfs(error_2d_all, error_3d_all, np.empty((0,), dtype=float))

    # 2D room path figure – single path so one figure is sufficient
    alf.plot_all_paths_in_one_2d(path_data_by_id, room_size_xy=(8.56, 4.0))

    plt.show()
