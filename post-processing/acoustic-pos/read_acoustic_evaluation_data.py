import acoustic_local_functions as alf


if __name__ == "__main__":
	PATH_ID = 0

	pos_records_all, rng_records_all = alf.load_mb_logs()
	pos_records, rng_records = alf.filter_records_by_path_id(pos_records_all, rng_records_all, PATH_ID)

	pos_data = alf.prepare_2d_position_data(pos_records, only_with_gt=True)
	pos_data_3d = alf.prepare_3d_position_data(pos_records, only_with_gt=True)
	rng_data = alf.prepare_ranging_error_data(rng_records)

	x_cdf_2d, y_cdf_2d = alf.empirical_cdf(pos_data["error_2d_m"])
	x_cdf_3d, y_cdf_3d = alf.empirical_cdf(pos_data_3d["error_3d_m"])

	print(f"Loaded {len(pos_records_all)} total position records and {len(rng_records_all)} total ranging records")
	print(f"Selected PATH_ID: {PATH_ID}")
	print(f"Filtered records -> position: {len(pos_records)}, ranging: {len(rng_records)}")
	print(f"2D position-error samples for CDF: {pos_data['error_2d_m'].size}")
	print(f"3D position-error samples for CDF: {pos_data_3d['error_3d_m'].size}")
	print(f"Per-anchor ranging-error samples: {rng_data['per_anchor_abs_error_m'].size}")
	print(f"CDF points prepared (2D): {x_cdf_2d.size}")
	print(f"CDF points prepared (3D): {x_cdf_3d.size}")
	if pos_data['path_id'].size:
		print(f"position path_id first 5: {pos_data['path_id'][:5]}")
	if rng_data['path_id'].size:
		print(f"ranging path_id first 5: {rng_data['path_id'][:5]}")

	# Preview what the data inside the files looks like
	if pos_records:
		print("\nPosition file preview")
		print(f"Position record keys: {sorted(pos_records[0].keys())}")
		print(f"First position record: {pos_records[0]}")
		if len(pos_records) > 1:
			print(f"Last position record: {pos_records[-1]}")
	else:
		print("\nPosition file preview: no records")

	if rng_records:
		print("\nRanging file preview")
		print(f"Ranging record keys: {sorted(rng_records[0].keys())}")
		print(f"First ranging record: {rng_records[0]}")
		if len(rng_records) > 1:
			print(f"Last ranging record: {rng_records[-1]}")
	else:
		print("\nRanging file preview: no records")

	# Preview arrays useful for 2D path plotting (estimated and real trajectory)
	print("\n2D path plotting preview")
	print(f"estimated_xy shape: {pos_data['estimated_xy'].shape}")
	print(f"ground_truth_xy shape: {pos_data['ground_truth_xy'].shape}")
	if pos_data['estimated_xy'].size:
		print(f"estimated_xy first 3 rows: {pos_data['estimated_xy'][:3]}")
		print(f"ground_truth_xy first 3 rows: {pos_data['ground_truth_xy'][:3]}")
		print(f"experiment_cycle first 5: {pos_data['experiment_cycle'][:5]}")
		print(f"2D error first 5 (m): {pos_data['error_2d_m'][:5]}")

	if rng_data['experiment_cycle'].size:
		print(f"ranging experiment_cycle first 5: {rng_data['experiment_cycle'][:5]}")

	# Plots for the selected PATH_ID
	alf.plot_position_error_cdfs(pos_data['error_2d_m'], pos_data_3d['error_3d_m'], PATH_ID)
	alf.plot_estimated_vs_gt_2d(pos_data['estimated_xy'], pos_data['ground_truth_xy'], PATH_ID)

