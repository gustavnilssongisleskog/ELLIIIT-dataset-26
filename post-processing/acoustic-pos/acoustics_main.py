import acoustic_local_functions as alf

EXPERIMENT_ID = "EXP008"
DATASET_PATH = None  # Set this to a specific .nc file when you do not want the newest match.
MAX_COORD_PREVIEW = 6



ds, dataset_path = alf.open_acoustic_dataset(EXPERIMENT_ID, DATASET_PATH)
SELECTED_EXPERIMENT_ID, SELECTED_CYCLE_ID, SELECTED_MICROPHONE_LABEL = alf.first_available_selection(ds, EXPERIMENT_ID)

print(f"Loaded dataset: {dataset_path}")
print(f"Selected experiment: {SELECTED_EXPERIMENT_ID}")
print(f"Selected cycle: {SELECTED_CYCLE_ID}")
print(f"Selected microphone: {SELECTED_MICROPHONE_LABEL}")
print(
    "Dataset shape:"
    f" experiment_id={ds.sizes.get('experiment_id', 0)},"
    f" cycle_id={ds.sizes.get('cycle_id', 0)},"
    f" microphone_label={ds.sizes.get('microphone_label', 0)},"
    f" sample_index={ds.sizes.get('sample_index', 0)}"
)
cycle_ids = ds["cycle_id"].values.astype(int)
if cycle_ids.size:
    print(f"Cycle ID range: {int(cycle_ids.min())} .. {int(cycle_ids.max())}")

