from pathlib import Path
import importlib.util
import sys
import matplotlib.pyplot as plt



# FIND CSI UTILS MODULE
NOTEBOOK_DIR = Path.cwd().resolve()
for candidate_dir in (
    NOTEBOOK_DIR,
    NOTEBOOK_DIR / "tutorials",
    NOTEBOOK_DIR / "processing" / "tutorials",
):
    if (candidate_dir / "csi_plot_utils.py").exists():
        NOTEBOOK_DIR = candidate_dir.resolve()
        break
else:
    raise ImportError(f"Could not locate csi_plot_utils.py from {Path.cwd().resolve()}")

UTILS_PATH = NOTEBOOK_DIR / "csi_plot_utils.py"
PROCESSING_DIR = NOTEBOOK_DIR.parent
PROJECT_ROOT = PROCESSING_DIR.parent
spec = importlib.util.spec_from_file_location("csi_plot_utils", UTILS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load utility module from {UTILS_PATH}")
csi = importlib.util.module_from_spec(spec)
sys.modules["csi_plot_utils"] = csi
spec.loader.exec_module(csi)



# CONSTANTS
EXPERIMENT_ID = "EXP003" # experiment, i.e. where to rig is manually placed
DATASET_PATH = None # ???
SELECTED_CYCLE_ID = None  # Set an integer cycle ID to override the automatic selection.
TARGET_POSITION = None  # Example: {"x": 1.20, "y": 2.40, "z": None}
HEATMAP_MAX_CYCLE_VALUES = csi.DEFAULT_HEATMAP_MAX_CYCLE_VALUES # ???



# RESOLVE POSITION
ds, dataset_path = csi.open_dataset(experiment_id=EXPERIMENT_ID, dataset_path=DATASET_PATH)
antenna_positions = csi.load_antenna_positions()
available_cycles = csi.available_cycle_ids(ds, EXPERIMENT_ID)

print(f"Loaded dataset: {dataset_path}")
print(f"CSI cycles available: {available_cycles.size}")

if TARGET_POSITION is not None:
    nearest = csi.find_nearest_position_cycle(
        ds,
        EXPERIMENT_ID,
        x=TARGET_POSITION["x"],
        y=TARGET_POSITION["y"],
        z=TARGET_POSITION.get("z"),
    )
    SELECTED_CYCLE_ID = int(nearest["cycle_id"])
    print("Selected the nearest recorded cycle for the requested point:")
    print(nearest)
elif SELECTED_CYCLE_ID is None:
    SELECTED_CYCLE_ID = int(available_cycles[0])
    print(f"Using the first cycle with CSI: {SELECTED_CYCLE_ID}")
else:
    SELECTED_CYCLE_ID = int(SELECTED_CYCLE_ID)

position = csi.cycle_position(ds, EXPERIMENT_ID, SELECTED_CYCLE_ID)
print("Rover position for the selected cycle:")
print(position)

# EXTRACT CSI VECTOR
snapshot = csi.extract_csi_snapshot(
    ds,
    EXPERIMENT_ID,
    SELECTED_CYCLE_ID,
    antenna_positions=antenna_positions,
)

print(
    f"Extracted {snapshot.sizes['hostname']} CSI values "
    f"for experiment {EXPERIMENT_ID}, cycle {SELECTED_CYCLE_ID}"
)
print(snapshot)