from pathlib import Path
import importlib.util
import sys
import numpy as np

import matplotlib.pyplot as plt

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


EXPERIMENT_ID = "EXP008"
DATASET_PATH = None
SELECTED_CYCLE_ID = None  # Set an integer cycle ID to override the automatic selection.
TARGET_POSITION = None  # Example: {"x": 1.20, "y": 2.40, "z": None}
HEATMAP_MAX_CYCLE_VALUES = csi.DEFAULT_HEATMAP_MAX_CYCLE_VALUES

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
position










x = ds.sel(experiment_id="EXP008")["rover_x"].values
y = ds.sel(experiment_id="EXP008")["rover_y"].values
z = ds.sel(experiment_id="EXP008")["rover_z"].values
last = np.argmax(np.arange(x.shape[0]) * ~np.isnan(x))
x = x[:last + 1]
y = y[:last + 1]
z = z[:last + 1]
z = np.zeros_like(z, dtype=float)


import plotly.graph_objects as go
import numpy as np


# Create index labels
indices = np.arange(1, len(x) + 1)

# Create the 3D scatter plot with a connected path.
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='lines',
    line=dict(color='rgba(70, 70, 70, 0.7)', width=4),
    hoverinfo='skip',
    showlegend=False
))
fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=indices,
        cmin=1,
        cmax=max(len(indices), 1),
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Point index')
    ),
    text=indices,  # This is what shows on hover
    hovertemplate='<b>Index: %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
    showlegend=False
))

fig.update_layout(
    title='3D Scatter Plot with Index on Hover',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    hovermode='closest'
)

import pathlib


# figs_dir = pathlib.Path("myhtml")
# figs_dir.mkdir(exist_ok=True)
# fig.write_html(str(figs_dir / "pulse_compression_lpf_plot.html"))


figs_dir = Path(__file__).parent / "figs"
figs_dir.mkdir(exist_ok=True)
fig.write_html(str(figs_dir / "EXP008_path.html"))

# fig.write_html()
# fig.show()