from pathlib import Path
import importlib.util
import sys

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Setup: Locate and import csi_plot_utils
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
    raise ImportError(f"Could not load csi_plot_utils from {UTILS_PATH}")
csi = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = csi
spec.loader.exec_module(csi)

RESULTS_DIR = PROJECT_ROOT / "results"
ACOUSTIC_DOWNLOAD_SCRIPT = Path("processing") / "dataset-download" / "download_acoustic_datasets.py"
plt.rcParams["figure.figsize"] = (12, 4.5)


def acoustic_download_instructions(experiment_id: str) -> str:
    script_path = ACOUSTIC_DOWNLOAD_SCRIPT.as_posix()
    return (
        f"Could not find an acoustic dataset for {experiment_id} in {RESULTS_DIR}. "
        "Expected a file such as acoustic_<EXP>.nc. "
        f"From the repo root, run `python {script_path} --experiment-id {experiment_id}` to download it into {RESULTS_DIR}, "
        f"or `python {script_path} --list` to inspect the server listing first."
    )


def resolve_acoustic_dataset_path(experiment_id: str, dataset_path: str | Path | None = None) -> Path:
    if dataset_path is not None:
        return Path(dataset_path).resolve()

    patterns = [
        f"acoustic_{experiment_id}.nc",
        f"acoustic_{experiment_id}_*.nc",
        f"acoustic_{experiment_id}*.nc",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(RESULTS_DIR.glob(pattern)))
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    if not candidates:
        raise FileNotFoundError(acoustic_download_instructions(experiment_id))
    return candidates[0]


def open_acoustic_dataset(experiment_id: str, dataset_path: str | Path | None = None) -> tuple[xr.Dataset, Path]:
    path = resolve_acoustic_dataset_path(experiment_id, dataset_path)
    ds = csi.open_netcdf_dataset(path, label=f"acoustic dataset for {experiment_id}")
    available = ds["experiment_id"].values.astype(str).tolist()
    if experiment_id not in available:
        ds.close()
        raise ValueError(
            f"Acoustic dataset {path} does not contain experiment_id={experiment_id!r}. "
            f"Available experiments: {available}"
        )
    return ds, path


def first_available_selection(ds: xr.Dataset, experiment_id: str | None = None) -> tuple[str, int, str]:
    if experiment_id is None:
        if ds.sizes.get("experiment_id", 0) == 0:
            raise ValueError("The acoustic dataset does not contain any experiments.")
        experiment_id = str(ds["experiment_id"].values[0])

    experiment = ds.sel(experiment_id=experiment_id)
    waveform_grid = np.asarray(experiment["values"].values)
    if waveform_grid.size == 0:
        raise ValueError(f"The acoustic dataset for {experiment_id} does not contain any waveform values.")

    cycle_mask = np.any(np.isfinite(waveform_grid), axis=(1, 2))
    cycle_positions = np.flatnonzero(cycle_mask)
    if cycle_positions.size == 0:
        raise ValueError(f"No complete waveform cycle is available for experiment {experiment_id}.")

    cycle_pos = int(cycle_positions[0])
    cycle_id = int(experiment["cycle_id"].values[cycle_pos])
    cycle_slice = experiment.sel(cycle_id=cycle_id)
    mic_waveforms = np.asarray(cycle_slice["values"].values)
    mic_mask = np.any(np.isfinite(mic_waveforms), axis=1)
    mic_positions = np.flatnonzero(mic_mask)
    if mic_positions.size == 0:
        microphone_label = str(cycle_slice["microphone_label"].values[0])
    else:
        microphone_label = str(cycle_slice["microphone_label"].values[int(mic_positions[0])])

    return str(experiment_id), int(cycle_id), microphone_label


def acoustic_xarray_structure_markdown(ds: xr.Dataset, max_coord_preview: int = 6) -> str:
    dimension_rows = []
    for dimension, size in ds.sizes.items():
        if dimension == "sample_index":
            meaning = "Waveform sample axis inside `values`."
        elif dimension in {"experiment_id", "cycle_id", "microphone_label"}:
            meaning = "Named measurement axis."
        else:
            meaning = "No description recorded."
        dimension_rows.append((dimension, int(size), meaning))

    coordinate_rows = [
        (
            coordinate_name,
            type(ds.indexes[coordinate_name]).__name__ if coordinate_name in ds.indexes else "(none)",
            csi.preview_coord_values(ds[coordinate_name].values, max_items=max_coord_preview),
        )
        for coordinate_name in ds.coords
    ]

    variable_rows = []
    for variable_name in ds.data_vars:
        if variable_name == "values":
            meaning = "Acoustic waveforms indexed by experiment_id, cycle_id, microphone_label, and sample_index."
        else:
            meaning = "No description recorded."
        variable_rows.append(
            (
                variable_name,
                ", ".join(ds[variable_name].dims),
                tuple(int(length) for length in ds[variable_name].shape),
                meaning,
            )
        )

    sections = [
        "## Dataset Axes",
        csi.markdown_table(["Dimension", "Size", "Meaning"], dimension_rows),
        "",
        "## Coordinate Indexes",
        csi.markdown_table(["Coordinate", "Index type", "Preview"], coordinate_rows),
        "",
        "## Data Variables",
        csi.markdown_table(["Variable", "Dims", "Shape", "Meaning"], variable_rows),
        "",
        "Think of the dataset as one stack of experiment slices.",
        "",
        "- `experiment_id` selects the experiment.",
        "- `cycle_id` selects one measurement cycle inside that experiment.",
        "- `microphone_label` selects the microphone inside that cycle.",
        "- `sample_index` is the waveform axis inside `values` and is a dimension, not a labeled coordinate.",
    ]
    return "\n".join(sections)


def acoustic_selection_walkthrough_markdown(ds: xr.Dataset, experiment_id: str, cycle_id: int, microphone_label: str) -> str:
    full_sizes = ", ".join(f"{name}={size}" for name, size in ds.sizes.items())
    experiment_slice = ds.sel(experiment_id=experiment_id)
    cycle_slice = experiment_slice.sel(cycle_id=int(cycle_id))
    microphone_slice = cycle_slice.sel(microphone_label=str(microphone_label))

    rows = [
        (
            "`ds`",
            full_sizes,
            "The complete acoustic dataset.",
        ),
        (
            f"`ds.sel(experiment_id=\"{experiment_id}\")`",
            ", ".join(f"{name}={size}" for name, size in experiment_slice.sizes.items()),
            "One experiment slice. `cycle_id` stays as the main measurement axis.",
        ),
        (
            f"`...sel(cycle_id={int(cycle_id)})`",
            ", ".join(f"{name}={size}" for name, size in cycle_slice.sizes.items()) or "(scalar)",
            "One acoustic cycle. The waveform is now arranged only by microphone and sample index.",
        ),
        (
            f"`...sel(microphone_label=\"{microphone_label}\")`",
            ", ".join(f"{name}={size}" for name, size in microphone_slice.sizes.items()) or "(scalar)",
            "One microphone waveform. Only `sample_index` remains.",
        ),
    ]

    sections = [
        "## Selection Walkthrough",
        csi.markdown_table(["Selection", "Remaining dims", "Meaning"], rows),
        "",
        "Use `.sel(...)` for named coordinates such as `experiment_id`, `cycle_id`, and `microphone_label`.",
        "Use `.isel(...)` only when you intentionally want integer positions instead of coordinate labels.",
    ]
    return "\n".join(sections)

