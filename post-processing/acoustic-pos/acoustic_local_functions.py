from pathlib import Path
import importlib.util
import sys

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


def open_csi_dataset(experiment_id: str, dataset_path: str | Path | None = None) -> tuple[xr.Dataset, Path]:
    if dataset_path is None:
        patterns = ["csi*.nc"]
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(sorted(RESULTS_DIR.glob(pattern)))
        candidates = list(dict.fromkeys(path.resolve() for path in candidates))
        if not candidates:
            raise FileNotFoundError(f"Could not find CSI dataset in {RESULTS_DIR}")
        dataset_path = candidates[0]

    dataset_path = Path(dataset_path).resolve()
    ds = csi.open_netcdf_dataset(dataset_path, label=f"CSI dataset")
    available = ds["experiment_id"].values.astype(str).tolist()
    if experiment_id not in available:
        ds.close()
        raise ValueError(
            f"CSI dataset {dataset_path} does not contain experiment_id={experiment_id!r}. "
            f"Available experiments: {available}"
        )
    return ds, dataset_path


def get_acoustic_dataset_shape(
    experiment_id: str,
    dataset_path: str | Path | None = None,
) -> dict[str, int]:
    ds, _ = open_acoustic_dataset(experiment_id, dataset_path)
    shape = {name: int(size) for name, size in ds.sizes.items()}
    ds.close()
    return shape


def get_acoustic_waveform(
    experiment_id: str,
    cycle_id: int,
    microphone_label: str,
    dataset_path: str | Path | None = None,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    ds, _ = open_acoustic_dataset(experiment_id, dataset_path)
    waveform = ds.sel(experiment_id=experiment_id, cycle_id=int(cycle_id))
    waveform = waveform["values"].sel(microphone_label=str(microphone_label)).load()
    ds.close()

    waveform_values = np.asarray(waveform.values, dtype=float)
    sample_index = np.arange(waveform_values.size)
    return waveform, waveform_values, sample_index


def get_rover_position(
    experiment_id: str,
    cycle_id: int,
    csi_dataset_path: str | Path | None = None,
) -> dict[str, object]:
    csi_ds, _ = open_csi_dataset(experiment_id, csi_dataset_path)
    position = csi.cycle_position(csi_ds, experiment_id, int(cycle_id))
    csi_ds.close()
    return position


