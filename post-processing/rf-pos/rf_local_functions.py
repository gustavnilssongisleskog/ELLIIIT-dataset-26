from pathlib import Path
import importlib.util
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# IMPORT CSI UTILS MODULE
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



def extract_csi_snapshots(
    ds: xr.Dataset,
    experiment_id: str,
    cycle_ids: list[int] | tuple[int] | np.ndarray,
    antenna_positions: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    cycle_ids = np.asarray(cycle_ids, dtype=int)
    if cycle_ids.size == 0:
        raise ValueError("cycle_ids must not be empty")

    experiment = ds.sel(experiment_id=experiment_id)
    selected = experiment.sel(cycle_id=cycle_ids)
    csi_available = selected["csi_available"].values > 0
    if not np.any(csi_available):
        raise ValueError(
            f"No CSI data available for experiment {experiment_id}, cycles {cycle_ids.tolist()}."
        )

    present_hostnames = selected["hostname"].values[np.any(csi_available, axis=0)].astype(str)
    hostnames = csi.ordered_hostnames(present_hostnames, antenna_positions)
    selected = selected.sel(hostname=hostnames)

    csi_real = selected["csi_real"].values.astype(float)
    csi_imag = selected["csi_imag"].values.astype(float)
    csi_complex = csi_real + 1j * csi_imag
    amplitude = np.abs(csi_complex)
    power_db = csi.power_to_db(np.square(amplitude))
    phase_deg = np.rad2deg(np.angle(csi_complex))

    ds_out = xr.Dataset(
        data_vars={
            "csi_real": (("cycle_id", "hostname"), csi_real),
            "csi_imag": (("cycle_id", "hostname"), csi_imag),
            "csi_amplitude": (("cycle_id", "hostname"), amplitude),
            "csi_power_db": (("cycle_id", "hostname"), power_db),
            "csi_phase_deg": (("cycle_id", "hostname"), phase_deg),
        },
        coords={
            "cycle_id": selected["cycle_id"].values.astype(int),
            "hostname": np.asarray(hostnames, dtype=str),
        },
        attrs={"experiment_id": experiment_id},
    )

    ds_out["csi_host_count"] = (
        ("cycle_id"),
        np.sum(csi_available, axis=1).astype(int),
    )
    for coord_name in ("rover_x", "rover_y", "rover_z", "position_available"):
        if coord_name in selected:
            ds_out[coord_name] = ("cycle_id", selected[coord_name].values)

    return ds_out


def extract_phase(requests: dict[str, list[int]]) -> xr.Dataset:
    antenna_positions = csi.load_antenna_positions()
    cycles_by_experiment = requests
    hostnames = _discover_requested_hostnames(cycles_by_experiment, antenna_positions)

    # CREATE DISTINCT OBSERVATIONS
    n_antennas = len(hostnames)
    observation_keys: list[tuple[str, int]] = []
    for experiment_id, cycle_ids in cycles_by_experiment.items():
        observation_keys.extend((experiment_id, cycle_id) for cycle_id in cycle_ids)
    n_observations = len(observation_keys)

    # INITIALIZE OUTPUT ARRAYS
    phase = np.full((n_observations, n_antennas), np.nan, dtype=float)
    rover_x = np.full((n_observations,), np.nan, dtype=float)
    rover_y = np.full((n_observations,), np.nan, dtype=float)
    rover_z = np.full((n_observations,), np.nan, dtype=float)
    position_available = np.zeros((n_observations,), dtype=bool)
    csi_host_count = np.zeros((n_observations,), dtype=int)
    source_experiment_id = np.empty((n_observations,), dtype=object)
    source_cycle_id = np.empty((n_observations,), dtype=int)

    # REORDER AND AGGREGATE DATA INTO OBSERVATIONS
    observation_start = 0
    for experiment_id, cycle_ids in cycles_by_experiment.items():
        ds, _ = csi.open_dataset(experiment_id=experiment_id, dataset_path=None)
        try:
            print(f"Processing experiment: {experiment_id}")
            experiment = ds.sel(experiment_id=experiment_id)
            selected = experiment.sel(cycle_id=cycle_ids).reindex(cycle_id=cycle_ids, hostname=hostnames)

            csi_real = selected["csi_real"].values.astype(float)
            csi_imag = selected["csi_imag"].values.astype(float)
            csi_complex = csi_real + 1j * csi_imag
            phase_slice = np.rad2deg(np.angle(csi_complex)) #TODO: VILL MAN VERKLIGEN HA I GRADER?

            n_cycles = len(cycle_ids)
            phase[observation_start : observation_start + n_cycles, :] = phase_slice
            csi_available = selected["csi_available"].values > 0
            csi_host_count[observation_start : observation_start + n_cycles] = np.sum(csi_available, axis=1).astype(int)

            for coord_name, target in (
                ("rover_x", rover_x),
                ("rover_y", rover_y),
                ("rover_z", rover_z),
            ):
                if coord_name in selected:
                    target[observation_start : observation_start + n_cycles] = selected[coord_name].values.astype(float)

            if "position_available" in selected:
                position_available[observation_start : observation_start + n_cycles] = selected["position_available"].values > 0

            for offset, cycle_id in enumerate(cycle_ids):
                source_experiment_id[observation_start + offset] = experiment_id
                source_cycle_id[observation_start + offset] = cycle_id

            observation_start += n_cycles
        finally:
            ds.close()

    antenna_hostnames = np.asarray(hostnames, dtype=str)
    antenna_xyz = np.full((n_antennas, 3), np.nan, dtype=float)
    for index, hostname in enumerate(hostnames):
        position = antenna_positions.get(hostname.upper())
        if position is not None:
            antenna_xyz[index] = np.asarray(position, dtype=float)

    # CREATE RESTRUCTURED DATASET
    phase_dataset = xr.Dataset(
        data_vars={
            "csi_phase_deg": (("observation", "antenna"), phase),
            "rover_x": (("observation",), rover_x),
            "rover_y": (("observation",), rover_y),
            "rover_z": (("observation",), rover_z),
            "position_available": (("observation",), position_available),
            "csi_host_count": (("observation",), csi_host_count),
            "antenna_hostname": (("antenna",), antenna_hostnames),
            "antenna_x": (("antenna",), antenna_xyz[:, 0]),
            "antenna_y": (("antenna",), antenna_xyz[:, 1]),
            "antenna_z": (("antenna",), antenna_xyz[:, 2]),
        },
        coords={
            "observation": np.arange(n_observations, dtype=int),
            "antenna": np.arange(n_antennas, dtype=int),
            "source_experiment_id": ("observation", source_experiment_id),
            "source_cycle_id": ("observation", source_cycle_id),
        },
    )

    return phase_dataset


def _discover_requested_hostnames(
    cycles_by_experiment: dict[str, list[int]],
    antenna_positions: dict[str, np.ndarray] | None,
) -> list[str]:
    hostnames: list[str] = []
    seen_hostnames: set[str] = set()

    for experiment_id, requested_cycle_ids in cycles_by_experiment.items():
        ds, _ = csi.open_dataset(experiment_id=experiment_id, dataset_path=None)
        try:
            print(f"Discovering hostnames for experiment: {experiment_id}")
            experiment = ds.sel(experiment_id=experiment_id)
            selected = experiment.sel(cycle_id=sorted(set(requested_cycle_ids)))
            csi_available = selected["csi_available"].values > 0
            present_hostnames = selected["hostname"].values[np.any(csi_available, axis=0)].astype(str)
            for hostname in csi.ordered_hostnames(present_hostnames, antenna_positions):
                if hostname not in seen_hostnames:
                    seen_hostnames.add(hostname)
                    hostnames.append(hostname)
        finally:
            ds.close()

    return hostnames



# if __name__ == "__main__":
#     # requests = {
#     #     "EXP003": [i + 1 for i in range(529)],
#     #     "EXP005": [i + 1 for i in range(777)],
#     #     "EXP006": [i + 1 for i in range(238)],
#     #     "EXP007": [i + 1 for i in range(378)],
#     #     "EXP008": [i + 1 for i in range(201)],
#     #     "EXP010": [i + 1 for i in range(458)],
#     #     "EXP011": [i + 1 for i in range(291)],
#     #     "EXP012": [i + 1 for i in range(784)]
#     # }
#     requests = {
#         "EXP003": [1, 100, 200, 300, 400],
#         "EXP005": [1, 2, 3],
#     }
#     ds = extract_phase(requests)

#     print(ds)