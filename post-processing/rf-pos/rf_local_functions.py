from pathlib import Path
import importlib.util
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
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


### DATA REFORMATTING ##################################################
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
            experiment = ds.sel(experiment_id=experiment_id)
            selected = experiment.sel(cycle_id=cycle_ids).reindex(cycle_id=cycle_ids, hostname=hostnames)

            csi_real = selected["csi_real"].values.astype(float)
            csi_imag = selected["csi_imag"].values.astype(float)
            csi_complex = csi_real + 1j * csi_imag
            phase_slice = np.angle(csi_complex)

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
            "csi_phase_rad": (("observation", "antenna"), phase),
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

### POSITION ESTIMATION ##################################################
def phase_to_distance(
    phase_rad: np.ndarray,
    frequency_hz: float,
    speed_of_light: float = const.speed_of_light,
) -> np.ndarray:
    """Convert a wrapped phase in radians to a range distance within one wavelength."""
    wavelength = float(speed_of_light) / float(frequency_hz)
    normalized_phase = np.mod(phase_rad, 2 * np.pi)
    return normalized_phase / (2 * np.pi) * wavelength


def project_distances_to_plane(
    distances: np.ndarray,
    height_offset: float | np.ndarray = 0.0, # 0,75m höjden på rover, och antenner sitter 2,4m
) -> np.ndarray:
    """Project 3D distances onto the antenna plane given a vertical height offset."""
    distances = np.asarray(distances, dtype=float)
    height_offset = np.asarray(height_offset, dtype=float)
    projected = np.sqrt(np.maximum(distances**2 - height_offset[..., np.newaxis] ** 2, 0.0))
    return projected


def _multilateration_2d_single(
    antenna_xy: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    antenna_xy = np.asarray(antenna_xy, dtype=float)
    distances = np.asarray(distances, dtype=float)
    if antenna_xy.ndim != 2 or antenna_xy.shape[1] != 2:
        raise ValueError("antenna_xy must have shape (n_antennas, 2)")
    if distances.ndim != 1 or distances.shape[0] != antenna_xy.shape[0]:
        raise ValueError("distances must have length n_antennas")
    if antenna_xy.shape[0] < 3:
        raise ValueError("At least three antennas are required for 2D multilateration")

    reference = antenna_xy[0] # Phase difference of arrival, with first antenna being the reference
    A = 2.0 * (antenna_xy[1:] - reference)
    b = (
        distances[1:] ** 2
        - distances[0] ** 2
        - np.sum(antenna_xy[1:] ** 2, axis=1)
        + np.sum(reference**2)
    )

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != antenna_xy.shape[0]:
            raise ValueError("weights must have length n_antennas")
        w = weights[1:]
        w = np.where(np.isnan(w), 0.0, w)
        AtWA = (A.T * w) @ A
        AtWb = (A.T * w) @ b
    else:
        AtWA = A.T @ A
        AtWb = A.T @ b

    try:
        position = np.linalg.solve(AtWA, AtWb)
        print(position)
    except np.linalg.LinAlgError:
        position, *_ = np.linalg.lstsq(AtWA, AtWb, rcond=None)
        position = position[:2]
    return position


def _distance_candidates_from_phase_rad(
    phase_rad: np.ndarray,
    frequency_hz: float,
    min_distance: float,
    max_distance: float,
) -> list[np.ndarray]:
    wavelength = float(const.speed_of_light) / float(frequency_hz)
    d_mod = phase_to_distance(phase_rad, frequency_hz)
    candidate_sets: list[np.ndarray] = []
    for d0 in d_mod:
        min_k = int(np.ceil(max(0.0, (min_distance - d0) / wavelength)))
        max_k = int(np.floor((max_distance - d0) / wavelength))
        if max_k < min_k:
            candidate_sets.append(np.empty((0,), dtype=float))
        else:
            candidate_sets.append(d0 + np.arange(min_k, max_k + 1, dtype=float) * wavelength)
    return candidate_sets


def _choose_candidate_distances(
    antenna_xy: np.ndarray,
    candidate_sets: list[np.ndarray],
    reference_index: int = 0,
) -> list[np.ndarray]:
    reference_candidates = candidate_sets[reference_index]
    antenna_deltas = np.linalg.norm(antenna_xy - antenna_xy[reference_index], axis=1)
    chosen_sets: list[np.ndarray] = []
    for k0 in reference_candidates:
        chosen = np.empty(len(candidate_sets), dtype=float)
        chosen[reference_index] = k0
        for i, candidates in enumerate(candidate_sets):
            if i == reference_index:
                continue
            target = k0
            sep = antenna_deltas[i]
            bounds = (target - sep - 0.5 * (candidate_sets[i][1] - candidate_sets[i][0]),
                      target + sep + 0.5 * (candidate_sets[i][1] - candidate_sets[i][0]))
            if candidates.size == 0:
                chosen[i] = np.nan
                continue
            in_range = candidates[(candidates >= bounds[0]) & (candidates <= bounds[1])]
            if in_range.size > 0:
                chosen[i] = in_range[np.argmin(np.abs(in_range - target))]
            else:
                chosen[i] = candidates[np.argmin(np.abs(candidates - target))]
        chosen_sets.append(chosen)
    return chosen_sets


def _score_position(
    position: np.ndarray,
    antenna_xy: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray | None = None,
    prior_position: np.ndarray | None = None,
    prior_weight: float | None = None,
) -> float:
    residuals = np.linalg.norm(position - antenna_xy, axis=1) - distances
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        residuals = residuals * weights
    score = np.nansum(residuals**2)
    if prior_position is not None and prior_weight is not None:
        score += float(prior_weight) * np.sum((position - prior_position) ** 2)
    return score


def estimate_positions_2d_from_phase(
    ds: xr.Dataset,
    frequency_hz: float,
    phase_var: str = "csi_phase_rad",
    amplitude_var: str | None = None,
    height_offset: float | np.ndarray | None = None,
    min_distance: float = 0.0,
    max_distance: float = 8.4,
    prior_position: np.ndarray | None = None,
    prior_weight: float | None = None,
) -> xr.Dataset:
    """Estimate a fused 2D position for each observation from antenna phase."""
    if "antenna_x" not in ds or "antenna_y" not in ds:
        raise ValueError("Dataset must include antenna_x and antenna_y coordinates for 2D positioning")

    antenna_xy = np.stack(
        [ds["antenna_x"].values, ds["antenna_y"].values], axis=-1
    )
    phase_rad = ds[phase_var].values.astype(float)

    weights = None
    if amplitude_var is not None and amplitude_var in ds:
        weights = ds[amplitude_var].values.astype(float)

    positions = np.full((phase_rad.shape[0], 2), np.nan, dtype=float)
    for observation_index, observation_phase in enumerate(phase_rad):
        observation_weights = weights[observation_index] if weights is not None else None
        candidate_results = _candidate_positions_and_scores(
            antenna_xy,
            observation_phase,
            frequency_hz,
            min_distance=min_distance,
            max_distance=max_distance,
            height_offset=height_offset,
            weights=observation_weights,
            prior_position=prior_position,
            prior_weight=prior_weight,
        )
        if candidate_results:
            positions[observation_index, :] = min(candidate_results, key=lambda item: item[1])[0]

    ds_out = xr.Dataset(
        data_vars={
            "position_x": (("observation",), positions[:, 0]),
            "position_y": (("observation",), positions[:, 1]),
        },
        coords={
            "observation": ds["observation"].values,
            "antenna": ds["antenna"].values,
            "source_experiment_id": ds["source_experiment_id"].values,
            "source_cycle_id": ds["source_cycle_id"].values,
        },
    )
    return ds_out


def _candidate_positions_and_scores(
    antenna_xy: np.ndarray,
    phase_rad: np.ndarray,
    frequency_hz: float,
    min_distance: float,
    max_distance: float,
    height_offset: float | np.ndarray | None = None,
    weights: np.ndarray | None = None,
    prior_position: np.ndarray | None = None,
    prior_weight: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    candidate_sets = _distance_candidates_from_phase_rad(phase_rad, frequency_hz, min_distance, max_distance)
    candidate_results: list[tuple[np.ndarray, float]] = []
    #print(_choose_candidate_distances(antenna_xy, candidate_sets))
    for chosen_distances in _choose_candidate_distances(antenna_xy, candidate_sets):
        chosen_distances_2d = chosen_distances
        if height_offset is not None:
            chosen_distances_2d = project_distances_to_plane(chosen_distances, height_offset)

        try:
            position = _multilateration_2d_single(
                antenna_xy,
                chosen_distances_2d,
                weights,
            )
        except ValueError:
            continue

        score = _score_position(
            position,
            antenna_xy,
            chosen_distances_2d,
            weights=weights,
            prior_position=prior_position,
            prior_weight=prior_weight,
        )
        candidate_results.append((position, score))

    return candidate_results

### PLOTTING ##################################################
def _make_likelihoods(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=float)
    shifted = scores - np.nanmin(scores)
    scale = np.nanmean(shifted[shifted > 0])
    if scale == 0 or np.isnan(scale):
        return np.where(np.isnan(shifted), 0.0, 1.0)
    likelihoods = np.exp(-shifted / scale)
    return likelihoods / np.nanmax(likelihoods)


def plot_position_candidates(
    ds: xr.Dataset,
    observation_index: int,
    frequency_hz: float,
    phase_var: str = "csi_phase_rad",
    amplitude_var: str | None = None,
    height_offset: float | np.ndarray | None = None,
    min_distance: float = 0.0,
    max_distance: float = 8.4,
    prior_position: np.ndarray | None = None,
    prior_weight: float | None = None,
    annotate_top: int = 5,
    figsize: tuple[int, int] = (8, 8),
):
    if "antenna_x" not in ds or "antenna_y" not in ds:
        raise ValueError("Dataset must include antenna_x and antenna_y coordinates for plotting")
    if observation_index < 0 or observation_index >= ds.sizes["observation"]:
        raise IndexError("observation_index is out of range")

    antenna_xy = np.stack([ds["antenna_x"].values, ds["antenna_y"].values], axis=-1)
    phase_rad = ds[phase_var].values.astype(float)[observation_index]
    weights = None
    if amplitude_var is not None and amplitude_var in ds:
        weights = ds[amplitude_var].values.astype(float)[observation_index]

    true_position = None
    if "rover_x" in ds and "rover_y" in ds:
        true_x = float(ds["rover_x"].values[observation_index])
        true_y = float(ds["rover_y"].values[observation_index])
        if not (np.isnan(true_x) or np.isnan(true_y)):
            true_position = np.array([true_x, true_y], dtype=float)

    candidates = _candidate_positions_and_scores(
        antenna_xy,
        phase_rad,
        frequency_hz,
        min_distance=min_distance,
        max_distance=max_distance,
        height_offset=height_offset,
        weights=weights,
        prior_position=prior_position,
        prior_weight=prior_weight,
    )
    if not candidates:
        raise RuntimeError("No candidate positions found for this observation")

    positions = np.asarray([pos for pos, _ in candidates], dtype=float)
    scores = np.asarray([score for _, score in candidates], dtype=float)
    likelihoods = _make_likelihoods(scores)
    best_index = np.nanargmin(scores)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=likelihoods,
        cmap="plasma",
        s=60,
        edgecolor="k",
        alpha=0.8,
        label="candidates",
    )
    ax.scatter(
        antenna_xy[:, 0],
        antenna_xy[:, 1],
        marker="^",
        c="black",
        s=100,
        label="antennas",
    )
    ax.scatter(
        positions[best_index, 0],
        positions[best_index, 1],
        marker="*",
        c="red",
        s=180,
        label="best",
    )
    if prior_position is not None:
        ax.scatter(
            prior_position[0],
            prior_position[1],
            marker="x",
            c="green",
            s=150,
            label="prior",
        )

    if true_position is not None:
        ax.scatter(
            true_position[0],
            true_position[1],
            marker="D",
            c="cyan",
            s=130,
            edgecolor="black",
            label="true",
        )

    for label_index in np.argsort(scores)[:annotate_top]:
        ax.annotate(
            f"{likelihoods[label_index]:.2f}",
            (positions[label_index, 0], positions[label_index, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Observation {observation_index} candidate positions")
    ax.legend(loc="best")
    plt.colorbar(scatter, ax=ax, label="relative likelihood")
    ax.axis("equal")
    return fig, ax

