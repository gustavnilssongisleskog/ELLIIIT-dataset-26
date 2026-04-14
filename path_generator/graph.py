import numpy as np
import xarray as xr


def generate_graph(ds: xr.Dataset, num_neighbors: int = 5, max_radius: float = 0.15, include: np.ndarray = None):
    """Generate a graph from the rover positions.

    Args:
        ds (xr.Dataset): Dataset
        num_neighbors (int, optional): The max number of neighbors each rover position is allowed to have. Defaults to 10.
        max_radius (float, optional): The max distance to neighbors. This is what may cause number of neighbors to be < num_neighbors. Defaults to 0.05.
        include (np.ndarray, optional): Boolean mask showing which rover positions to include. Should have length 5011... (i.e. number of measurements that are not NaN). Defaults to None.

    Returns:
        tuple[list[list[tuple[float, int]]]], list[tuple[int, int]]: Graph as adjacency list with edges (weight, neighbor) and a list mapping from node to measurement.
    """
    available: np.ndarray = np.array(ds.position_available.data, dtype=bool)
    available_flat = np.reshape(available, -1)
    num_measurements = np.count_nonzero(available)
    if include is None:
        include = np.ones(num_measurements, dtype=bool)
    N = np.count_nonzero(include)
    node_to_measurement = []
    cur_measurement = 0
    for i in range(available.shape[0]):
        for j in range(available.shape[1]):
            if not available[i, j]:
                continue
            if include[cur_measurement]:
                node_to_measurement.append((i, j))
            cur_measurement += 1

    # node_to_measurement[node] = (EXPXXX, cycle_id)
    dimensions = ["rover_x", "rover_y", "rover_z"]
    positions = np.empty((3, N))
    for i, dimension in enumerate(dimensions):
        all_vals = np.reshape(ds.data_vars[dimension].data, -1)
        positions[i, :] = all_vals[available_flat][include]

    graph = []
    for node in range(N):
        dists = np.linalg.norm(positions - positions[:, node, np.newaxis], axis=0)
        neighbors = np.argsort(dists)[1:num_neighbors + 1]
        neighbors = neighbors[dists[neighbors] < max_radius]
        neighbor_dists = dists[neighbors].tolist()
        neighbors = neighbors.tolist()

        graph.append(list(zip(neighbor_dists, neighbors)))

    return graph, node_to_measurement