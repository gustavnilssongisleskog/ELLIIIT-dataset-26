from path_generator.dijkstra import dijkstra
from path_generator.graph import generate_graph
from processing.tutorials.csi_plot_utils import open_dataset, positions_for_experiments
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


ds, dataset_path = open_dataset()


rng = np.random.default_rng(123)
num_measurements = 5011
train = rng.integers(0, 2, num_measurements, dtype=bool)
test = ~train
graph_train, node_to_measurement_train = generate_graph(ds, include=train)
graph_test, node_to_measurement_test = generate_graph(ds, include=test)

print(len(graph_train))
print(len(graph_test))
N_train = len(graph_train)
N_test = len(graph_test)


for plot in range(25):
    row = plot // 5
    col = plot % 5
    plt.subplot(5, 5,plot+1)
    path = []
    midpoints = rng.integers(0, N_train, 11).tolist()
    for i in range(10):
        start, end = midpoints[i:i+2]
        # start, end = map(int, rng.choice(N_train, 2, replace=False))
        new_path = dijkstra(graph_train, [], start, end)
        if new_path is None:
            continue
        path.extend(new_path[:-1])
        start = end
    path.append(midpoints[-1])
    print(midpoints)
    print(path[0], path[-1])

    path_positions = []

    for node in path:
        exp, cycle = node_to_measurement_train[node]
        point = (ds.data_vars["rover_x"].data[exp, cycle],
                ds.data_vars["rover_y"].data[exp, cycle])
        path_positions.append(point)

    midpoint_positions = []
    for node in midpoints:
        exp, cycle = node_to_measurement_train[node]
        point = (ds.data_vars["rover_x"].data[exp, cycle],
                ds.data_vars["rover_y"].data[exp, cycle])
        midpoint_positions.append(point)

    path_positions = np.array(path_positions).T
    midpoint_positions = np.array(midpoint_positions).T
    print(np.linalg.norm(np.diff(path_positions, axis=1), axis=0).max())

    plt.plot(path_positions[0], path_positions[1], "-o")
    plt.plot(midpoint_positions[0], midpoint_positions[1], "or")

plt.show()