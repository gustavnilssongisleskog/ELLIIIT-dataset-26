import xarray as xr
import pickle
from path_generator.path_generator import format_walks


def load_pickle(file_name: str, ds: xr.Dataset, formatting: bool=True):
    if not file_name.endswith(".pickle"):
        file_name += ".pickle"
    with open(file_name, "rb") as file:
        walks = pickle.load(file)

    if not formatting:
        return walks

    walks_formatted = format_walks(walks, ds)
    return walks_formatted


def save_pickle(file_name: str, walks):
    if not file_name.endswith(".pickle"):
        file_name += ".pickle"
    with open(file_name, "wb") as file:
        pickle.dump(walks, file)