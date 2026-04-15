from pathlib import Path
import importlib.util
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import samplerate
import numpy as np
import xarray as xr
import requests
import yaml
import matplotlib.pyplot as plt
import librosa as lbr
from scipy.signal import butter, lfilter, correlate, firwin, filtfilt, find_peaks, peak_prominences
import plotly.graph_objects as go
import plotly.colors
import plotly.io as pio
from tqdm import tqdm
import json
from scipy.optimize import curve_fit, least_squares
from path_generator.pickler import load_pickle

pio.renderers.default = "browser"

import plotly.graph_objects as go

plt.rcParams['interactive']

# Get the directory where this script is located
script_dir = Path(__file__).parent
config_path = script_dir / 'config.json'

with open(config_path) as json_file:
    config = json.load(json_file)

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


MICROPHONE_POSITIONS_URL = (
    "https://raw.githubusercontent.com/techtile-by-dramco/"
    "techtile-description/refs/heads/main/geometry/"
    "techtile_microphone_locations.yml"
)


def load_microphone_positions(positions_url: str = MICROPHONE_POSITIONS_URL) -> dict[str, np.ndarray]:
    response = requests.get(positions_url, timeout=20)
    response.raise_for_status()
    config = yaml.safe_load(response.text)

    positions: dict[str, np.ndarray] = {}
    for entry in config["microphones"]:
        microphone = str(entry["tile"]).upper()
        positions[microphone] = np.array([entry["x"], entry["y"], entry["z"]], dtype=float)
    return positions


def resample_chirp(fs_source, fs_mic, chirp_orig):
    """
    Downsample original chirp signal if fs source =/= fs mic
    :param fs_source: sample frequency at speaker
    :param fs_mic: sample frequency at receiver
    :param chirp_orig: original chirp signal
    :return: chirp_orig_resampl: resampled original chirp signal
    """
    if fs_source != fs_mic:
        fs_ratio = float(fs_mic) / float(fs_source)
        new_length = int(fs_ratio * chirp_orig.shape[0])
        re_sampled_orig_chirp_signal = np.zeros((1, new_length))

        # why sinc: http://www.mega-nerd.com/SRC/api_misc.html#ErrorReporting
        re_sampled_orig_chirp_signal = samplerate.resample(chirp_orig, fs_ratio, "sinc_best")
        chirp_orig_resampl = re_sampled_orig_chirp_signal

    else:  # in the sample rates are identical, no up/down sampling is required
        chirp_orig_resampl = chirp_orig

    return chirp_orig_resampl


def plot_received_signal_mic(sample_index, waveform_values, EXPERIMENT_ID, CYCLE_ID, mic_label):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sample_index, waveform_values, color="navy", linewidth=1.2)
    ax.set_title(
        f"Acoustic waveform for {EXPERIMENT_ID} / cycle {CYCLE_ID} / mic {mic_label}"
    )
    ax.set_xlabel("sample_index")
    ax.set_ylabel("values")
    ax.grid(True, alpha=0.25)
    plt.show()

def get_selected_mic_positions(microphone_positions, MICROPHONE_LABEL):
    selected_mic_positions = {}
    missing_mics = []

    for mic_label in MICROPHONE_LABEL:
        mic_pos = microphone_positions.get(mic_label.upper())
        if mic_pos is not None:
            selected_mic_positions[mic_label] = mic_pos
        else:
            missing_mics.append(mic_label)

    if missing_mics:
        print(f"Missing microphone labels (ignored): {missing_mics}")

    print(f"Loaded positions for {len(selected_mic_positions)} microphones")
    # Loop through all selected microphones
    # for mic_label, position in selected_mic_positions.items():
    #     # print(f"{mic_label}: {position}")

    # Example usage: selected_mic_positions['D06'] returns the [x, y, z] position array
    return selected_mic_positions

def select_anchors(waveform_values: np.ndarray, microphone_label: str, experiment_id: str, cycle_id: int, n_selected_anchors: int = 1, sumrate_threshold: float = 2.8) -> dict[str, object]:
    values = np.asarray(waveform_values, dtype=float)
    signal_rms = float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0
    abs_values = np.abs(values)
    signal_sumrate = float(np.sum(abs_values[abs_values > sumrate_threshold])) if values.size else 0.0

    return {
        "experiment_id": str(experiment_id),
        "cycle_id": int(cycle_id),
        "microphone_label": str(microphone_label),
        "n_selected_anchors": int(n_selected_anchors),
        "signal_rms": signal_rms,
        "signal_sumrate": signal_sumrate,
        "waveform_values": values,
    }

def select_top_anchors(anchor_candidates, anchor_selection_method, n_selected_anchors):
    """Select top anchors based on the chosen method and return as dictionary."""
    sort_key = 'signal_rms' if anchor_selection_method == 'rms' else 'signal_sumrate'
    selected_anchors = sorted(anchor_candidates, key=lambda x: x[sort_key], reverse=True)[:n_selected_anchors]
    selected_anchors_dict = {anchor['microphone_label']: anchor for anchor in selected_anchors}
    return selected_anchors_dict, sort_key


def print_selected_anchors_info(selected_anchors_dict, sort_key, anchor_selection_method):
    """Print information about the selected anchors."""
    metric_name = 'RMS' if anchor_selection_method == 'rms' else 'Sum Rate'
    print(f"Selected {len(selected_anchors_dict)} anchors with strongest signals ({metric_name}):")
    for mic_label, anchor in selected_anchors_dict.items():
        value = anchor[sort_key]
        rms = anchor['signal_rms']
        sumrate = anchor['signal_sumrate']
        waveform = anchor['waveform_values']
        
        print(f"  {mic_label}: {metric_name}={value:.4f} (RMS={rms:.4f}, SumRate={sumrate:.2f}, Waveform length={len(waveform)})")


def read_transmit_chirp():
    original_chirp = "post-processing/acoustic-pos/Old_functions/chirp.wav"
    chirp_orig, fs_source = lbr.load(original_chirp, sr=None)
    duration_chirp = lbr.get_duration(y=chirp_orig, sr=fs_source)

    return fs_source, chirp_orig, duration_chirp


def butter_highpass(highcut, fs, order=4):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='high')
    return b, a

def butter_highpass_filter(data, highcut, fs, order=4):
    b, a = butter_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_received_signal_comb(sample_index, waveform_values, waveform_values_filtered, EXPERIMENT_ID, CYCLE_ID, mic_label):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sample_index, waveform_values, color="navy", linewidth=1.2, alpha=0.7, label="Received Signal")
    ax.plot(sample_index, waveform_values_filtered, color="red", linewidth=1.2, alpha=0.7, label="High-pass Filtered")
    ax.set_title(
        f"Acoustic waveforms for {EXPERIMENT_ID} / cycle {CYCLE_ID} / mic {mic_label}"
    )
    ax.set_xlabel("sample_index")
    ax.set_ylabel("values")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.show()

def gain_equaliser(TX, RX_matrix, WRX_min, WRX_max):
    """
    Adjust the gain of multiple received signals to match the transmitted signal for better correlation.
    :param TX: transmitted signal (original chirp)
    :param RX_matrix: 2D array where each row is a received signal
    :param WRX_min: start index for RMS window
    :param WRX_max: end index for RMS window
    :return: RX_matrix adjusted to have similar amplitude as TX
    """
    tx_rms = np.sqrt(np.mean(TX ** 2))
    windowed = RX_matrix[:, WRX_min:WRX_max]
    rx_rms = np.sqrt(np.mean(windowed ** 2, axis=1))
    gains = np.where(rx_rms > 0, tx_rms / rx_rms, 1.0)
    RX_adjusted = RX_matrix * gains[:, np.newaxis]
    return RX_adjusted
    
def plot_received_signal_comb_diff(sample_index_1, sample_index_2, waveform_values, waveform_values_filtered, EXPERIMENT_ID, CYCLE_ID, mic_label):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sample_index_1, waveform_values, color="navy", linewidth=1.2, alpha=0.7, label="Received Signal")
    ax.plot(sample_index_2, waveform_values_filtered, color="red", linewidth=1.2, alpha=0.7, label="Equalised Transmitted Signal")
    ax.set_title(
        f"Acoustic waveforms for {EXPERIMENT_ID} / cycle {CYCLE_ID} / mic {mic_label}"
    )
    ax.set_xlabel("sample_index")
    ax.set_ylabel("values")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.show()


def plot_adjusted_waveforms_with_chirp(selected_anchors_dict, chirp_orig_resampl, experiment_id, cycle_id):
      
    fig = go.Figure()
    
    # Add original chirp
    fig.add_trace(go.Scatter(
        x=list(range(len(chirp_orig_resampl))), 
        y=chirp_orig_resampl, 
        mode='lines', 
        name='Original Chirp',
        line=dict(color='blue', width=2)
    ))
    
    # Use Plotly's qualitative color palette for automatic color assignment
    colors = plotly.colors.qualitative.Plotly
    
    # Add each adjusted waveform
    for i, (mic_label, anchor) in enumerate(selected_anchors_dict.items()):
        adjusted = anchor['adjusted_waveform']
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=list(range(len(adjusted))), 
            y=adjusted, 
            mode='lines', 
            name=f'Adjusted {mic_label}',
            line=dict(color=color, width=1)
        ))
    
    fig.update_layout(
        title=f'Adjusted Waveforms and Original Chirp - {experiment_id} Cycle {cycle_id}',
        xaxis_title='Sample Index',
        yaxis_title='Amplitude',
        legend_title='Signals'
    )

    fig.update_traces(opacity=.6)
    
    figs_dir = Path(__file__).parent / "Figs"
    fig.write_html(str(figs_dir / "plot_received_and_filtered_acoustic_signals.html"))


def norm_correlate(TX, rx_matrix, n):
    # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
    corr_val = np.abs(np.correlate(TX, rx_matrix[n, :], "full"))

    # Normalize y peak values to have smaller values (not 1e9)
    corr_val = corr_val / np.max(corr_val)
    return corr_val


def get_peak_prom_index(LPF_val, corr_val):
    # find all peaks and calculate promineces
    peaks, _ = find_peaks(LPF_val)
    prominences = peak_prominences(LPF_val, peaks)[0]
    most_prom = prominences[prominences > config["peak_prominence_factor"]][-1]
    most_prom_idx = np.where(np.around(prominences, decimals=5) == np.around(most_prom,
                                                                             decimals=5))  # Select first index from row which > PP Threshold [0][0]
    idx_peak_samples = peaks[most_prom_idx]

    # print('PPF: ', most_prom)
    # print('Sample index with selected peak: ', idx_peak_samples)

    # calculate height of each peak's contour line
    contour_heights = LPF_val[peaks] - prominences

    index_opt_general = idx_peak_determination_PP(corr_val, idx_peak_samples)
    return index_opt_general


def idx_peak_determination_PP(corr_val, max_corr_index):
    """
    Determines the peak index value after already selecting the most prominant peak
    :param corr_val: original correlation values
    :param max_corr_index: the selected index value for the peak from LPF curve
    :return: index: index of peak value mapped after maximum of fitting curve value
    """
    # Select all peaks to use later for mapping from max to most close peak
    peaks_index, _ = find_peaks(x=corr_val)

    # Find closed match with peak indexes
    eucl_dist = np.abs(peaks_index - max_corr_index)
    index = np.argmin(eucl_dist)
    max_corr_index_mapped = peaks_index[index]

    return max_corr_index_mapped

def LPF(x, typeF, order, cutoff, fs):
    """
    Function to add LPF
        x: input signal
        typeF: filter type ('bandpass', 'lowpass', 'highpass', 'bandstop')
        order: length/order of the filter
        cutoff: cutoff frequency
        fs: sample frequency
        return xf: the filtered signal
    """

    # Use signal.firwin to generate the filter coefficients
    b = firwin(order, cutoff, pass_zero=typeF, fs=fs)

    # Use signal.filtfilt to filter x
    xf = 2 * filtfilt(b, 1, x)

    # Adjust gain
    xf_gained = (np.max(x) / np.max(xf)) * xf

    return xf_gained


def get_wake_up_part(RX, TX, fs):
    """
    Select wake-up part of RX using the same logic as demo code, with safe bounds.
    """
    n_wake_up_samples = config["wake_up_duration"] * fs
    n_wake_up_samples_eff = int(min(n_wake_up_samples, np.size(TX)))
    wake_up_at_sample = int(np.size(TX) - n_wake_up_samples_eff)
    rx_audio_wake = RX[:, wake_up_at_sample:int(wake_up_at_sample + n_wake_up_samples_eff)]
    return rx_audio_wake, n_wake_up_samples_eff, wake_up_at_sample

def get_corr_with_LPF_curve(TX, selected_anchors_dict):
    """
    Get the correlation LPF and index of selected correlation peak
    :param TX: TX chirp
    :param selected_anchors_dict: dict of selected anchors with adjusted waveforms
    :param fs: sampling frequency
    :param config: configuration dict
    :return: pulse compression values, LPF values and selected peak of the pulse compression
    """
    print("Pulse compression and LPF\n")
    
    # Extract adjusted waveforms into a matrix for efficient processing
    rx_audio_full = np.array([anchor['adjusted_waveform'] for anchor in selected_anchors_dict.values()])
    rx_audio_amp, _, _ = get_wake_up_part(rx_audio_full, TX, config['fs_mic'])

    n_nodes = rx_audio_amp.shape[0]  # amount of anchors

    size_arrays = np.size(np.correlate(TX, rx_audio_amp[0, :], "full"))
    pulse_compr_all = np.empty(size_arrays)
    LPF_all = np.empty(size_arrays)
    corr_index_array = np.array([])


    for rx in tqdm(range(n_nodes)):
        # Pulse compression
        corr_val = norm_correlate(TX, rx_audio_amp, rx)
        pulse_compr_all = np.vstack((pulse_compr_all, corr_val))

        # Add LPF to determine envelope
        LPF_val = LPF(corr_val, 'lowpass', 10, config['fs_mic']/26, config['fs_mic'])  # 70, fs_mic/35   #1000, 5000, #1000, 10000
        LPF_all = np.vstack((LPF_all, LPF_val))

        # Get the peak prominence index
        index_opt_general = get_peak_prom_index(LPF_val, corr_val)
        corr_index_array = np.append(corr_index_array, index_opt_general)

    pulse_compr_all = np.delete(pulse_compr_all, 0, 0)
    LPF_all = np.delete(LPF_all, 0, 0)

    return pulse_compr_all, LPF_all, corr_index_array


def plot_pulsecomp_and_lpf(pulse_compr_all, LPF_all, corr_index_array, fs, selected_anchors_dict, TX, true_distances=None, plot_full_pulse_compression=True):
    """
    Plot pulse compression values, LPF values, and vertical lines for chosen peaks, with x-axis in meters.
    """
    temp = config["temperature"]
    v_sound = 20 * np.sqrt(273 + temp)
    n_wake_up_samples = config["wake_up_duration"] * fs
    n_wake_up_samples_eff = min(n_wake_up_samples, np.size(TX))

    x_len = pulse_compr_all.shape[1]      # = N_tx + N_rx - 1
    N_tx = int(np.size(TX))
    N_rx = x_len - N_tx + 1              # length of each received waveform
    n_anchors = pulse_compr_all.shape[0]
    anchor_names = list(selected_anchors_dict.keys())

    corr_idx_array = np.arange(0, x_len, 1)
    wake_up_at_sample = int(np.size(TX) - n_wake_up_samples_eff)
    eff_starts_all_samples = (corr_idx_array + 1) - n_wake_up_samples_eff
    delta_distances_array = wake_up_at_sample - eff_starts_all_samples
    x_signal = (delta_distances_array / fs) * v_sound

    data = []
    colors = plotly.colors.qualitative.Plotly
    lpf_dash = 'dash' if plot_full_pulse_compression else 'solid'
    for RX_idx in range(n_anchors):
        color = colors[RX_idx % len(colors)]
        if plot_full_pulse_compression:
            data.append(go.Scatter(
                x=x_signal, y=pulse_compr_all[RX_idx, :],
                mode='lines', name=f'Pulse Comp {anchor_names[RX_idx]}',
                line=dict(color=color, width=1)
            ))
        data.append(go.Scatter(
            x=x_signal, y=LPF_all[RX_idx, :],
            mode='lines', name=f'LPF {anchor_names[RX_idx]}',
            line=dict(color=color, width=2, dash=lpf_dash)
        ))

    fig = go.Figure(data=data)

    # Stagger annotation heights so labels do not overlap
    label_y_offsets = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    for i, peak_idx in enumerate(corr_index_array):
        peak_idx = int(peak_idx)
        if peak_idx < 0 or peak_idx >= x_len:
            continue
        distance = float(x_signal[peak_idx])
        if distance < 0:
            continue
        peak_y = float(LPF_all[i, peak_idx])
        annotation_y = min(1.15, peak_y + label_y_offsets[i % len(label_y_offsets)])
        fig.add_vline(x=distance, line_color='red', line_width=1, line_dash='dash')
        fig.add_annotation(
            x=distance, y=annotation_y,
            xref='x', yref='y',
            text=f'{anchor_names[i]}: {distance:.3f} m',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='red', borderwidth=1,
            yanchor='bottom'
        )

    if true_distances is not None:
        gt_label_offsets = [0.02, 0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58]
        for i, anchor_name in enumerate(anchor_names):
            if anchor_name not in true_distances:
                continue
            dist_gt = float(true_distances[anchor_name])
            if dist_gt < 0:
                continue
            fig.add_vline(x=dist_gt, line_color='green', line_width=1, line_dash='dot')
            fig.add_annotation(
                x=dist_gt,
                y=min(1.15, gt_label_offsets[i % len(gt_label_offsets)]),
                xref='x',
                yref='y',
                text=f'GT {anchor_name}: {dist_gt:.3f} m',
                showarrow=False,
                bgcolor='rgba(240,255,240,0.85)',
                bordercolor='green',
                borderwidth=1,
                yanchor='bottom'
            )

    fig.update_layout(
        title='Pulse Compression and LPF with Chosen Peaks',
        xaxis_title='Ranging Distance (m)',
        yaxis_title='Normalized Correlation Value',
        yaxis=dict(range=[0, 1.3])
    )
    fig.update_xaxes(range=[0, float(np.max(x_signal))])

    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    fig.write_html(str(figs_dir / "pulse_compression_lpf_plot.html"))

    #fig.show()


def load_experiment_setup() -> dict:
    """Load all experiment setup values from config, microphone positions, and chirp file."""
    n_selected_ans = config['number_of_selected_anchors']
    anchor_selection_method = config['anchor_selection_method']
    sumrate_threshold = config['sumrate_threshold']
    v_sound = 20 * np.sqrt(273 + config["temperature"])
    fs_mic = config['fs_mic']
    plot_full_pulse_compression = config['plot_full_pulse_compression']
    mic_processing_workers = config.get('mic_processing_workers')

    if config['use_all_mics']:
        mic_labels = config['all_mics']
    else:
        mic_labels = config['used_mics']

    microphone_positions = load_microphone_positions()
    selected_mic_positions = get_selected_mic_positions(microphone_positions, mic_labels)

    fs_source, chirp_orig, duration_chirp = read_transmit_chirp()
    chirp_orig_resampl = resample_chirp(fs_source, fs_mic, chirp_orig) * 10

    return {
        'n_selected_ans': n_selected_ans,
        'anchor_selection_method': anchor_selection_method,
        'sumrate_threshold': sumrate_threshold,
        'v_sound': v_sound,
        'fs_mic': fs_mic,
        'fs_source': fs_source,
        'duration_chirp': duration_chirp,
        'selected_mic_positions': selected_mic_positions,
        'chirp_orig_resampl': chirp_orig_resampl,
        'plot_full_pulse_compression': plot_full_pulse_compression,
        'mic_processing_workers': mic_processing_workers,
    }


def _process_single_microphone(task: tuple[str, str, int, str | Path | None, float, int, float]) -> dict[str, object]:
    mic_label, experiment_id, cycle_id, dataset_path, fs_mic, n_selected_ans, sumrate_threshold = task
    _waveform, waveform_values, _sample_index = get_acoustic_waveform(experiment_id, cycle_id, mic_label, dataset_path)
    waveform_filtered = butter_highpass_filter(waveform_values, 15000, fs_mic, 10)
    return select_anchors(waveform_filtered, mic_label, experiment_id, cycle_id, n_selected_ans, sumrate_threshold)


def collect_anchor_candidates(selected_mic_positions: dict, experiment_id: str, cycle_id: int, dataset_path, chirp_orig_resampl: np.ndarray, fs_mic: float, n_selected_ans: int, sumrate_threshold: float, n_workers: int | None = None) -> list:
    """Collect anchor candidates for all selected microphones, optionally using multiprocessing."""
    mic_labels = list(selected_mic_positions)
    if not mic_labels:
        return []

    if n_workers is None:
        cpu_count = os.cpu_count() or 1
        n_workers = min(len(mic_labels), max(1, cpu_count - 1))
    else:
        n_workers = max(1, int(n_workers))

    tasks = [
        (mic_label, experiment_id, int(cycle_id), dataset_path, fs_mic, n_selected_ans, sumrate_threshold)
        for mic_label in mic_labels
    ]

    if n_workers <= 1 or len(mic_labels) == 1:
        anchor_candidates = []
        for task in tqdm(tasks, total=len(tasks), desc="Selecting anchor candidates", unit="mic"):
            anchor_candidates.append(_process_single_microphone(task))
        return anchor_candidates

    try:
        results_by_mic = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single_microphone, task): task[0]
                for task in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing microphones ({n_workers} proc)", unit="mic"):
                mic_label = futures[future]
                results_by_mic[mic_label] = future.result()
        return [results_by_mic[mic_label] for mic_label in mic_labels]
    except Exception as exc:
        print(f"Multiprocessing failed ({exc}); falling back to sequential microphone processing.")
        anchor_candidates = []
        for task in tqdm(tasks, total=len(tasks), desc="Selecting anchor candidates", unit="mic"):
            anchor_candidates.append(_process_single_microphone(task))
        return anchor_candidates


def apply_gain_equalisation(selected_anchors_dict: dict, chirp_orig_resampl: np.ndarray, wmin: int = 8000, wmax: int = 10000) -> dict:
    """Apply gain equalisation to all selected anchors and store adjusted waveforms in-place."""
    signals = np.array([anchor['waveform_values'] for anchor in selected_anchors_dict.values()])
    adjusted_signals = gain_equaliser(chirp_orig_resampl, signals, wmin, wmax)
    for i, mic_label in enumerate(selected_anchors_dict.keys()):
        selected_anchors_dict[mic_label]['adjusted_waveform'] = adjusted_signals[i]
    return selected_anchors_dict


def compute_true_distances(position: dict, selected_anchors_dict: dict, selected_mic_positions: dict) -> dict | None:
    """Compute ground-truth distances from rover position to each selected microphone."""
    if not position["position_available"]:
        return None
    rover_xyz = np.array([position['rover_x'], position['rover_y'], position['rover_z']], dtype=float)
    true_distances = {}
    for mic_label in selected_anchors_dict.keys():
        mic_xyz = np.asarray(selected_mic_positions[mic_label], dtype=float)
        true_distances[mic_label] = float(np.linalg.norm(mic_xyz - rover_xyz))
    return true_distances


def compute_ranging(corr_index_array: np.ndarray, chirp_orig_resampl: np.ndarray, fs_mic: float, v_sound: float) -> np.ndarray:
    """Convert correlation peak indices to measured distances (m)."""
    n_wake_up_samples = config["wake_up_duration"] * fs_mic
    n_wake_up_samples_eff = int(min(n_wake_up_samples, np.size(chirp_orig_resampl)))
    wake_up_at_sample = int(np.size(chirp_orig_resampl) - n_wake_up_samples_eff)
    eff_start_samp_chirp = (((corr_index_array + 1) - n_wake_up_samples_eff)[np.newaxis]).T
    delta_sample = wake_up_at_sample - eff_start_samp_chirp
    distances_meas = (delta_sample / fs_mic) * v_sound
    return distances_meas


def print_ranging_errors(distances_meas: np.ndarray, true_distances: dict | None, selected_anchors_dict: dict) -> None:
    """Print measured distances and, if available, ranging errors against ground truth."""
    #print("Measured distances (m):", distances_meas)
    if true_distances is None:
        return
    #print("Theoretical distances (m):")
    ranging_errors = []
    for idx, mic_label in enumerate(selected_anchors_dict.keys()):
        meas_val = float(distances_meas[idx, 0])
        gt_val = float(true_distances[mic_label])
        error_val = meas_val - gt_val
        ranging_errors.append(abs(error_val))
        # print(f"  {mic_label}: measured={meas_val:.3f}, theoretical={gt_val:.3f}, error={error_val:+.3f}")
    mean_ranging_error = float(np.mean(ranging_errors))
    p95_ranging_error = float(np.percentile(ranging_errors, 95))
    print(f"Mean absolute ranging error (m): {mean_ranging_error:.3f}")
    print(f"P95 absolute ranging error (m): {p95_ranging_error:.3f}")



def LS_positioning(anchor_positions, distances, x0, selected_mic_positions=None):
    """
    Least Squares positioning estimate. Minimise the difference in measured distance to anchor point and estimated distances to do positioning
    :param anchor_positions: e.g. np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]])
    :param distances: e.g. np.array([1.5, 1.3, 1.2, 1.4, 0.7])
    :param x0: Initial guess for the position of the point e.g. np.array([0.5, 0.5, 0.5])
    :param selected_mic_positions: optional dict with microphone xyz positions keyed by label
    :return: the estimated position
    """
    if isinstance(anchor_positions, dict):
        first_value = next(iter(anchor_positions.values()), None)
        if first_value is None:
            raise ValueError("anchor_positions is empty")

        # If dict already maps mic label -> [x, y, z], use it directly.
        if np.asarray(first_value).shape == (3,):
            anchor_positions = np.asarray(list(anchor_positions.values()), dtype=float)
        else:
            # If dict is selected_anchors_dict, map labels using selected_mic_positions.
            if selected_mic_positions is None:
                raise ValueError(
                    "selected_mic_positions must be provided when anchor_positions is selected_anchors_dict"
                )
            anchor_labels = list(anchor_positions.keys())
            anchor_positions = np.asarray([selected_mic_positions[label] for label in anchor_labels], dtype=float)
    else:
        anchor_positions = np.asarray(anchor_positions, dtype=float)

    distances = np.asarray(distances, dtype=float).reshape(-1)
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    if anchor_positions.ndim != 2 or anchor_positions.shape[1] != 3:
        raise ValueError("anchor_positions must have shape (N, 3)")
    if distances.size != anchor_positions.shape[0]:
        raise ValueError(
            f"Mismatch between distances ({distances.size}) and anchors ({anchor_positions.shape[0]})"
        )

    # Define the function to minimize
    def minimise_function_LS(x, anchor_positions, distances):
        # Calculate the squared differences between the estimated distances and the actual distances
        return np.sqrt(np.sum((np.linalg.norm(anchor_positions - x, axis=1) - distances) ** 2))
    # Call the least squares optimizer
    res = least_squares(minimise_function_LS, x0, args=(anchor_positions, distances))
    return res.x


def append_dict_record(file_path: Path, record: dict) -> int:
    """Append one dictionary record to a .npy file that stores a list of dictionaries."""
    if file_path.exists():
        existing = np.load(file_path, allow_pickle=True).tolist()
        if isinstance(existing, dict):
            existing = [existing]
        elif not isinstance(existing, list):
            existing = list(existing)
    else:
        existing = []

    existing.append(record)
    np.save(file_path, np.array(existing, dtype=object))
    return len(existing)


def compute_position_error_metrics(position: dict, position_estimate: np.ndarray) -> dict:
    """Compute 3D/2D position error metrics from GT position and LS estimate."""
    estimated_position_xyz = np.asarray(position_estimate, dtype=float).reshape(-1)
    estimated_position_xy = estimated_position_xyz[:2]

    metrics = {
        "position_available": bool(position["position_available"]),
        "true_position_xyz": None,
        "true_position_xy": None,
        "estimated_position_xyz": estimated_position_xyz,
        "estimated_position_xy": estimated_position_xy,
        "estimation_error_vector_xyz": None,
        "position_error_m": float(np.nan),
        "position_error_2d_m": float(np.nan),
    }

    if metrics["position_available"]:
        true_position_xyz = np.array([position['rover_x'], position['rover_y'], position['rover_z']], dtype=float)
        true_position_xy = true_position_xyz[:2]
        estimation_error_vector = estimated_position_xyz - true_position_xyz
        metrics.update(
            {
                "true_position_xyz": true_position_xyz,
                "true_position_xy": true_position_xy,
                "estimation_error_vector_xyz": estimation_error_vector,
                "position_error_m": float(np.linalg.norm(estimation_error_vector)),
                "position_error_2d_m": float(np.linalg.norm(estimated_position_xy - true_position_xy)),
            }
        )

    return metrics


def print_position_error_report(metrics: dict) -> None:
    """Print a compact position error report."""
    if metrics["position_available"]:
        print(
            "Position error report:\n"
            f"  True position      : {metrics['true_position_xyz']}\n"
            f"  Estimated position : {metrics['estimated_position_xyz']}\n"
            f"  Error vector (m)   : {metrics['estimation_error_vector_xyz']}\n"
            f"  Euclidean error (3D, m): {metrics['position_error_m']:.4f}\n"
            f"  Euclidean error (2D, m): {metrics['position_error_2d_m']:.4f}"
        )
    else:
        print("Position error report: GT position unavailable, saving NaN error values.")


def build_position_record(experiment_id: str, cycle_id: int, path_id: int | str, metrics: dict) -> dict:
    """Create a serializable position record dictionary."""
    return {
        "experiment_id": str(experiment_id),
        "cycle_id": int(cycle_id),
        "path_id": int(path_id) if str(path_id).isdigit() else str(path_id),
        "position_available": bool(metrics["position_available"]),
        "ground_truth_position_xyz": None if metrics["true_position_xyz"] is None else metrics["true_position_xyz"].tolist(),
        "ground_truth_position_xy": None if metrics["true_position_xy"] is None else metrics["true_position_xy"].tolist(),
        "estimated_position_xyz": metrics["estimated_position_xyz"].tolist(),
        "estimated_position_xy": metrics["estimated_position_xy"].tolist(),
        "position_error_m": float(metrics["position_error_m"]),
        "position_error_2d_m": float(metrics["position_error_2d_m"]),
    }


def build_ranging_record(experiment_id: str, cycle_id: int, path_id: int | str, metrics: dict, distances_meas: np.ndarray, true_distances: dict | None, selected_anchors_dict: dict) -> dict:
    """Create a serializable ranging record dictionary with per-anchor details."""
    measured_distances = np.asarray(distances_meas, dtype=float).reshape(-1)
    anchor_labels = list(selected_anchors_dict.keys())

    per_anchor_ranging = []
    abs_errors = []
    for idx, mic_label in enumerate(anchor_labels):
        measured_m = float(measured_distances[idx])
        theoretical_m = None
        error_m = None
        abs_error_m = None

        if true_distances is not None and mic_label in true_distances:
            theoretical_m = float(true_distances[mic_label])
            error_m = measured_m - theoretical_m
            abs_error_m = abs(error_m)
            abs_errors.append(abs_error_m)

        per_anchor_ranging.append(
            {
                "microphone_label": mic_label,
                "measured_distance_m": measured_m,
                "theoretical_distance_m": theoretical_m,
                "error_m": error_m,
                "abs_error_m": abs_error_m,
            }
        )

    if abs_errors:
        mean_abs_ranging_error = float(np.mean(abs_errors))
        p95_abs_ranging_error = float(np.percentile(abs_errors, 95))
    else:
        mean_abs_ranging_error = float(np.nan)
        p95_abs_ranging_error = float(np.nan)

    return {
        "experiment_id": str(experiment_id),
        "cycle_id": int(cycle_id),
        "path_id": int(path_id) if str(path_id).isdigit() else str(path_id),
        "position_available": bool(metrics["position_available"]),
        "ground_truth_position_xyz": None if metrics["true_position_xyz"] is None else metrics["true_position_xyz"].tolist(),
        "estimated_position_xyz": metrics["estimated_position_xyz"].tolist(),
        "mean_abs_ranging_error_m": mean_abs_ranging_error,
        "p95_abs_ranging_error_m": p95_abs_ranging_error,
        "per_anchor": per_anchor_ranging,
    }


def save_position_and_ranging_records(output_dir: Path, position_record: dict, ranging_record: dict) -> tuple[Path, int, Path, int]:
    """Append position and ranging records to their generic .npy files."""
    position_error_file = output_dir / "MB_position_errors.npy"
    ranging_error_file = output_dir / "MB_ranging_errors.npy"

    position_records_count = append_dict_record(position_error_file, position_record)
    ranging_records_count = append_dict_record(ranging_error_file, ranging_record)

    return position_error_file, position_records_count, ranging_error_file, ranging_records_count


def _load_dict_list_from_npy(file_path: Path) -> list[dict]:
	"""Load a .npy file that stores either one dict or a list of dict records."""
	if not file_path.exists():
		raise FileNotFoundError(f"File not found: {file_path}")

	raw = np.load(file_path, allow_pickle=True).tolist()
	if isinstance(raw, dict):
		return [raw]
	if isinstance(raw, list):
		return raw
	return list(raw)


def load_mb_logs(base_dir: str | Path | None = None) -> tuple[list[dict], list[dict]]:
	"""
	Load MB position and ranging records.

	Returns:
		position_records: list of dictionaries from MB_position_errors.npy
		ranging_records: list of dictionaries from MB_ranging_errors.npy
	"""
	if base_dir is None:
		base_dir = Path(__file__).parent
	else:
		base_dir = Path(base_dir)

	position_file = base_dir / "MB_position_errors.npy"
	ranging_file = base_dir / "MB_ranging_errors.npy"

	position_records = _load_dict_list_from_npy(position_file)
	ranging_records = _load_dict_list_from_npy(ranging_file)
	return position_records, ranging_records


def prepare_2d_position_data(position_records: list[dict], only_with_gt: bool = True) -> dict[str, np.ndarray]:
    """
    Convert MB position records into numpy arrays convenient for plotting/CDF.

    Returns keys:
        estimated_xy: (N, 2)
        ground_truth_xy: (N, 2)
        error_2d_m: (N,)
        path_id: (N,)
        experiment_cycle: (N,) string labels EXPxxx_pathX_cycley
    """
    estimated_xy = []
    ground_truth_xy = []
    error_2d_m = []
    path_id = []
    experiment_cycle = []

    for rec in position_records:
        gt_xy = rec.get("ground_truth_position_xy")
        est_xy = rec.get("estimated_position_xy")
        err_2d = rec.get("position_error_2d_m")
        pos_avail = bool(rec.get("position_available", False))

        if est_xy is None:
            continue
        if only_with_gt and (not pos_avail or gt_xy is None or err_2d is None or np.isnan(err_2d)):
            continue

        estimated_xy.append(np.asarray(est_xy, dtype=float))
        if gt_xy is None:
            ground_truth_xy.append(np.array([np.nan, np.nan], dtype=float))
        else:
            ground_truth_xy.append(np.asarray(gt_xy, dtype=float))

        error_2d_m.append(float(err_2d) if err_2d is not None else np.nan)
        path_val = rec.get("path_id", "NA")
        path_id.append(path_val)
        experiment_cycle.append(f"{rec.get('experiment_id', 'NA')}_path{path_val}_cycle{rec.get('cycle_id', 'NA')}")

    if not estimated_xy:
        return {
            "estimated_xy": np.empty((0, 2), dtype=float),
            "ground_truth_xy": np.empty((0, 2), dtype=float),
            "error_2d_m": np.empty((0,), dtype=float),
            "path_id": np.empty((0,), dtype=object),
            "experiment_cycle": np.empty((0,), dtype=object),
        }

    return {
        "estimated_xy": np.vstack(estimated_xy),
        "ground_truth_xy": np.vstack(ground_truth_xy),
        "error_2d_m": np.asarray(error_2d_m, dtype=float),
        "path_id": np.asarray(path_id, dtype=object),
        "experiment_cycle": np.asarray(experiment_cycle, dtype=object),
    }


def prepare_ranging_error_data(ranging_records: list[dict]) -> dict[str, np.ndarray]:
    """Flatten ranging records into arrays for plotting/global statistics."""
    mean_abs = []
    p95_abs = []
    per_anchor_abs = []
    path_id = []
    experiment_cycle = []

    for rec in ranging_records:
        mean_abs.append(float(rec.get("mean_abs_ranging_error_m", np.nan)))
        p95_abs.append(float(rec.get("p95_abs_ranging_error_m", np.nan)))
        path_val = rec.get("path_id", "NA")
        path_id.append(path_val)
        experiment_cycle.append(f"{rec.get('experiment_id', 'NA')}_path{path_val}_cycle{rec.get('cycle_id', 'NA')}")

        for anchor in rec.get("per_anchor", []):
            abs_err = anchor.get("abs_error_m")
            per_anchor_abs.append(np.nan if abs_err is None else float(abs_err))

    return {
        "mean_abs_ranging_error_m": np.asarray(mean_abs, dtype=float),
        "p95_abs_ranging_error_m": np.asarray(p95_abs, dtype=float),
        "per_anchor_abs_error_m": np.asarray(per_anchor_abs, dtype=float),
        "path_id": np.asarray(path_id, dtype=object),
        "experiment_cycle": np.asarray(experiment_cycle, dtype=object),
    }


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF arrays (x_sorted, cdf_y) from a 1D array."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    x_sorted = np.sort(x)
    cdf_y = np.arange(1, x_sorted.size + 1, dtype=float) / float(x_sorted.size)
    return x_sorted, cdf_y


def _path_id_matches(record_path_id, selected_path_id) -> bool:
    """Return True when record path_id matches the selected PATH_ID (robust int/str compare)."""
    return str(record_path_id) == str(selected_path_id)


def filter_records_by_path_id(
    position_records: list[dict],
    ranging_records: list[dict],
    path_id: int | str,
) -> tuple[list[dict], list[dict]]:
    """Filter position and ranging record lists to a single PATH_ID."""
    pos_filtered = [rec for rec in position_records if _path_id_matches(rec.get("path_id", "NA"), path_id)]
    rng_filtered = [rec for rec in ranging_records if _path_id_matches(rec.get("path_id", "NA"), path_id)]
    return pos_filtered, rng_filtered


def prepare_3d_position_data(position_records: list[dict], only_with_gt: bool = True) -> dict[str, np.ndarray]:
    """
    Convert MB position records into numpy arrays for 3D error/CDF analysis.

    Returns keys:
        estimated_xyz: (N, 3)
        ground_truth_xyz: (N, 3)
        error_3d_m: (N,)
        path_id: (N,)
        experiment_cycle: (N,)
    """
    estimated_xyz = []
    ground_truth_xyz = []
    error_3d_m = []
    path_id = []
    experiment_cycle = []

    for rec in position_records:
        gt_xyz = rec.get("ground_truth_position_xyz")
        est_xyz = rec.get("estimated_position_xyz")
        err_3d = rec.get("position_error_m")
        pos_avail = bool(rec.get("position_available", False))

        if est_xyz is None:
            continue
        if only_with_gt and (not pos_avail or gt_xyz is None or err_3d is None or np.isnan(err_3d)):
            continue

        estimated_xyz.append(np.asarray(est_xyz, dtype=float))
        if gt_xyz is None:
            ground_truth_xyz.append(np.array([np.nan, np.nan, np.nan], dtype=float))
        else:
            ground_truth_xyz.append(np.asarray(gt_xyz, dtype=float))

        error_3d_m.append(float(err_3d) if err_3d is not None else np.nan)
        path_val = rec.get("path_id", "NA")
        path_id.append(path_val)
        experiment_cycle.append(f"{rec.get('experiment_id', 'NA')}_path{path_val}_cycle{rec.get('cycle_id', 'NA')}")

    if not estimated_xyz:
        return {
            "estimated_xyz": np.empty((0, 3), dtype=float),
            "ground_truth_xyz": np.empty((0, 3), dtype=float),
            "error_3d_m": np.empty((0,), dtype=float),
            "path_id": np.empty((0,), dtype=object),
            "experiment_cycle": np.empty((0,), dtype=object),
        }

    return {
        "estimated_xyz": np.vstack(estimated_xyz),
        "ground_truth_xyz": np.vstack(ground_truth_xyz),
        "error_3d_m": np.asarray(error_3d_m, dtype=float),
        "path_id": np.asarray(path_id, dtype=object),
        "experiment_cycle": np.asarray(experiment_cycle, dtype=object),
    }


def plot_position_error_cdfs(error_2d_m: np.ndarray, error_3d_m: np.ndarray, path_id: int | str) -> None:
    """Plot CDF curves for 2D and 3D positioning errors for one PATH_ID."""
    x2, y2 = empirical_cdf(error_2d_m)
    x3, y3 = empirical_cdf(error_3d_m)

    fig, ax = plt.subplots(figsize=(9, 5))
    if x2.size:
        ax.plot(x2, y2, linewidth=2, label="2D Position Error CDF")
    if x3.size:
        ax.plot(x3, y3, linewidth=2, linestyle="--", label="3D Position Error CDF")

    ax.set_title(f"Position Error CDFs - PATH_ID {path_id}")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    if x2.size or x3.size:
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ranging_error_cdf(per_anchor_abs_error_m: np.ndarray, path_id: int | str) -> None:
    """Plot one CDF curve for all absolute ranging errors combined for one PATH_ID."""
    x_rng, y_rng = empirical_cdf(per_anchor_abs_error_m)
    if x_rng.size == 0:
        print(f"No ranging-error samples available for PATH_ID {path_id}.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_rng, y_rng, linewidth=2, label="Combined Ranging Error CDF")
    ax.set_title(f"Combined Ranging Error CDF - PATH_ID {path_id}")
    ax.set_xlabel("Absolute Ranging Error (m)")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_estimated_vs_gt_2d(estimated_xy: np.ndarray, ground_truth_xy: np.ndarray, path_id: int | str, room_size_xy: tuple[float, float] = (8.56, 4.0)) -> None:
    """Plot estimated and ground-truth 2D trajectories for one PATH_ID, with room boundary."""
    if estimated_xy.size == 0:
        print(f"No 2D position samples available for PATH_ID {path_id}.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    room_x, room_y = room_size_xy
    room_outline_x = [0.0, room_x, room_x, 0.0, 0.0]
    room_outline_y = [0.0, 0.0, room_y, room_y, 0.0]
    ax.plot(room_outline_x, room_outline_y, "k-", linewidth=2.0, label="Room boundary")

    ax.plot(estimated_xy[:, 0], estimated_xy[:, 1], "o-", linewidth=1.5, markersize=4, label="Estimated path")

    # Plot GT only where available (non-NaN rows)
    valid_gt = ~np.isnan(ground_truth_xy).any(axis=1)
    if np.any(valid_gt):
        gt_xy = ground_truth_xy[valid_gt]
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "s-", linewidth=1.5, markersize=4, label="Ground-truth path")

    ax.set_title(f"Estimated vs GT 2D Path - PATH_ID {path_id}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.set_xlim(-0.2, room_x + 0.2)
    ax.set_ylim(-0.2, room_y + 0.2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def get_position_and_ranging_record_by_index(position_records: list[dict], ranging_records: list[dict], index: int) -> tuple[dict, dict | None]:
    """Return one position record and its matching ranging record using experiment/path/cycle identity."""
    if index < 0 or index >= len(position_records):
        raise IndexError(f"index {index} is out of range for {len(position_records)} position records")

    pos_rec = position_records[index]
    exp = pos_rec.get("experiment_id")
    cyc = pos_rec.get("cycle_id")
    pid = pos_rec.get("path_id")

    for rng_rec in ranging_records:
        if (
            str(rng_rec.get("experiment_id")) == str(exp)
            and str(rng_rec.get("cycle_id")) == str(cyc)
            and str(rng_rec.get("path_id")) == str(pid)
        ):
            return pos_rec, rng_rec

    return pos_rec, None


def _add_room_wireframe(fig: go.Figure, room_dims_xyz: tuple[float, float, float]) -> None:
    """Add a simple rectangular-room wireframe to a Plotly 3D figure."""
    x_max, y_max, z_max = room_dims_xyz
    corners = np.array(
        [
            [0, 0, 0],
            [x_max, 0, 0],
            [x_max, y_max, 0],
            [0, y_max, 0],
            [0, 0, z_max],
            [x_max, 0, z_max],
            [x_max, y_max, z_max],
            [0, y_max, z_max],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i0, i1 in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[corners[i0, 0], corners[i1, 0]],
                y=[corners[i0, 1], corners[i1, 1]],
                z=[corners[i0, 2], corners[i1, 2]],
                mode="lines",
                line=dict(color="rgba(80,80,80,0.6)", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def plot_selected_position_3d_with_mics(position_record: dict, ranging_record: dict | None, microphone_positions: dict[str, np.ndarray], room_dims_xyz: tuple[float, float, float] = (8.56, 4.0, 2.4)) -> go.Figure:
    """Plot selected 3D estimate/GT plus used and not-used microphones in Plotly."""
    fig = go.Figure()
    _add_room_wireframe(fig, room_dims_xyz)

    used_labels: list[str] = []
    if ranging_record is not None:
        used_labels = [str(x.get("microphone_label")) for x in ranging_record.get("per_anchor", []) if x.get("microphone_label") is not None]
    used_set = {lbl.upper() for lbl in used_labels}

    configured_labels = [str(lbl).upper() for lbl in config.get("all_mics", [])]
    all_labels = [lbl for lbl in configured_labels if lbl in microphone_positions]
    used_xyz = []
    used_names = []
    not_used_xyz = []
    not_used_names = []
    for lbl in all_labels:
        pos = np.asarray(microphone_positions[lbl], dtype=float)
        if lbl.upper() in used_set:
            used_xyz.append(pos)
            used_names.append(lbl)
        else:
            not_used_xyz.append(pos)
            not_used_names.append(lbl)

    if not_used_xyz:
        not_used_xyz = np.vstack(not_used_xyz)
        fig.add_trace(
            go.Scatter3d(
                x=not_used_xyz[:, 0],
                y=not_used_xyz[:, 1],
                z=not_used_xyz[:, 2],
                mode="markers",
                name="Not used microphones",
                marker=dict(size=4, color="#B94E48", symbol="square"),
                text=[f"Mic {n}" for n in not_used_names],
            )
        )

    if used_xyz:
        used_xyz = np.vstack(used_xyz)
        fig.add_trace(
            go.Scatter3d(
                x=used_xyz[:, 0],
                y=used_xyz[:, 1],
                z=used_xyz[:, 2],
                mode="markers",
                name="Used microphones",
                marker=dict(size=5, color="#28965A", symbol="circle"),
                text=[f"Mic {n}" for n in used_names],
            )
        )

    est = position_record.get("estimated_position_xyz")
    if est is not None:
        est = np.asarray(est, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=[est[0]], y=[est[1]], z=[est[2]],
                mode="markers",
                name="Estimated position",
                marker=dict(size=8, color="#1F77B4", symbol="diamond"),
            )
        )

    gt = position_record.get("ground_truth_position_xyz")
    if gt is not None:
        gt = np.asarray(gt, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=[gt[0]], y=[gt[1]], z=[gt[2]],
                mode="markers",
                name="Ground-truth position",
                marker=dict(size=8, color="#F39C12", symbol="x"),
            )
        )

    path_id = position_record.get("path_id", "NA")
    cycle_id = position_record.get("cycle_id", "NA")
    exp_id = position_record.get("experiment_id", "NA")

    camera_params = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.6, y=-2.2, z=1.4),
    )

    fig.update_scenes(
        xaxis=dict(title_text="x [m]", range=[-0.2, room_dims_xyz[0] + 0.2]),
        yaxis=dict(title_text="y [m]", range=[-0.2, room_dims_xyz[1] + 0.2]),
        zaxis=dict(title_text="z [m]", range=[-0.2, room_dims_xyz[2] + 0.2]),
        aspectmode="data",
    )
    fig.update_layout(
        title=f"3D Positioning View - {exp_id} path {path_id} cycle {cycle_id}",
        scene_camera=camera_params,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
        width=1200,
        height=800,
    )

    figs_dir = Path(__file__).parent / "Figs/position_examples"
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_file = figs_dir / f"path_{path_id}_Cycle_{cycle_id}_EXP_{exp_id}.html"
    fig.write_html(str(output_file))
    return fig


def load_runs_to_process(
    path_ids_to_process: list[int],
    walks_pickle_path: str | Path | None = None,
    csi_dataset_path: str | Path | None = None,
) -> list[tuple[int, str, int]]:
    if not path_ids_to_process:
        raise ValueError("PATH_IDS_TO_PROCESS is empty. Provide at least one path index.")

    walks_pickle_path = Path(walks_pickle_path or (PROJECT_ROOT / "walks" / "train.pickle")).resolve()
    csi_ds, resolved_csi_dataset_path = open_csi_dataset("EXP008", csi_dataset_path)
    try:
        walks = load_pickle(str(walks_pickle_path), csi_ds)
    finally:
        csi_ds.close()

    invalid_path_ids = [path_id for path_id in path_ids_to_process if path_id < 0 or path_id >= len(walks)]
    if invalid_path_ids:
        raise IndexError(
            f"Requested path indices {invalid_path_ids} are out of range for {len(walks)} walks in {walks_pickle_path}"
        )

    runs_to_process: list[tuple[int, str, int]] = []
    for path_id in path_ids_to_process:
        walk = walks[path_id]
        static_stops = [
            (path_id, str(stop["experiment_id"]), int(stop["cycle_id"]))
            for stop in walk
            if bool(stop.get("resting", False))
        ]
        print(f"Path {path_id}: selected {len(static_stops)} resting positions")
        runs_to_process.extend(static_stops)

    print(f"Loaded walks from {walks_pickle_path}")
    print(f"CSI dataset used for walk formatting: {resolved_csi_dataset_path}")
    return runs_to_process