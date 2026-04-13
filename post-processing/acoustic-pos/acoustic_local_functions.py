from pathlib import Path
import importlib.util
import sys
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
        print(f"\nSelected {len(selected_anchors_dict)} anchors with strongest signals ({metric_name}):")

    print(f"Loaded positions for {len(selected_mic_positions)} microphones")
    # Loop through all selected microphones
    for mic_label, position in selected_mic_positions.items():
        print(f"{mic_label}: {position}")

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


def plot_pulsecomp_and_lpf(
    pulse_compr_all,
    LPF_all,
    corr_index_array,
    fs,
    selected_anchors_dict,
    TX,
    true_distances=None,
    plot_full_pulse_compression=True,
):
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