import numpy as np
from scipy.signal import chirp, spectrogram, firwin, filtfilt, resample, butter, lfilter
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration, VoltageUnits
from scipy.signal import find_peaks
import plotly.graph_objects as go
import nidaqmx as ni
from nidaqmx.constants import AcquisitionType, TaskMode
import json
import subprocess
import os
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit, least_squares
from itertools import chain
import dicts
from tqdm import tqdm
import samplerate

with open('config.json') as json_file:
    config = json.load(json_file)

####################################################
#---------- DAQ related functions -----------------#
####################################################
def DAQ_set_sync():
    """
    Synchronize the DAQ by routing the internal oscillator of card PXIe-6672 to the DAQ chassis
    :return: Sync routings
    """
    # Path to exe files
    exe_path_sync = 'sync_exe_files/sync.exe'
    absolute_path_sync = os.path.join(os.getcwd(), exe_path_sync)
    # Setup synchronization through .exe files
    result_sync = subprocess.run(absolute_path_sync, shell=True, stdout=subprocess.PIPE, text=True)
    print(result_sync.stdout)

def DAQ_reset_sync():
    """
    Reset the DAQ routing to cleanup
    :return: derouting information
    """
    # Path to exe files
    exe_path_cleanup = 'sync_exe_files/reset_sync.exe'
    absolute_path_cleanup = os.path.join(os.getcwd(), exe_path_cleanup)
    # clean-up synchronization through .exe files
    result_cleanup = subprocess.run(absolute_path_cleanup, shell=True, stdout=subprocess.PIPE, text=True)
    print(result_cleanup.stdout)


def read_system_list():
    #   Printing al physicals channels from al the differents units with proper name
    system = ni.system.System.local()
    # print(system)
    # print(system.local())
    # print (system.devices)
    # print('device')
    for device in system.devices:
        print('CO-channels')
        for channel in device.co_physical_chans:
            print(channel)
        print('AI-channels')
        for channel in device.ai_physical_chans:
            print(channel)
        print('AO-channels')
        for channel in device.ao_physical_chans:
            print(channel)

def read_system_terminals():
    system = ni.system.System.local()
    for device in system.devices:
        for tr in device.terminals:
            print(tr)

def DAQ(TX_data, anchor_names):
    fs = config["sample_rate"]

    with ni.Task(new_task_name='out_slot2') as out1, ni.Task(new_task_name="in") as in1:

        #in1 = MASTER, in2 = SLAVE
        # Setup input and output channels
        out1.ao_channels.add_ao_voltage_chan("/PXI1Slot2/ao2")

        connections_list = create_anchor_DAQ_input_list(anchor_names)
        for connection in connections_list:
            in1.ai_channels.add_ai_voltage_chan(connection)
        # in1.ai_channels.add_ai_voltage_chan("/PXI1Slot2/ai8:13")


        # Reference Clock Synchronization setup (get backplane external PXIe_Clk100,
        # optimized and shared from the Synchronization module)
        # Setup same reference clock and triggers for synchronization over PXI_Trig
        out1.timing.ref_clk_src = "PXIe_Clk100"
        out1.timing.ref_clk_rate = 100000000
        out1.timing.cfg_samp_clk_timing(rate=fs, samps_per_chan=np.size(TX_data)) # sample_mode=AcquisitionType.CONTINUOUS
        out1.triggers.sync_type.SLAVE = True

        in1.timing.ref_clk_src = "PXIe_Clk100"
        in1.timing.ref_clk_rate = 100000000
        in1.timing.cfg_samp_clk_timing(rate=fs, samps_per_chan=2*np.size(TX_data), sample_mode=AcquisitionType.FINITE)
        in1.triggers.sync_type.MASTER = True

        out1.control(TaskMode.TASK_COMMIT)
        in1.control(TaskMode.TASK_COMMIT)

        #out1.triggers.start_trigger.cfg_dig_edge_start_trig("/PXI1Slot2/ai/StartTrigger")
        out1.triggers.start_trigger.cfg_dig_edge_start_trig(in1.triggers.start_trigger.term)

        print("DAQ data")
        out1.write(TX_data, auto_start=False)

        #ORDER IS IMPORTANT, last one is MASTER!
        out1.start()
        in1.start()

        # First one is always the write
        RX_in1 = in1.read(number_of_samples_per_channel=2*np.size(TX_data))

        RX = np.delete(RX_in1, 0, axis=1)

        out1.stop()
        in1.stop()

        return RX

#######################################################
#----------- Selection I/O functions -----------------#
#######################################################

def flatten(A):
    """
    Flatten a list of lists with stings in it
    :param A: the main list to flatten
    :return: flattened list
    """
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt

def add_segment(anchor_list, tile_letter):
    '''
    Add a full segment to the anchor name list (e.g. A01-A14)
    :param anchor_name: list of selected tiles
    :param tile_letter: letter of the tile segment you want to add
    :return: list of the selected tiles and the added segment
    '''
    tile_nr = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
    segment = [tile_letter + tile for tile in tile_nr]
    anchor_list.append(segment)
    anchor_name = flatten(anchor_list)

    return anchor_name

def add_part_segment(anchor_list, section, direction):
    """
    Add a part of a segment e.g. A01-A04 is section A, direction E (East)
    :param anchor_list: list of anchors already selected
    :param section: section A-G
    :param direction: W (West), E (East), R (Roof)
    :return: added anchor locations to array
    """
    if direction == 'E':
        tile_nr = ['01', '02', '03', '04']
    elif direction == 'R':
        tile_nr = ['05', '06', '07', '08', '09', '10']
    elif direction == 'W':
        tile_nr = ['11', '12', '13', '14']
    else:
        print('ERROR: wrong selected direction')
        exit()

    segment = [section + tile for tile in tile_nr]
    anchor_list.append(segment)
    anchor_name = flatten(anchor_list)

    return anchor_name

def add_surface(anchor_list, direction):
    """
    Add a full surface with the given direction
    :param anchor_list: list of anchors already selected
    :param direction: W (West), E (East), R (Roof)
    :return: added anchor locations to array
    """
    if direction == 'E':
        tile_nr = ['01', '02', '03', '04']
    elif direction == 'R':
        tile_nr = ['05', '06', '07', '08', '09', '10']
    elif direction == 'W':
        tile_nr = ['11', '12', '13', '14']
    else:
        print('ERROR: wrong selected direction')
        exit()

    section_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    for section in section_list:
        segment = [section + tile for tile in tile_nr]
        anchor_list.append(segment)
    anchor_name = flatten(anchor_list)

    return anchor_name

def add_dam_pattern(anchor_list, type):
    """
    Create a dam_pattern as selection
    :param anchor_list: list of anchors already selected
    :param type: True or False, depending on which pattern
    :return: added anchor locations to array
    """
    if type:
        l1 = ['01', '03', '05', '07', '09', '11', '13']
        l2 = ['02', '04', '06', '08', '10', '12', '14']
    else:
        l1 = ['02', '04', '06', '08', '10', '12', '14']
        l2 = ['01', '03', '05', '07', '09', '11', '13']

    section_list_l1 = ['A', 'C', 'E', 'G']
    section_list_l2 = ['B', 'D', 'F']

    for section in section_list_l1:
        segment = [section + tile for tile in l1]
        anchor_list.append(segment)

    for section in section_list_l2:
        segment = [section + tile for tile in l2]
        anchor_list.append(segment)

    anchor_name = flatten(anchor_list)

    return anchor_name

def add_all(anchor_list):
    anchor_list = add_surface(anchor_list, 'E')
    anchor_list = add_surface(anchor_list, 'R')
    anchor_list = add_surface(anchor_list, 'W')

    return anchor_list

def clean_redundant(anchor_list):
    """
    Check if some anchors ar in the list multiple times and clean them out
    :param anchor_list: list of anchors
    :return: cleaned-out list with no redundant information

    """
    unique_list = []

    for item in anchor_list:
        if item not in unique_list:
            unique_list.append(item)

    return unique_list

def remove_tile(anchor_list, tile_list):
    """
    Remove the anchors listed in tile list
    :param anchor_list: original list of selected anchors
    :param tile_list: e.g. ['A01', 'A05'], anchors you want to remove from the list
    :return: new list without tile_list anchors
    """
    new_list = []

    for item in anchor_list:
        if item not in tile_list:
            new_list.append(item)

    return new_list

def create_anchor_position_matrix(anchor_name):
    """
    From the mic_dict get the corresponding positions given the name of the anchor
    :param anchor_name: list of selected anchor names
    :return: matrix with positions of anchors from list
    """
    dict = dicts.mic_dict

    positions = [dict[anchor][1] for anchor in anchor_name]
    anchor_xyz_np = np.array(positions)

    return anchor_xyz_np

def create_anchor_DAQ_input_list(anchor_name):
    """
    Create a list with the DAQ input names given a list with anchor names
    :param anchor_name: list with selected anchors
    :return: list with DAQ input names
    """
    dict = dicts.mic_dict

    connections = [dict[anchor][0] for anchor in anchor_name]

    return connections

#######################################################
#---------- Signal related functions -----------------#
#######################################################
def create_chirp(start_freq, stop_freq, chirp_duration, fs, amplitude, offset):
    """
    Creates the chirp signal as a numpy array
    :param start_freq: start frequency of the chirp
    :param stop_freq: stop frequency of the chirp
    :param chirp_duration: duration of the chirp in s
    :param fs: sample frequency
    :return: w: the chirp signal in np array
    """
    t = np.linspace(0., chirp_duration, int(fs * chirp_duration))
    w = (amplitude * chirp(t, f0=start_freq, f1=stop_freq, t1=chirp_duration, method='linear', phi=270)) + offset
    w[0] = 0
    w[len(w) - 1] = 0
    return w


def create_TX_squarewave(n_zeros_symbol, n_ones_symbol, n_symbols, amplitude):
    zero = np.zeros(n_zeros_symbol)
    one = np.ones(n_ones_symbol)
    symbol = np.append(one, zero)
    signal_d = np.tile(symbol, n_symbols)
    writedata = (amplitude * signal_d)

    return writedata

##########################################################
#---------- Signal processing functions -----------------#
##########################################################

def rms(x, axis=0):
    """
    Calculates the rms value of a signal
    :param x: the signal
    :return: the rms value
    """
    return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=True))


# Invoegen LDF
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

def resample_signals(fs_source, fs_mic, signal):
    """
    Downsample original signal if fs source =/= fs mic
    :param fs_source: sample frequency at speaker
    :param fs_mic: sample frequency at receiver
    :param signal: signal to resample
    :return: chirp_orig_resampl: resampled signal
    """
    if fs_source != fs_mic:
        fs_ratio = float(fs_mic) / float(fs_source)
        new_length = int(fs_ratio * signal.shape[0])
        re_sampled_orig_chirp_signal = np.zeros((1, new_length))

        # why sinc: http://www.mega-nerd.com/SRC/api_misc.html#ErrorReporting
        re_sampled_orig_chirp_signal = samplerate.resample(signal, fs_ratio, "sinc_best")
        chirp_orig_resampl = re_sampled_orig_chirp_signal

    else:  # in the sample rates are identical, no up/down sampling is required
        chirp_orig_resampl = signal

    #np.save('Sim_data\\chirp_orig_not_or_resampled.npy', chirp_orig_resampl)

    return chirp_orig_resampl

def get_filter_mask(received_data, threshold):
    """
    Filter the received data and create a mask to forward which index has to be removed (False), based
    on the RMS value < threshold
    :param received_data: received dataset
    :param threshold: value of threshold in VOLT (max 2V for mics)
    :return: array mask with True (keep) and False (remove)
    """
    #TODO, can add filter for ultrasound frequenties or decide on correlation values which ones to remove
    #Filter out DC
    data = DC_filter(received_data)

    #Get RMS
    rms_values = rms(data.T)[0]
    #print('AC RMS values of mics: \n', rms_values)
    # print(len(rms_values))

    #Create mask array if < or > threshold
    mask = rms_values > threshold
    #print('Mask: ', mask)

    return mask

def filter_anchors(anchor_name, filter_mask):
    """
    Based on a mask with true and false values, filter the anchor names list
    :param anchor_name: list of anchor names
    :param filter_mask: list with true and false as mask
    :return: used_anchors: anchors to keep with true from mask, removed_anchors: others
    """
    used_anchors = []
    removed_anchors = []
    i = 0
    for item in anchor_name:
        if filter_mask[i]:
            used_anchors.append(item)
        else:
            removed_anchors.append(item)
        i += 1

    return used_anchors, removed_anchors


def butter_highpass(highcut, fs, order=4):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='high')
    return b, a


def butter_highpass_filter(data, highcut, fs, order=4):
    b, a = butter_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def DC_filter(signal):
    """
    Signal - mean(signal)
    :param signal: signal to be filtered
    :return: signal without DC offset
    """
    filtered_signal = signal - np.mean(signal)
    return filtered_signal


def fade_in(signal, fadein_duration, fs):
    n_fade_samples = int(fs * fadein_duration)
    fade_array = np.linspace(0.05, 1, n_fade_samples)  # (1-np.linspace(1, 0.05, n_fade_samples))
    faded_signal = signal.copy()
    faded_signal[:n_fade_samples] *= fade_array
    return faded_signal


def fade_out(signal, fadeout_duration, fs):
    n_fade_samples = int(fs * fadeout_duration)
    fade_array = np.linspace(1, 0.05, n_fade_samples)  # (1-np.linspace(1, 0.05, n_fade_samples))
    faded_signal = signal.copy()
    faded_signal[-n_fade_samples:] *= fade_array
    return faded_signal


def scale_x_boundries(focuspoint, window, max_value):
    '''
    Put x-axis 0 as <= 0 others make x = focuspoint - window/2, if x > max_value make it max value, others again in window + scale to have big enough window always
    :param focuspoint: point of interest
    :param window: window to view the graph
    :return: min x_value
    '''
    xmin = focuspoint - (window / 2)
    xmax = focuspoint + (window / 2)

    if xmin < 0:
        xmin = 0
        xmax = window

    elif xmax > max_value:
        xmin = max_value - window
        xmax = max_value

    x_min_max = [xmin, xmax]

    return x_min_max

####################################################
# ---------- Positioning function -----------------#
####################################################
def get_wake_up_part(RX, TX):
    """
    Select the wake up part of the audio signal, assuming TX starts at t=0
    :param RX: received signal
    :param TX: transmitted signal
    :return: audio wake up signal, amount of wake up samples and wake up at sample
    """
    fs = config["downsample_freq"]

    # Calculate amount of samples within wake-up duration
    n_wake_up_samples = config["wake_up_duration"] * fs
    wake_up_at_sample = int(np.size(TX) - n_wake_up_samples)  # The sample where the wake-up signal is in effect

    # Select the wake-up piece of the audio fragment
    rx_audio_wake = RX[:, wake_up_at_sample:int(wake_up_at_sample + n_wake_up_samples)]
    return rx_audio_wake, n_wake_up_samples, wake_up_at_sample

def norm_correlate(TX, rx_matrix, n):
    # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
    corr_val = np.abs(np.correlate(TX, rx_matrix[n, :], "full"))

    # Normalize y peak values to have smaller values (not 1e9)
    corr_val = corr_val / np.max(corr_val)
    return corr_val

def norm_correlate_flipped(TX, rx_matrix, n):
    # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
    corr_val = np.abs(np.correlate(rx_matrix[n, :], TX, "full"))

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
def get_corr_with_LPF_curve(TX, rx_audio_amp):
    """
    Get the correlation LPF and index of selected correlation peak
    :param TX: TX chirp
    :param rx_audio_amp: received and filtered RX audio signal part
    :return: pulse compression values, LPF values and selected peak of de pulse compression
    """
    print("Pulse compression and LPF\n")
    n_nodes = np.size(rx_audio_amp, axis=0)  # amount of anchors
    fs = config["downsample_freq"]

    size_arrays = np.size(np.correlate(TX, rx_audio_amp[0, :], "full"))
    pulse_compr_all = np.empty(size_arrays)
    LPF_all = np.empty(size_arrays)
    corr_index_array = np.array([])


    for rx in tqdm(range(n_nodes)):
        # Pulse compression
        corr_val = norm_correlate(TX, rx_audio_amp, rx)
        pulse_compr_all = np.vstack((pulse_compr_all, corr_val))

        # Add LPF to determine envelope
        LPF_val = LPF(corr_val, 'lowpass', 10, config['downsample_freq']/26, fs)  # 70, fs_mic/35   #1000, 5000, #1000, 10000
        LPF_all = np.vstack((LPF_all, LPF_val))

        # Get the peak prominence index
        index_opt_general = get_peak_prom_index(LPF_val, corr_val)
        corr_index_array = np.append(corr_index_array, index_opt_general)

    pulse_compr_all = np.delete(pulse_compr_all, 0, 0)
    LPF_all = np.delete(LPF_all, 0, 0)

    return pulse_compr_all, LPF_all, corr_index_array


def get_corr_with_LPF_curve_not_Flipped(TX, rx_audio_amp):
    """
    Get the correlation LPF and index of selected correlation peak RX and TX switched to have normal correlation
    :param TX: TX chirp
    :param rx_audio_amp: received and filtered RX audio signal part
    :return: pulse compression values, LPF values and selected peak of de pulse compression
    """
    n_nodes = np.size(rx_audio_amp, axis=0)   # amount of anchors
    fs = config["downsample_freq"]

    size_arrays = np.size(np.correlate(rx_audio_amp[0, :], TX, "full"))
    pulse_compr_all = np.empty(size_arrays)
    LPF_all = np.empty(size_arrays)
    corr_index_array = np.array([])

    for rx in range(n_nodes):
        # Pulse compression
        corr_val = norm_correlate_flipped(TX, rx_audio_amp, rx)
        pulse_compr_all = np.vstack((pulse_compr_all, corr_val))

        # Add LPF to determine envelope
        LPF_val = LPF(corr_val, 'lowpass', 10, config['downsample_freq']/26, fs)  # 70, fs_mic/35   #1000, 5000
        LPF_all = np.vstack((LPF_all, LPF_val))

        # Get the peak prominence index
        index_opt_general = get_peak_prom_index(LPF_val, corr_val)
        corr_index_array = np.append(corr_index_array, index_opt_general)

    pulse_compr_all = np.delete(pulse_compr_all, 0, 0)
    LPF_all = np.delete(LPF_all, 0, 0)

    return pulse_compr_all, LPF_all, corr_index_array

def calc_ranges_ToA(RX, TX):
    """
    Determine the ranges between anchor and MN for ToA based systems
    :param RX: received signal
    :param TX: transmitted signal
    :return: measured distances, pulse compression values and LPF values
    """
    n_nodes = np.size(RX, axis=0)   # amount of anchors
    fs = config["downsample_freq"]

    rx_audio_wake, n_wake_up_samples, wake_up_at_sample = get_wake_up_part(RX, TX)

    # DC filter (no DC offset anymore)
    rx_audio_wake_AC = DC_filter(rx_audio_wake)

    # Calculate needed amplification factor to have similar values to original signal amplitude
    amp = rms(TX) / rms(rx_audio_wake_AC, axis=1)

    # multiplication with ampl
    rx_audio_amp = rx_audio_wake_AC * amp

    # get the correlation LPF and index of selected correlation peak
    pulse_compr_all, LPF_all, corr_index_array = get_corr_with_LPF_curve(TX, rx_audio_amp)

    # sample in effective chirp corresponding with start of chirp selection = (corr_index_max+1)-size(chirp_sigment)
    eff_start_samp_chirp = (((corr_index_array + 1) - n_wake_up_samples)[np.newaxis]).T

    # Calculate difference in amount of samples between synchronisation point(start wake-up) and part of received chirp (start-point)
    delta_sample = wake_up_at_sample - eff_start_samp_chirp

    v_sound = 20 * np.sqrt(273 + config["temperature"])
    print('\nSpeed of sound: ', v_sound, ' m/s\n')

    # Determine distance
    distances_meas = (delta_sample / fs) * v_sound
    return distances_meas, pulse_compr_all, LPF_all

def LS_positioning(anchor_positions, distances, x0):
    """
    Least Squares positioning estimate. Minimise the difference in measured distance to anchor point and estimated distances to do positioning
    :param anchor_positions: e.g. np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]])
    :param distances: e.g. np.array([1.5, 1.3, 1.2, 1.4, 0.7])
    :param x0: Initial guess for the position of the point e.g. np.array([0.5, 0.5, 0.5])
    :return: the estimated position
    """
    # Define the function to minimize
    def minimise_function_LS(x, anchor_positions, distances):
        # Calculate the squared differences between the estimated distances and the actual distances
        return np.sqrt(np.sum((np.linalg.norm(anchor_positions - x, axis=1) - distances) ** 2))
    # Call the least squares optimizer
    res = least_squares(minimise_function_LS, x0, args=(anchor_positions, distances))
    return res.x


def calculate_tdoa(samples, reference_sample, fs):
    return [abs(sample - reference_sample)/fs for sample in samples]


def multilateration_3d(anchor_positions, time_diffs, speed_of_sound, initial_estimate):
    num_anchors = len(anchor_positions)

    def error_function(position_estimate):
        residuals = [
            np.linalg.norm(position_estimate - anchor_positions[i]) - (time_diffs[i] * speed_of_sound)
            for i in range(num_anchors)
        ]
        return residuals

    result = least_squares(error_function, initial_estimate)
    estimated_position = result.x

    return estimated_position


def calc_3dpos_TDoA(anchor_xyz, RX, TX):
    """
    Determine the ranges between anchor and MN for ToA based systems
    :param RX: received signal
    :param TX: transmitted signal
    :return: measured distances, pulse compression values and LPF values
    """
    n_nodes = np.size(RX, axis=0)   # amount of anchors
    fs = config["downsample_freq"]

    # DC filter (no DC offset anymore)
    rx_audio_wake_AC = DC_filter(RX)

    # Calculate needed amplification factor to have similar values to original signal amplitude
    amp = rms(TX) / rms(rx_audio_wake_AC, axis=1)

    # multiplication with ampl
    rx_audio_amp = rx_audio_wake_AC * amp

    # get the correlation LPF and index of selected correlation peak
    pulse_compr_all, LPF_all, corr_index_array = get_corr_with_LPF_curve_not_Flipped(TX, rx_audio_amp)

    reference_sample = np.min(corr_index_array)

    # Calculate time differences for each anchor
    time_diffs = calculate_tdoa(corr_index_array, reference_sample, fs)

    # Initial estimate for the 3D position
    initial_estimate = np.array([4, 2, 1])

    v_sound = 20 * np.sqrt(273 + config["temperature"])
    print('\nSpeed of sound: ', v_sound, ' m/s\n')

    # Determine distance
    # distances_meas = (delta_sample / fs) * v_sound
    position = multilateration_3d(anchor_xyz, time_diffs, v_sound, initial_estimate)

    return position, pulse_compr_all, LPF_all, corr_index_array

####################################################
# ------------- Plotting functions ----------------#
####################################################
def plot_signals(RX_data, TX_data, anchor_name):
    plt.plot(TX_data, label='Generated Signal')

    for i in range(0, len(anchor_name)):
        plt.plot(RX_data[i], label=anchor_name[i], alpha=0.4)

    plt.title('Sampled Values DAQ')
    plt.xlabel("Sample Index")
    plt.ylabel("Value [V]")
    plt.xlim(0, 15600)
    plt.ylim(-6, 6)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()


def plot_audio_signal(title, signal, fs, dur_orig_sig, delay):
    """
    Show the audio signal
    :param title: title of the plot
    :param signal: audio signal
    :param fs: sampling frequency
    :param dur_orig_sig: duration of the original transmitted signal
    :param delay: the delay of the simulation
    """
    # audio: signal, fs: sample freq, m: mic nr, s: speaker nr, delay: delay of simulation
    x_signal = np.arange(0, np.size(signal), 1) / fs
    plt.plot(x_signal, signal)
    plt.title(title)
    plt.ylabel('Sound level')
    plt.xlabel('Time [s]')
    plt.xlim([delay, 1.5 * dur_orig_sig])
    plt.grid()
    plt.show()


def plot_spectrogram(title, signal, fs, dur_orig_sig, delay):
    """
    Show spectogram of the audio signal
    :param title: title of the plot
    :param signal: audio signal
    :param fs: sampling frequency
    :param dur_orig_sig: duration of the original transmitted signal
    :param delay: the delay of the simulation
    """
    ff, tt, Sxx = spectrogram(signal, fs=fs, nperseg=256, nfft=576)
    # c=plt.pcolormesh(tt, ff[:145], Sxx[:145], cmap='Dark2_r', shading='auto')
    c = plt.pcolormesh(tt, ff[:145], Sxx[:145], shading='gouraud')
    plt.title(title)
    # plt.colorbar(c)
    plt.xlabel('t (sec)')
    plt.ylabel('Frequency (Hz)')
    # plt.xlim([delay, 1.5*dur_orig_sig])
    # plt.ylim([22000, 47000])
    plt.grid()
    plt.show()


def plot_corr_LPF(title, corr, y_LPF, index_distance):
    """
    Show the pulse compression + LPF
    :param title: title of the plot
    :param corr: correlation values
    """
    plt.plot(corr, label='Correlation Values')
    plt.plot(y_LPF, label='LPF Values')
    plt.vlines(x=index_distance, ymin=0, ymax=1.1, colors='brown', alpha=0.4)
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.ylabel('Correlation')
    plt.xlabel('Samples')
    # plt.xlim(2000,2500)
    # plt.ylim(0.996,1.001)
    plt.show()


def plot_CDF_one(sortedData, title, filename):
    p = 1. * np.arange(len(sortedData)) / (len(sortedData) - 1)

    fig = go.Figure(data=[
        go.Scatter(x=sortedData, y=p, marker=dict(color='#264653'), name='Simple Intersection'),
    ])
    fig.update_layout(title=title, autosize=True, xaxis_title='m', yaxis_title='CDF')
    fig.update_xaxes(range=[0, 0.55])
    fig.write_html(filename + ".html")
    fig.write_image(filename + ".svg")

    fig.show()

def edges_from_vertices_2D(vertices):
    """
    Creates xy plane edges (2D projection or ground plan of the room)
    :param vertices: the coordinates of the room
    :return: the edges of the room
    """
    edges = []
    for i in range(len(vertices)):
        edges.append((vertices[i], vertices[(i+1) % len(vertices)]))
    return edges

colors = {
    'background': '#111111',
    'text': '#FFFFFF'
}

def plot_generated_anchors(vertices, height, anchor_locs, removed_anchor_locs, est_position, filename, title, anchor_name, removed_anchor_names):
    """
        Plots the room with anchors and mobile nodes (mn) given its vertices and height an all positions within the room
        Directivity vectors and directivities are also plotted if set to True
    :param vertices: coordinates of corners in 2D
    :param height: height of the room
    :param anchor_locs: location of the anchors np array: [[x1, y1, z1, azi1, coalt1 ], [x2, y2, ... ], ...]
    :param filename: name of the .html file to save it
    :param title: Title of the plotly plot
    """
    x_an = anchor_locs[:,0]
    y_an = anchor_locs[:,1]
    z_an = anchor_locs[:,2]

    if len(removed_anchor_locs)>0:
        x_an_removed = removed_anchor_locs[:,0]
        y_an_removed = removed_anchor_locs[:,1]
        z_an_removed = removed_anchor_locs[:,2]
    # azimuth_anch = anchor_locs[:,3]
    # coaltitude_anch = anchor_locs[:,4]

    est_position_x = est_position[0]
    est_position_y = est_position[1]
    est_position_z = est_position[2]

    traces_vert = []
    for corner in vertices:
        trace = go.Scatter3d(
            x=[corner[0], corner[0]], y=[corner[1], corner[1]], z=[0, height],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_vert.append(trace)

    # Create ground and top plane of room
    edges_top_ground = edges_from_vertices_2D(vertices)

    traces_top_bottom = []
    for edge in edges_top_ground:
        x_tr, y_tr = zip(*edge)
        trace1 = go.Scatter3d(
            x=x_tr, y=y_tr, z=[height, height],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_top_bottom.append(trace1)
        trace2 = go.Scatter3d(
            x=x_tr, y=y_tr, z=[0, 0],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_top_bottom.append(trace2)


    anchers = [(go.Scatter3d(x=x_an, y=y_an, z=z_an, mode='markers', name='Used anchors', marker_size=5,
                             marker=dict(color="#386055", symbol='square') , text=['Anchor '+str(anchor_name[i]) for i in range(len(x_an))]))]

    if len(removed_anchor_locs) > 0:
        removed_anchors = [(go.Scatter3d(x=x_an_removed, y=y_an_removed, z=z_an_removed, mode='markers', name='Ignored anchors', marker_size=5,
                             marker=dict(color="#B94E48", symbol='square') , text=['Anchor '+str(removed_anchor_names[i]) for i in range(len(x_an_removed))]))]


    est_pos = [(go.Scatter3d(x=[est_position_x], y=[est_position_y], z=[est_position_z], mode='markers', name='Estimated Position', marker_size=10,
                             marker=dict(color="#636efa", symbol='circle')))]

    if len(removed_anchor_locs) > 0:
        data = anchers + removed_anchors + traces_vert + traces_top_bottom + est_pos
    else:
        data = anchers + traces_vert + traces_top_bottom + est_pos

    # Create a Scatter trace for the points
    fig = go.Figure(data=data)

    camera_params = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1, y=-1.75, z=1.1)
    )

    fig.update_scenes(xaxis = dict( title_text='x [m]'),
                      yaxis = dict( title_text='y [m]'),
                      zaxis = dict( title_text='z [m]'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0, 0, 0, 0)', font_color=colors['text'],
                      scene_camera=camera_params, legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
                      width=1200, height=800)
    #fig.write_html(filename + ".html")
    #fig.show()
    return fig

def plot_signals_dash(RX_data, TX_data, fs, anchor_name):
    x_signal_TX = (np.arange(0, np.size(TX_data, axis=0), 1) / fs) * 1000
    TX_data_fig = go.Scatter(x=x_signal_TX, y=TX_data, mode='lines', name='TX signal')
    data = [TX_data_fig]

    if RX_data.ndim > 1:
        x_len = RX_data.shape[1]
        n_anchors = RX_data.shape[0]
    else:
        x_len = RX_data.shape[0]
        n_anchors = 1

    if len(RX_data) > 0:
        x_signal = (np.arange(0, x_len, 1) / fs)*1000

    if RX_data.ndim > 1:
        for RX_idx in range (0, n_anchors):   #y=DC_filter(RX_data[RX_idx,:])
            mic_fig = go.Scatter(x=x_signal, y=DC_filter(RX_data[RX_idx,:]), mode='lines', name=anchor_name[RX_idx])
            data.append(mic_fig)

    else:
        mic_fig = go.Scatter(x=x_signal, y=DC_filter(RX_data), mode='lines', name=anchor_name)
        data.append(mic_fig)

    fig = go.Figure(data=data)
    fig.update_layout(title='TX and RX signals', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0, 0, 0, 0)',
                      yaxis=dict(range=[-5, 5]),
                      xaxis_title='Time (ms)', yaxis_title='Received Signal (V)', font_color=colors['text'])

    return fig

def plot_pulsecomp_dash_ToA(pulse_compr_all, LPF_all, distances_meas, fs, anchor_name, TX):
    temp = config["temperature"]
    v_sound = 20 * np.sqrt(273 + temp)
    n_wake_up_samples = config["wake_up_duration"] * fs
    wake_up_at_sample = int(np.size(TX, axis=0) - n_wake_up_samples)
    if pulse_compr_all.ndim > 1:
        x_len = pulse_compr_all.shape[1]
        n_anchors = pulse_compr_all.shape[0]
    else:
        x_len = pulse_compr_all.shape[0]
        n_anchors = 1

    corr_idx_array = np.arange(0, x_len, 1)

    eff_starts_all_samples = ((corr_idx_array + 1) - n_wake_up_samples)  # to scale x-axis
    delta_distances_array = wake_up_at_sample - eff_starts_all_samples
    # delta_distances_array_inv = delta_distances_array-delta_distances_array[0]
    x_signal = (delta_distances_array / fs) * v_sound

    data = []

    if pulse_compr_all.ndim > 1:
        for RX_idx in range (0, np.size(pulse_compr_all, axis=0)):
            comp_val = go.Scatter(x=x_signal, y=pulse_compr_all[RX_idx,:], mode='lines', name=anchor_name[RX_idx])
            data.append(comp_val)
            LPF_val = go.Scatter(x=x_signal, y=LPF_all[RX_idx,:], mode='lines', name='LPF '+anchor_name[RX_idx])
            data.append(LPF_val)

        #fig = go.Figure(data=data)
        #y = [1, 1.1, 1.2, 1.3, 0.9, 1.4, 0.8]
        #i = 0
        #for x in distances_meas:
        #    fig.add_vline(x=x[0], line_color='red', line_width=2,
        #                  name='Selected peak: ' + str(np.round(x[0], 3)) + ' m')
        #    fig.add_annotation(x=x[0], y=y[i % 7], text=str(anchor_name[i]) + ':\n' + str(np.round(x[0], 3)) + ' m',
        #                       showarrow=True, ax=0, ay=-10)
        #    i += 1
    else:
        comp_val = go.Scatter(x=x_signal, y=pulse_compr_all, mode='lines', name=anchor_name)
        data.append(comp_val)
        LPF_val = go.Scatter(x=x_signal, y=LPF_all, mode='lines', name='LPF ' + anchor_name)
        data.append(LPF_val)
        fig = go.Figure(data=data)
        fig.add_vline(x=distances_meas[0], line_color='red', line_width=2,
                      name='Selected peak: ' + str(np.round(distances_meas, 3)) + ' m')
        fig.add_annotation(x=distances_meas[0], y=1.1, text=str(anchor_name) + ':\n' + str(np.round(distances_meas, 3)) + ' m',
                           showarrow=True, ax=0, ay=-10)

    fig.update_layout(title='Pulse Compression', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0, 0, 0, 0)',
                      yaxis=dict(range=[0, 1.2]), xaxis=dict(range=[0, 10]), xaxis_title='Ranging Distance (m)', yaxis_title='Norm. Correlation Value', font_color=colors['text'])

    return fig

def plot_pulsecomp_dash_TDoA(pulse_compr_all, LPF_all, distances_meas, anchor_name):
    x_signal = np.arange(0, np.size(pulse_compr_all, axis=1), 1) # correlation index values

    data = []
    for RX_idx in range (0, np.size(pulse_compr_all, axis=0)):
        comp_val = go.Scatter(x=x_signal, y=pulse_compr_all[RX_idx,:], mode='lines', name=anchor_name[RX_idx])
        data.append(comp_val)
        LPF_val = go.Scatter(x=x_signal, y=LPF_all[RX_idx,:], mode='lines', name='LPF '+anchor_name[RX_idx])
        data.append(LPF_val)

    fig = go.Figure(data=data)
    y = [1, 1.1, 1.2, 1.3]
    i = 0
    for x in distances_meas:
        fig.add_vline(x=x, line_color='red', line_width=2, name='Selected peak: '+str(np.round(x,3))+' m')
        fig.add_annotation(x= x, y = y[i%4], text=str(anchor_name[i])+':\n'+str(np.round(x,3)), showarrow=True, ax=0, ay=-10)
        i += 1

    fig.update_layout(title='Pulse Compression', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0, 0, 0, 0)',
                      yaxis=dict(range=[0, 1.4]), xaxis=dict(range=[8000, 18000]), xaxis_title='Correlation Index', yaxis_title='Norm. Correlation Value', font_color=colors['text'])

    return fig