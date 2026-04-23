"""Example of AI multitask operation."""
import nidaqmx as ni
from nidaqmx.constants import AcquisitionType, TaskMode
import numpy as np
import matplotlib
from nidaqmx.constants import TaskMode
import pylab as plt
import json
import localFunctions as lf
from scipy.signal import find_peaks, peak_prominences
import subprocess
import os
matplotlib.use('TkAgg')

exe_path_sync = 'sync_exe_files/sync.exe'
exe_path_fireSWtrig = 'sync_exe_files/fire_SWtrigger.exe'
exe_path_cleanup = 'sync_exe_files/reset_sync.exe'
absolute_path_sync = os.path.join(os.getcwd(), exe_path_sync)
absolute_path_firetrig = os.path.join(os.getcwd(), exe_path_fireSWtrig)
absolute_path_cleanup = os.path.join(os.getcwd(), exe_path_cleanup)

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

def DAQ(TX_data):
    fs = config["sample_rate"]

    with ni.Task(new_task_name='out_slot2') as out1, ni.Task(new_task_name="in") as in1:

        #in1 = MASTER, in2 = SLAVE
        # Setup input and output channels
        out1.ao_channels.add_ao_voltage_chan("/PXI1Slot2/ao0")
        in1.ai_channels.add_ai_voltage_chan("/PXI1Slot2/ai0")
        in1.ai_channels.add_ai_voltage_chan("/PXI1Slot2/ai8")
        in1.ai_channels.add_ai_voltage_chan("/PXI1Slot5/ai0")
        in1.ai_channels.add_ai_voltage_chan("/PXI1Slot15/ai0")

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

def plot_signals(RX_data, TX_data):
    rx_data1 = RX_data[0]
    rx_data2 = np.array(RX_data[1])+0.2
    rx_data3 = np.array(RX_data[2])+0.3
    rx_data4 = np.array(RX_data[3])+0.4

    plt.plot(TX_data, label='Generated Signal')
    plt.plot(rx_data1, label='slot2Ai0', alpha=0.4)
    plt.plot(rx_data2, label='slot2Ai8', alpha=0.4)
    plt.plot(rx_data3, label='slot5Ai0', alpha=0.4)
    plt.plot(rx_data4, label='slot15Ai0', alpha=0.5)
    #plt.plot(RX_data[3], label='E09', alpha=0.5)
    # plt.plot(RX_data[4], label='A02', alpha=0.5)
    # plt.plot(RX_data[5], label='A11', alpha=0.5)
    # plt.plot(RX_data[6], label='G14', alpha=0.5)
    # plt.plot(RX_data[7], label='C06', alpha=0.5)
    plt.title('Sampled Values DAQ')
    plt.xlabel("Sample Index")
    plt.ylabel("Value [V]")
    plt.xlim(0, 15600)
    plt.ylim(-6, 6)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()

def create_TX_squarewave(n_zeros_symbol, n_ones_symbol, n_symbols, amplitude):
    zero = np.zeros(n_zeros_symbol)
    one = np.ones(n_ones_symbol)
    symbol = np.append(one, zero)
    signal_d = np.tile(symbol, n_symbols)
    writedata = (amplitude * signal_d)

    return writedata

# #############################################################################################
# ----------------------------------- Configurations ------------------------------------------
# #############################################################################################
with open('config.json') as json_file:
    config = json.load(json_file)

# Configuration file information
# "chirp_f_start": Start frequency of the chrip signal
# "chirp_f_stop": Stop frequency of the chirp signal
# "chirp_duration": Duration of the signal in s
# "chirp_DC": DC offset of the chirp
# "chirp_ampl": Amplitude of chirp signal in V
# "wake_up_duration": Duration of wake up for the microphone
# "id_daq": Id of the DAQ 'Dev1'
# "id_output": Output channel of the daq
# "id_input": Input channel of the daq
# "max_val_out": Max. output voltage of daq
# "min_val_out": Min. output voltage of daq
# "max_val_in": Max. output voltage of daq
# "min_val_in": Min. output voltage of daq
# "sample_rate": Sample rate to create the signal, for speaker and microphone
# "n_meas": Amount of chrips sended for 1 ranging estimation
# "peak_prominence_factor": peak prominence value max 1
# "temperature": temperature in celcius to determine speed op sound
# "plot_signals": Plot signals true or false
# "plot_spectrogram": Plot spectrogram true or false
# "plot_pulse_compr": Plot pulse compression true or false

# #############################################################################################
# ---------------------------------------- SETUP ---------------------------------------------
# #############################################################################################
fs = config["sample_rate"]
temp = config["temperature"]
v_sound = 20*np.sqrt(273+temp)
print('\nSpeed of sound: ', v_sound, ' m/s\n')

# Create chirp signal
chirp = lf.create_chirp(start_freq=config["chirp_f_start"], stop_freq=config["chirp_f_stop"],
                        chirp_duration=config["chirp_duration"], fs=fs,
                        amplitude=config["chirp_ampl"], offset=config["chirp_DC"])

# lf.plot_spectrogram('Spectogram TX Chirp', chirp, fs, config["chirp_duration"], 0)

# Calculate amount of samples within wake-up duration
n_wake_up_samples = config["wake_up_duration"] * fs
wake_up_at_sample = int(np.size(chirp) - n_wake_up_samples)  # The sample where the wake-up signal is in effect

if __name__ == "__main__":
    #read_system_list()
    #read_system_terminals()
    n_nodes = 4

    # Setup synchronization through .exe files
    result_sync = subprocess.run(absolute_path_sync, shell=True, stdout=subprocess.PIPE, text=True)
    print(result_sync.stdout)

    RX_data = DAQ(chirp)

    # clean-up synchronization through .exe files
    result_cleanup = subprocess.run(absolute_path_cleanup, shell=True, stdout=subprocess.PIPE, text=True)
    print(result_cleanup.stdout)

    plot_signals(RX_data, chirp)

    received_data = np.array(RX_data)

    # ---------------------
    #   Data Processing
    # ---------------------
    # Select the wake-up piece of the audio fragment
    rx_audio_wake = received_data[:, wake_up_at_sample:int(wake_up_at_sample + n_wake_up_samples)]

    # DC filter (no DC offset anymore)
    rx_audio_wake_AC = lf.DC_filter(rx_audio_wake)

    # Calculate needed amplification factor to have similar values to original signal amplitude
    amp = lf.rms(chirp) / lf.rms(rx_audio_wake_AC, axis=1)

    # multiplication with ampl
    rx_audio_amp = rx_audio_wake_AC * amp

    size_arrays = np.size(np.correlate(chirp, rx_audio_amp[0, :], "full"))
    pulse_compr_all = np.empty(size_arrays)
    LPF_all = np.empty(size_arrays)
    corr_index_array = np.array([])
    estimation_data_all = np.array([])
    ranging_faults = np.empty(n_nodes)

    for rx in range(n_nodes):

        # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
        corr_val = np.abs(np.correlate(chirp, rx_audio_amp[rx, :], "full"))

        # Normalize y peak values to have smaller values (not 1e9)
        corr_val = corr_val / np.max(corr_val)
        pulse_compr_all = np.vstack((pulse_compr_all, corr_val))

        # Add LPF to determine envelope
        LPF = lf.LPF(corr_val, 'lowpass', 1000, 10000, fs)  # 70, fs_mic/35   #1000, 5000
        LPF_all = np.vstack((LPF_all, LPF))

        # find all peaks and calculate promineces
        peaks, _ = find_peaks(LPF)
        prominences = peak_prominences(LPF, peaks)[0]
        most_prom = prominences[prominences > config["peak_prominence_factor"]][-1]
        most_prom_idx = np.where(np.around(prominences, decimals=5) == np.around(most_prom,
                                                                                 decimals=5))  # Select first index from row which > PP Threshold [0][0]
        idx_peak_samples = peaks[most_prom_idx]

        print('PPF: ', most_prom)
        print('Sample index with selected peak: ', idx_peak_samples)

        # calculate height of each peak's contour line
        contour_heights = LPF[peaks] - prominences

        index_opt_general = lf.idx_peak_determination_PP(corr_val, idx_peak_samples)

        # Save good index in array
        corr_index_array = np.append(corr_index_array, index_opt_general)

    pulse_compr_all = np.delete(pulse_compr_all, 0, 0)
    LPF_all = np.delete(LPF_all, 0, 0)

    # sample in effective chirp corresponding with start of chirp selection = (corr_index_max+1)-size(chirp_sigment)
    eff_start_samp_chirp = (((corr_index_array + 1) - n_wake_up_samples)[np.newaxis]).T

    # Calculate difference in amount of samples between synchronisation point(start wake-up) and part of received chirp (start-point)
    delta_sample = wake_up_at_sample - eff_start_samp_chirp

    # Determine distance
    distances_meas = (delta_sample / fs) * v_sound
    print('\nMeasured distances between mic and speaker (in m) \n', distances_meas)
    #TODO measure 10 time for each range ?