import numpy as np
import localFunctions as lf
from scipy.signal import *
import librosa as lbr
import json
from tqdm import tqdm

with open('config.json') as json_file:
    config = json.load(json_file)

save_loc = config['save_loc']

path_audio = save_loc+'RX_audio\\'
path_rirs = save_loc+'RIRs\\'
path_audio_awgn = save_loc+'RX_audio_with_AWGN\\'
path_audio_sir = save_loc+'RX_audio_with_SIR_and_AWGN\\'
path_corr = save_loc+'correlation_functions\\'
path_corr_AGC = save_loc+'correlation_functions_with_AGC\\'
path_LPF_curve = save_loc+'LPF_curve\\'
path_dist_th = save_loc+'dist_th\\'
path_estimation_data = save_loc+'estimation_data\\'
path_output_data = save_loc+'Sim_output_data\\'
path_PP_data = save_loc+'PP\\'

n_speakers_one_sim = config['n_speakers_simultaneous_in_simulation']
AGC = config['AGC']
SNR = config['addSNR']
show_pulse_comp = config['plot_pulse_compr']
peak_prominence = config['peak_prominence']
peak_prominence_factor = config['peak_prominence_factor']   # prominence threshold to later use the index of the first peak in the arry of the
                                                            # promineces larger than the threshold
v_sound = lf.get_speed_of_sound(config['temperature'])
print("Speed of sound: " + str(v_sound) + " m/s")

# read anchor node positions
anchor_loc_all = np.load(save_loc+'Sim_data\\anchor_positions.npy')[:,0:3]
n_anchor_nodes = np.size(anchor_loc_all, axis=0)
print('\nAmount of anchor node positions: ', n_anchor_nodes)

# read mobile node positions
mn_loc_all = np.load(save_loc+'Sim_data\\positions_mobile_node.npy')[:,0:3]

# calculate amount of positions integrated
n_mobile_nodes_all = np.size(mn_loc_all, axis=0)
print('\nAmount of mobile node positions: ', n_mobile_nodes_all)

# Base only on test set to compare: filter out test set
out_counted = np.load(save_loc+'Sim_data\\outcounted.npy')
# Calculate amount of test set grid points (for non-shoebox not n_x_test * n_y_test * n_z_test)
#n_test_set_positions = (config['n_x_test'] * config['n_y_test'] * config['n_z_test'])-out_counted

n_test_set_positions = config['n_test_set_positions']
mn_loc_test_set = mn_loc_all[:n_test_set_positions, :]
print('Amount of test set mobile node positions: {}'.format(n_test_set_positions))

# If need also train dev set
# mn_loc_traindev_set = mn_loc_all[n_test_set_positions:, :]
# n_traindev_set_positions = np.size(mn_loc_traindev_set, axis=0)
# print('Amount of train and dev set mobile node positions: {}'.format(n_traindev_set_positions))

# Determine if speaker is anchor or mobile node
if config['anchors'] == 'speakers':
    print('Anchor nodes are speakers, mobile nodes are microphones')
    sp_locs = anchor_loc_all
    mic_locs = mn_loc_test_set
    n_speakers = n_anchor_nodes
    n_mics = n_test_set_positions
    # Selecting in speakers, thus sim nr
else:
    print('Anchor nodes are microphones, mobile nodes are speakers')
    sp_locs = mn_loc_test_set
    mic_locs = anchor_loc_all
    n_speakers = n_test_set_positions
    n_mics = n_anchor_nodes
    # Selecting in microphones, thus mics

wake_up_duration = config['wake_up_duration']    # Duration of the wake-up signal in s
chirp_orig_resampl = np.load(save_loc+'Sim_data\\chirp_orig_not_or_resampled.npy')

if SNR:
    fs_mic = np.load(path_audio_awgn + "fs.npy")

else:
    fs_mic = np.load(path_audio + "fs.npy")

# Calculate amount of samples within wake-up duration
n_wake_up_samples = wake_up_duration * fs_mic
wake_up_at_sample = int(np.size(chirp_orig_resampl) - n_wake_up_samples)  # The sample where the wake-up signal is in effect

# Read the pulse compression results
if AGC:
    agc_text='AGC'
else:
    agc_text='notAGC'

# calculate lengths for init
LPF_array_length = int(config["sample_rate_RX"]*(config["chirp_duration"]+config["wake_up_duration"])-1)

################################################
# ----------------- Ranging --------------------
################################################

# loop every postion anchor pair to determine ranging estimates
estimation_data_all = np.array([])
ranging_faults = np.empty(n_mics)
d_meas_matrix = np.empty(n_mics)
corr_peak_idx_matrix = np.empty(n_mics)
for sim_nr in tqdm(range(0, n_speakers)):
    if n_speakers == 1:   #TODO check if still needed somehow
        speaker_loc = sp_locs
    else:
        speaker_loc = sp_locs[sim_nr, :]

    pulse_compr_all = np.empty(LPF_array_length)
    LPF_all = np.empty(LPF_array_length)
    corr_index_array = np.array([])
    for rx in range(n_mics):
        for sp in range(n_speakers_one_sim):
            pulse_comp = np.load(path_corr + 'corr_val_mic' + str(rx) + 'speaker' + str(sp) + 'simulation'+ str(sim_nr) +'.npy')
            pulse_compr_all = np.vstack((pulse_compr_all, pulse_comp))

            LPF = np.load(path_LPF_curve + 'LPF_mic' + str(rx) + 'speaker' + str(sp) + str(agc_text)+'simulation'+ str(sim_nr) + '.npy')
            LPF_all = np.vstack((LPF_all, LPF))

            if peak_prominence:  #Always uses LPF for determination
                # Peak prominence to determine good peak ------------------------
                # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is
                # defined as the vertical distance between the peak and its lowest contour line

                # find all peaks and calculate prominences
                peaks, _ = find_peaks(LPF)
                prominences = peak_prominences(LPF, peaks)[0]
                most_prom = prominences[prominences > peak_prominence_factor][-1]
                most_prom_idx = np.where(np.around(prominences, decimals = 5) == np.around(most_prom, decimals=5))    # Select first index from row which > PP Threshold [0][0]
                idx_peak_samples = peaks[most_prom_idx][0]

                # calculate height of each peak's contour line
                contour_heights = LPF[peaks]-prominences
                index_opt_general = lf.idx_peak_determination_PP(pulse_comp, idx_peak_samples)

                # Save good index in array
                corr_index_array = np.append(corr_index_array, index_opt_general)

                # Save other PP values
                np.save(path_PP_data+'peak_idxs_mic' + str(rx) + 'speaker' + str(sp) + str(agc_text)+'simulation'+ str(sim_nr) + '.npy', peaks)
                np.save(path_PP_data+'peak_prominences_mic' + str(rx) + 'speaker' + str(sp) + str(agc_text)+'simulation'+ str(sim_nr) + '.npy', prominences)

                if show_pulse_comp:
                    # Plot correlation function
                    lf.plot_corr_LPF_peaks("Correlation at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, simulation " +str(sim_nr),
                                           pulse_comp, LPF, peaks, LPF[peaks], contour_heights, index_opt_general)
                # ---------------------------------------------------------------
            else:
                if config['LPF']:
                    # peak based on maximum value as easy distance estimation
                    # Determine index of the (one) max value on easy way
                    index_opt_general = lf.one_peak_determination_LPF(pulse_comp, LPF)
                else:
                    index_opt_general = lf.easy_peak_determination(pulse_comp)
                    # def func(x, a, b, c):  #OR WITH FITTING FUNCTION
                    #     return a * x ** 2 + b * x + c
                    #
                    # index_opt_general, y_peaks_index_fit, fit_func = lf.peak_determination(func, pulse_comp, int(fs_mic / 3000))

                # Save good index in array
                corr_index_array = np.append(corr_index_array, index_opt_general)

                if show_pulse_comp:
                    # Plot correlation function
                    lf.plot_corr_LPF("Correlation at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, simulation " + str(sim_nr),
                                           pulse_comp, LPF, index_opt_general)

    # sample in effective chirp corresponding with start of chirp selection = (corr_index_max+1)-size(chirp-segment)
    eff_start_samp_chirp = (((corr_index_array+1)-n_wake_up_samples)[np.newaxis]).T

    # Calculate difference in amount of samples between synchronisation point(start wake-up) and part of received chirp (start-point)
    delta_sample = wake_up_at_sample-eff_start_samp_chirp

    # Determine distance
    distances_meas = (delta_sample / fs_mic) * v_sound

    # Read theoretical distances
    distances_th = np.load(path_dist_th+'distances_th_simulation_'+str(sim_nr)+'.npy')

    # Calculate difference between theoretical and measured ranging distances
    np.set_printoptions(suppress=True)
    diff = (distances_th[0:n_mics,:] - distances_meas) # n_mics because: #MN's when becaon is speaker --> #mics, #anchors when beacon is mic --> #mics
    ranging_faults = np.vstack((ranging_faults, np.squeeze(diff, 1)))

    d_meas = np.squeeze(distances_meas.T, axis=0)
    d_meas_matrix = np.vstack((d_meas_matrix, d_meas))
    corr_peak_idx_matrix = np.vstack((corr_peak_idx_matrix, corr_index_array))


d_meas_matrix = np.delete(d_meas_matrix, 0, 0)
ranging_faults = np.delete(ranging_faults, 0, 0)
corr_peak_idx_matrix = np.delete(corr_peak_idx_matrix, 0, 0)

if config['anchors'] == "speakers":
    # Transpose: each row gives ranging distances to anchors for one position
    # (get always matrix n_mobile_nodes x n_anchors (amount of rows = amount of mobile nodes) --> make it universal
    d_meas_matrix = d_meas_matrix.T
    ranging_faults = ranging_faults.T
    corr_peak_idx_matrix = corr_peak_idx_matrix.T

# n_anch_positions = n_anchor_nodes
#
# anch_x_coords = anchor_loc_all[:n_anch_positions, 0]
# anch_y_coords = anchor_loc_all[:n_anch_positions, 1]
# anch_z_coords = anchor_loc_all[:n_anch_positions, 2]
#
# mn_x_coords = mn_loc_test_set[:, 0]
# mn_y_coords = mn_loc_test_set[:, 1]
# mn_z_coords = mn_loc_test_set[:, 2]

np.save(path_output_data+'rangingfault_all.npy', ranging_faults)
np.save(path_output_data+'d_meas_matrix.npy', d_meas_matrix)
np.save(path_PP_data+'1PICKED_corr_peak_idx_matrix.npy', corr_peak_idx_matrix)

