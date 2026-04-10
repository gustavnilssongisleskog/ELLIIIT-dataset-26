import numpy as np
import localFunctions as lf
from scipy.signal import *
import librosa as lbr
import json
from tqdm import tqdm

with open('config.json') as json_file:
    config = json.load(json_file)

save_loc = config['save_loc']

anchor_selection_tech = config['anchor_selection_tech']

if anchor_selection_tech=="All":
    print('Select All anchors')
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\All\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\All\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'

elif anchor_selection_tech== "RMS":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on biggest RMS values')
    extra_save_path_test_set = save_loc+'anchor_selection_results\\RMS\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'
    extra_save_path_ML = save_loc+'anchor_selection_results\\RMS\\'+str(config['number_used_anchors_pos'][0])+'anchors\\traindev\\'

elif anchor_selection_tech=="Max_CorrPeak_height":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on max correlation peak height')
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\Max_CorrPeak_height\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\Max_CorrPeak_height\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'

elif anchor_selection_tech== "Picked_CorrPeak_height":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on picked correlation peak height')
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\Picked_CorrPeak_height\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\Picked_CorrPeak_height\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'

elif anchor_selection_tech== "PP_value":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based PP values')
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\PP_value\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\PP_value\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'

elif anchor_selection_tech== "N_sig_peaks":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on amount of significant peaks')
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\N_sig_peaks\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\N_sig_peaks\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'

elif anchor_selection_tech== "RMS_diff_dist":
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\RMS_diff_dist\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\RMS_diff_dist\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'

elif anchor_selection_tech== "Sincsplit":
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\Sincsplit\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\Sincsplit\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'

elif anchor_selection_tech== "Sincsplit_dist":
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\Sincsplit_dist\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\Sincsplit_dist\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'

elif anchor_selection_tech== "ML":
    extra_save_path_test_set = save_loc + 'anchor_selection_results\\ML\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
    extra_save_path_ML = save_loc + 'anchor_selection_results_CNN\\ML\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'

else:
    print('WRONG ANCHOR SELECTION METHOD')

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
path_wakeuppart = save_loc+'RX_audio_onlywakeup_and_noise\\'
path_notnorm_corr = save_loc+'not_normalized_correlations\\'
path_PP_data = save_loc+'PP\\'

n_speakers_one_sim = config['n_speakers_simultaneous_in_simulation']
AGC = config['AGC']
SNR = config['addSNR']
show_pulse_comp = config['plot_pulse_compr']
peak_prominence = config['peak_prominence']
peak_prominence_factor = config['peak_prominence_factor']   # prominence threshold to later use the index of the first peak in the arry of the
                                                            # promineces larger than the threshold
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
# # Calculate amount of test set grid points (for non-shoebox not n_x_test * n_y_test * n_z_test)
# n_test_set_positions = (config['n_x_test'] * config['n_y_test'] * config['n_z_test'])-out_counted
n_test_set_positions = config['n_test_set_positions']
mn_loc_test_set = mn_loc_all[:n_test_set_positions, :]
print('Amount of test set mobile node positions: {}'.format(n_test_set_positions))

# If need also train dev set
mn_loc_traindev_set = mn_loc_all[n_test_set_positions:, :]
n_traindev_set_positions = np.size(mn_loc_traindev_set, axis=0)
print('Amount of train and dev set mobile node positions: {}'.format(n_traindev_set_positions))

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

# Get ideal chirp information
chirp_orig_resampl, fs_mic = lf.get_transmitted_resampled_chirp('Sim_data\\chirp.wav')
n_wake_up_samples = int(config['wake_up_duration'] * fs_mic)

###################################################
# ------------- Anchor Selection -----------------
###################################################
n_anch_positions = config['number_used_anchors_pos'][0]

"""
if speakers are anchors: return anchor selection matrix as matrix with n_sim indices
if microphones are anchors: return anchor selection matrix as matrix with rx indices
"""
info_matrix_3D_first_start = True
for mn in tqdm(range(0, n_traindev_set_positions)):
    new_mn = True
    for anchor in range(0, n_anchor_nodes):
        sp = 0
        if config['anchors'] == 'speakers':
            sim_nr = anchor
            rx = mn+n_test_set_positions
        else:
            sim_nr = mn+n_test_set_positions
            rx = anchor

        if anchor_selection_tech== 'All':
            if config['anchors'] == 'speakers':
                anch_sel_list_per_position = np.arange(0, n_speakers)
                anchor_selection_matrix = np.repeat([anch_sel_list_per_position], n_mics, axis=0)
            else:
                anch_sel_list_per_position = np.arange(0, n_mics)
                anchor_selection_matrix = np.repeat([anch_sel_list_per_position], n_speakers, axis=0)
            value_matrix = [[n_anch_positions]]
            info_matrix = [[n_anch_positions]]
            break

        elif anchor_selection_tech== 'RMS':
            # Get wake-up part
            wakeup_part = np.load(
                path_wakeuppart + 'rx_wakeup_mic' + str(rx) + 'speaker' + str(sp) + 'simulation' + str(
                    sim_nr) + '.npy')

            # Create information matrix
            if new_mn:
                single_mn_info_matrix = np.array([wakeup_part])
                new_mn = False
            else:
                single_mn_info_matrix = np.vstack((single_mn_info_matrix, wakeup_part))

            if single_mn_info_matrix.shape[0] == n_anchor_nodes:
                # Create RMS value matrix per position
                value_list = lf.rms(single_mn_info_matrix, axis=1).T[0]
                # Create anchor selection matrix
                anchor_selection_list = np.array([lf.top_indices_in_1d_array(value_list, n_anch_positions)])
                # Add in 3D
                if info_matrix_3D_first_start:
                    info_matrix = single_mn_info_matrix
                    value_matrix = value_list
                    anchor_selection_matrix = anchor_selection_list
                    info_matrix_3D_first_start = False
                else:
                    info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
                    # anchors x samples x MN --> select MN position 1: [:,:,1]
                    value_matrix = np.vstack((value_matrix, value_list))
                    anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))

        elif anchor_selection_tech=='Max_CorrPeak_height':
            # Get correlation values
            corr_graph = np.load(path_notnorm_corr + 'corr_val_mic' + str(rx) + 'speaker' + str(sp) + 'simulation' + str(
                    sim_nr) + '.npy')

            # Create information matrix
            if new_mn:
                single_mn_info_matrix = np.array([corr_graph])
                new_mn = False
            else:
                single_mn_info_matrix = np.vstack((single_mn_info_matrix, corr_graph))

            if single_mn_info_matrix.shape[0] == n_anchor_nodes:
                # Create Peak Height value matrix per position
                value_list = np.max(single_mn_info_matrix, axis=1).T #TODO

                # Create anchor selection matrix
                anchor_selection_list = np.array([lf.top_indices_in_1d_array(value_list, n_anch_positions)])
                # Add in 3D
                if info_matrix_3D_first_start:
                    info_matrix = single_mn_info_matrix
                    value_matrix = value_list
                    anchor_selection_matrix = anchor_selection_list
                    info_matrix_3D_first_start = False
                else:
                    info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
                    # anchors x samples x MN --> select MN position 1: [:,:,1]
                    value_matrix = np.vstack((value_matrix, value_list))
                    anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))
        elif anchor_selection_tech== "Picked_CorrPeak_height":
            print('Not yet possible to do, no saved PP data for training set')
            # picked_corr_idx_matrix = np.load(path_PP_data+'1PICKED_corr_peak_idx_matrix.npy')[mn,:]
            # # Get correlation values
            # corr_graph = np.load(
            #     path_notnorm_corr + 'corr_val_mic' + str(rx) + 'speaker' + str(sp) + 'simulation' + str(
            #         sim_nr) + '.npy')
            #
            # # Create information matrix
            # if new_mn:
            #     single_mn_info_matrix = np.array([corr_graph])
            #     new_mn = False
            # else:
            #     single_mn_info_matrix = np.vstack((single_mn_info_matrix, corr_graph))
            #
            # if single_mn_info_matrix.shape[0] == n_anchor_nodes:
            #     vert_select = np.arange(0, n_anchor_nodes)
            #     picked_corr_idx_matrix = np.array(picked_corr_idx_matrix, dtype=int) #TODO if higher sample rate int overflow?
            #     # Create Peak Height value matrix per position
            #     value_list = single_mn_info_matrix[vert_select, picked_corr_idx_matrix]
            #
            #     # Create anchor selection matrix
            #     anchor_selection_list = np.array([lf.top_indices_in_1d_array(value_list, n_anch_positions)])
            #     # Add in 3D
            #     if info_matrix_3D_first_start:
            #         info_matrix = single_mn_info_matrix
            #         value_matrix = value_list
            #         anchor_selection_matrix = anchor_selection_list
            #         info_matrix_3D_first_start = False
            #     else:
            #         info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
            #         # anchors x samples x MN --> select MN position 1: [:,:,1]
            #         value_matrix = np.vstack((value_matrix, value_list))
            #         anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))

        elif anchor_selection_tech== "N_sig_peaks":
            print('Not yet possible to do, no saved PP data for training set')
            # # Get list with peak idxs of sig peaks
            # peaks_idx_list = np.load(path_PP_data+'peak_idxs_mic'+str(rx) + 'speaker' + str(sp) + 'notAGCsimulation' + str(
            #         sim_nr) + '.npy')
            #
            # n_peaks = np.size(peaks_idx_list)
            #
            # if new_mn:
            #     single_mn_info_matrix = np.array([n_peaks])
            #     new_mn = False
            # else:
            #     single_mn_info_matrix = np.vstack((single_mn_info_matrix, n_peaks))
            #
            # if single_mn_info_matrix.shape[0] == n_anchor_nodes:
            #     # Create Peak Height value matrix per position
            #     value_list = single_mn_info_matrix.T[0]
            #
            #     # Create anchor selection matrix
            #     anchor_selection_list = np.array([lf.top_indices_in_1d_array(value_list, n_anch_positions)])
            #     # Add in 3D
            #     if info_matrix_3D_first_start:
            #         info_matrix = single_mn_info_matrix
            #         value_matrix = value_list
            #         anchor_selection_matrix = anchor_selection_list
            #         info_matrix_3D_first_start = False
            #     else:
            #         info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
            #         # anchors x samples x MN --> select MN position 1: [:,:,1]
            #         value_matrix = np.vstack((value_matrix, value_list))
            #         anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))

        elif anchor_selection_tech== "PP_value":
            print('Not yet possible to do, no saved PP data for training set')
            # pp_values = np.load(
            #     path_PP_data + 'peak_prominences_mic' + str(rx) + 'speaker' + str(sp) + 'notAGCsimulation' + str(
            #         sim_nr) + '.npy')
            #
            # pp_max = np.max(pp_values)
            #
            # if new_mn:
            #     single_mn_info_matrix = np.array([pp_max])
            #     new_mn = False
            # else:
            #     single_mn_info_matrix = np.vstack((single_mn_info_matrix, pp_max))
            #
            # if single_mn_info_matrix.shape[0] == n_anchor_nodes:
            #     # Create Peak Height value matrix per position
            #     value_list = single_mn_info_matrix.T[0]
            #
            #     # Create anchor selection matrix
            #     anchor_selection_list = np.array([lf.top_indices_in_1d_array(-value_list, n_anch_positions)])
            #     # Add in 3D
            #     if info_matrix_3D_first_start:
            #         info_matrix = single_mn_info_matrix
            #         value_matrix = value_list
            #         anchor_selection_matrix = anchor_selection_list
            #         info_matrix_3D_first_start = False
            #     else:
            #         info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
            #         # anchors x samples x MN --> select MN position 1: [:,:,1]
            #         value_matrix = np.vstack((value_matrix, value_list))
            #         anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))

        elif anchor_selection_tech== "RMS_diff_dist":
            print('Not yet possible to do, no ranging information for training set')
            # # Get wake-up part
            # wakeup_part = np.load(
            #     path_wakeuppart + 'rx_wakeup_mic' + str(rx) + 'speaker' + str(sp) + 'simulation' + str(
            #         sim_nr) + '.npy')
            #
            # # Create information matrix
            # if new_mn:
            #     single_mn_info_matrix = np.array([wakeup_part])
            #     new_mn = False
            # else:
            #     single_mn_info_matrix = np.vstack((single_mn_info_matrix, wakeup_part))
            #
            # if single_mn_info_matrix.shape[0] == n_anchor_nodes:
            #     # Create RMS value matrix per position
            #     value_list = lf.rms(single_mn_info_matrix, axis=1).T[0]
            #
            #     d_meas = d_meas_matrix[mn, :]
            #
            #     # Create anchor selection matrix
            #     anchor_rms_order_list = np.argsort(-value_list)
            #
            #     # Adjust also order of d_meas array
            #     d_meas_ordered_list = d_meas[anchor_rms_order_list]
            #
            #     # Select 6 with diff in ranging distance of at least 1m
            #     dist = 0.15
            #     anchor_selection_list = lf.select_anchors_rms_distance(anchor_rms_order_list, d_meas_ordered_list, n_anch_positions, dist)
            #
            #     # Add in 3D
            #     if info_matrix_3D_first_start:
            #         info_matrix = single_mn_info_matrix
            #         value_matrix = value_list
            #         anchor_selection_matrix = anchor_selection_list
            #         info_matrix_3D_first_start = False
            #     else:
            #         info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
            #         # anchors x samples x MN --> select MN position 1: [:,:,1]
            #         value_matrix = np.vstack((value_matrix, value_list))
            #         anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))
        elif anchor_selection_tech== "Sincsplit":
            print('Not yet possible to do, no PP information for training set')
            # LPF_not_ideal = np.load(
            #     path_LPF_curve + 'LPF_mic' + str(rx) + 'speaker' + str(sp) + 'notAGCsimulation' + str(sim_nr) + '.npy')
            #
            # # Map ideal on non ideal LPF
            # idx_peak_samples, eff_start_samp_chirp = lf.get_chirp_index_start_from_PP_corr(rx, sp, sim_nr,
            #                                                                                path_PP_data,
            #                                                                                peak_prominence_factor,
            #                                                                                n_wake_up_samples)
            # corr_val_ideal_mapped, LPF_ideal_mapped = lf.create_ideal_sinc_at_idx(chirp_orig_resampl, fs_mic,
            #                                                                       eff_start_samp_chirp)
            #
            # gamma = LPF_not_ideal - LPF_ideal_mapped
            #
            # # Create information matrix
            # if new_mn:
            #     single_mn_info_matrix = np.array([gamma])
            #     new_mn = False
            # else:
            #     single_mn_info_matrix = np.vstack((single_mn_info_matrix, gamma))
            #
            # if single_mn_info_matrix.shape[0] == n_anchor_nodes:
            #     # Create Peak Height value matrix per position
            #     gamma_abs_matrix = np.abs(single_mn_info_matrix)
            #     value_list = np.sum(gamma_abs_matrix, axis = 1)
            #
            #     # Create anchor selection matrix
            #     anchor_selection_list = np.array([lf.top_indices_in_1d_array(value_list, n_anch_positions)])
            #     # Add in 3D
            #     if info_matrix_3D_first_start:
            #         info_matrix = single_mn_info_matrix
            #         value_matrix = value_list
            #         anchor_selection_matrix = anchor_selection_list
            #         info_matrix_3D_first_start = False
            #     else:
            #         info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
            #         # anchors x samples x MN --> select MN position 1: [:,:,1]
            #         value_matrix = np.vstack((value_matrix, value_list))
            #         anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))

        elif anchor_selection_tech== "Sincsplit_dist":
            print('Not yet possible to do, no PP information for training set')
            # LPF_not_ideal = np.load(
            #     path_LPF_curve + 'LPF_mic' + str(rx) + 'speaker' + str(sp) + 'notAGCsimulation' + str(
            #         sim_nr) + '.npy')
            #
            # # Map ideal on non ideal LPF
            # idx_peak_samples, eff_start_samp_chirp = lf.get_chirp_index_start_from_PP_corr(rx, sp, sim_nr,
            #                                                                                path_PP_data,
            #                                                                                peak_prominence_factor,
            #                                                                                n_wake_up_samples)
            # corr_val_ideal_mapped, LPF_ideal_mapped = lf.create_ideal_sinc_at_idx(chirp_orig_resampl, fs_mic,
            #                                                                       eff_start_samp_chirp)
            #
            # gamma = LPF_not_ideal - LPF_ideal_mapped
            #
            # # Create information matrix
            # if new_mn:
            #     single_mn_info_matrix = np.array([gamma])
            #     new_mn = False
            # else:
            #     single_mn_info_matrix = np.vstack((single_mn_info_matrix, gamma))
            #
            # if single_mn_info_matrix.shape[0] == n_anchor_nodes:
            #     # Create Peak Height value matrix per position
            #     gamma_abs_matrix = np.abs(single_mn_info_matrix)
            #     value_list = np.sum(gamma_abs_matrix, axis=1)
            #
            #     d_meas = d_meas_matrix[mn, :]
            #
            #     # Create anchor selection matrix
            #     anchor_sinc_order_list = np.argsort(value_list)
            #
            #     # Adjust also order of d_meas array
            #     d_meas_ordered_list = d_meas[anchor_sinc_order_list]
            #
            #     # Select 6 with diff in ranging distance of at least 1m
            #     dist = 0.15
            #     anchor_selection_list = lf.select_anchors_rms_distance(anchor_sinc_order_list, d_meas_ordered_list,
            #                                                            n_anch_positions, dist)
            #
            #     # Add in 3D
            #     if info_matrix_3D_first_start:
            #         info_matrix = single_mn_info_matrix
            #         value_matrix = value_list
            #         anchor_selection_matrix = anchor_selection_list
            #         info_matrix_3D_first_start = False
            #     else:
            #         info_matrix = np.dstack((info_matrix, single_mn_info_matrix))
            #         # anchors x samples x MN --> select MN position 1: [:,:,1]
            #         value_matrix = np.vstack((value_matrix, value_list))
            #         anchor_selection_matrix = np.vstack((anchor_selection_matrix, anchor_selection_list))

        elif anchor_selection_tech== "ML":
            print('Dont because not scalable for all rooms etc.')
            print('todo train: input corr funtions, output list of 6 anchor numbers --> check positioning for all positions in a room and improve, lable data: best anchors to select (ext. search?) or positioning and use normal positioning method as cost function')
        else:
            print('WRONG ANCHOR SELECTION METHOD')

    else:
        continue
    break

# Save selection information
np.save(extra_save_path_ML+'info_matrix.npy', info_matrix)
np.save(extra_save_path_ML+'value_matrix.npy', value_matrix)
np.save(extra_save_path_ML+'anchor_selection_matrix.npy', anchor_selection_matrix)