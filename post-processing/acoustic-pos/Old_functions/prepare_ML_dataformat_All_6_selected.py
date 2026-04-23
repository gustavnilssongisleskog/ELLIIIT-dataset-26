import numpy as np
import matplotlib.pyplot as plt
import localFunctions as lf
from scipy.signal import *
import json
import os
from tqdm import tqdm
# Prepare all ML data formats: for 6 selected, one hot encoding for anchors
# Audio: ordered based on RMS, ordered based on anchor nr
# LPF: idem
# Corration values: idem
# RIRs: idem
#TODO do the same for all anchors

# ------------------------------------------------
#           Configurations
# ------------------------------------------------
with open('config.json') as json_file:
    config = json.load(json_file)

save_loc = config['save_loc']
save_loc_ml_results = save_loc+'anchor_selection_results_CNN\\'

anchor_selection_tech = config['anchor_selection_tech']
data_already_available = False

print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on biggest RMS values')
extra_save_path_test_set = save_loc + 'anchor_selection_results\\RMS\\' + str(
    config['number_used_anchors_pos'][0]) + 'anchors\\'
extra_save_path_ML = extra_save_path_test_set + 'traindev\\'

save_path_ML_data_sim = save_loc+'ML_Data\\Techtile_sim_data\\' # save path for ML data

path_audio = save_loc+'RX_audio\\'
#path_rirs = save_loc+'RIRs\\'
#path_audio_awgn = save_loc+'RX_audio_with_AWGN\\'
#path_corr = save_loc+'correlation_functions\\'
path_LPF_curve = save_loc+'LPF_curve\\'
#path_wakeuppart = save_loc+'RX_audio_onlywakeup_and_noise\\'
#path_notnorm_corr = save_loc+'not_normalized_correlations\\'
path_LPF_curve_per_position_downsampled = save_loc+'LPF_curves_per_simulation_downsampled\\'

# ------------------------------------------------
#           Read information
# ------------------------------------------------
AGC = config['AGC']
SNR = config['addSNR']

n_anchors = config['number_used_anchors_pos'][0]  # amount of anchors used (first element in list of config file!!)
n_speakers_simultaneous = config['n_speakers_simultaneous_in_simulation']    #amount of speakers DURING ONE position simulation

if config['AGC']:
    agc_text='AGC'
else:
    agc_text='notAGC'

# Read mn positions / labels
lables = np.load(save_loc+'Sim_data\\positions_mobile_node.npy')[:, 0:3]

# calculate amount of positions integrated
n_sets = np.size(lables, axis=0)   #amount of measured positions = amount of training examples

print('\nTotal number of mobile node positions: ', n_sets)
print('\nLabels/positions: \n', lables)
print('\nLable size: ', lables.shape)

n_onehots = 20

if not data_already_available:  # If you only want to know the size of the dataset --> true
    # Read datafile with LPF data for each position from all mics
    # 3D matrix: height amount of mics per 1 speaker position, width: amount of samples after LPF, depth: amount of positions

    # Load anchor information
    print('\nLoad Anchor information\n')
    anchors = np.load(save_loc+'Sim_data\\anchor_positions.npy')
    n_anchors_total =np.size(anchors, axis=0)

    anchor_selection_matrix_testset = np.load(extra_save_path_test_set+'anchor_selection_matrix.npy')
    anchor_selection_matrix_traindevset = np.load(extra_save_path_ML+'anchor_selection_matrix.npy')
    anchor_selection_matrix = np.vstack((anchor_selection_matrix_testset, anchor_selection_matrix_traindevset))

    #RIR_cutoff = 90000
    # Load
    print("\nLoad input data:\n")
    print("Downsampled LPF curves")
    LPF_dataset_downsampled = np.empty([n_anchors, config['n_samples_NN']+n_onehots])
    # print("Received Audio Parts")
    # Audio_dataset = np.empty([n_anchors, int(config['sample_rate_RX']*config['wake_up_duration'])+n_onehots])
    # print("Correlation values")
    # Correlation_dataset = np.empty([n_anchors, 2975+n_onehots])
    # print("RIRs")
    # RIR_dataset = np.empty([n_anchors, RIR_cutoff+n_onehots])

    #create one-hot matrix in numpy
    one_hot_matrix = np.diag(np.full(n_onehots, 1))

    if config['anchors'] == "speakers":
        for position in tqdm(range(0, n_sets)):
            LPF_all_one_position_combined = []
            # audio_all_one_position_combined = []
            # correlation_all_one_position_combined = []
            # RIR_all_one_position_combined = []
            for simulation in range(0, n_anchors_total):
                # Get LPF values
                LPF_all = np.load(path_LPF_curve_per_position_downsampled+'LPF_dataset_downsampled_1position_'+ str(agc_text)+'_simulation_'+str(simulation)+'.npy')
                LPF1_speaker_1mic = np.append(one_hot_matrix[simulation, :], LPF_all[position, :])
                LPF_all_one_position_combined = np.append(LPF_all_one_position_combined, LPF1_speaker_1mic)

                # # Get received wake-up audio parts values
                # audio_1_speaker_1mic = np.load(path_wakeuppart+'rx_wakeup_mic'+str(position)+'speaker0simulation'+str(simulation)+'.npy')
                # audio_1_speaker_1mic_onehot = np.append(one_hot_matrix[simulation, :], audio_1_speaker_1mic)
                # audio_all_one_position_combined = np.append(audio_all_one_position_combined, audio_1_speaker_1mic_onehot)
                #
                # # Get correlation values
                # correlation_1_speaker_1mic = np.load(path_corr + 'corr_val_mic'+str(position)+'speaker0simulation'+str(simulation)+'.npy')
                # correaltion_1_speaker_1mic_onehot = np.append(one_hot_matrix[simulation, :], correlation_1_speaker_1mic)
                # correlation_all_one_position_combined = np.append(correlation_all_one_position_combined, correaltion_1_speaker_1mic_onehot)

                # # Get RIR values
                # RIR_1_speaker_1mic = np.load(path_rirs + 'rir_RX_' + str(position) + 'from_source_0sim_' + str(simulation) + '.npy')
                # # Since RIR don't have the same array length, make them the same
                # RIR_1_speaker_1mic_const_len = lf.adjust_array_length(RIR_1_speaker_1mic, RIR_cutoff)
                #
                # RIR_1_speaker_1mic_onehot = np.append(one_hot_matrix[simulation, :], RIR_1_speaker_1mic_const_len)
                # RIR_all_one_position_combined = np.append(RIR_all_one_position_combined, RIR_1_speaker_1mic_onehot)

            # # Reshape all data
            LPF_all_1tr_down = np.reshape(LPF_all_one_position_combined, (n_anchors_total, config['n_samples_NN']+n_onehots))
            # audio_all_1tr_down = np.reshape(audio_all_one_position_combined, (n_anchors_total, int(config['sample_rate_RX']*config['wake_up_duration']) + n_onehots))
            # corr_all_1tr_down = np.reshape(correlation_all_one_position_combined, (n_anchors_total, 2975 + n_onehots))
            #RIR_all_1tr_down = np.reshape(RIR_all_one_position_combined, (n_anchors_total, RIR_cutoff + n_onehots))

            # ORDER THE LIST (if needed)
            anchor_selection_matrix_sorted_1pos = np.sort(anchor_selection_matrix[position, :])

            # #Select correct anchors data (selected ones)
            LPF_selected_anchors_one_position = LPF_all_1tr_down[anchor_selection_matrix_sorted_1pos,:]
            # Audio_selected_anchors_one_position = audio_all_1tr_down[anchor_selection_matrix_sorted_1pos, :]
            # Correlation_selected_anchors_one_position = corr_all_1tr_down[anchor_selection_matrix_sorted_1pos, :]
            #RIR_selected_anchors_one_position = RIR_all_1tr_down[anchor_selection_matrix_sorted_1pos, :]
            #
            LPF_dataset_downsampled = np.dstack((LPF_dataset_downsampled, LPF_selected_anchors_one_position))
            # Audio_dataset = np.dstack((Audio_dataset, Audio_selected_anchors_one_position))
            # Correlation_dataset = np.dstack((Correlation_dataset, Correlation_selected_anchors_one_position))
            #RIR_dataset = np.dstack((RIR_dataset, RIR_selected_anchors_one_position))

    else:
        for position in tqdm(range(0, n_sets)):
            #TODO this is not adjusted code
            #TODO adjust to selected anchor, no not the case not working
            #LPF_dataset = np.dstack((LPF_dataset, LPF_all_1tr))
            LPF_all_1tr_down = np.load(path_LPF_curve_per_position_downsampled+'LPF_dataset_downsampled_1position_'+ str(agc_text)+'_simulation_'+str(position)+'.npy')
            LPF_dataset_downsampled = np.dstack((LPF_dataset_downsampled, LPF_all_1tr_down))

    # #LPF_dataset = np.delete(LPF_dataset, 0, 2)
    data_LPF = np.delete(LPF_dataset_downsampled, 0, 2)
    # data_audio = np.delete(Audio_dataset, 0, 2)
    # data_correlation = np.delete(Correlation_dataset, 0, 2)
    #data_RIR = np.delete(RIR_dataset, 0, 2)

    ##save data
    np.save(save_path_ML_data_sim+"ML_dataset_sorted_anchor_nr_LPF_onehot_simulation.npy", data_LPF)
    # np.save(save_path_ML_data + "ML_dataset_sorted_anchor_nr_Audio_onehot.npy", data_audio)
    # np.save(save_path_ML_data + "ML_dataset_sorted_anchor_nr_Correlation_onehot.npy", data_correlation)
    #np.save(save_path_ML_data + "ML_dataset_sorted_anchor_nr_RIR_onehot.npy", data_RIR)

else:
    data_LPF = np.load(save_path_ML_data_sim+"ML_dataset_sorted_anchor_nr_LPF_onehot_simulation.npy")
    # data_audio = np.load(save_path_ML_data + "ML_dataset_sorted_anchor_nr_Audio_onehot.npy")
    # data_correlation = np.load(save_path_ML_data + "ML_dataset_sorted_anchor_nr_Correlation_onehot.npy")
    #data_RIR = np.load(save_path_ML_data + "ML_dataset_sorted_anchor_nr_RIR_onehot.npy")

print('\nData LPF size: ', data_LPF.shape)
# print('\nData AUDIO size: ', data_audio.shape)
# print('\nData CORRELATION size: ', data_correlation.shape)
#print('\nData RIR size: ', data_RIR.shape)