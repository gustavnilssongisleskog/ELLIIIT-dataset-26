import numpy as np
import localFunctions as lf
import json
import matplotlib.pyplot as plt
import tikzplotlib

with open('config.json') as json_file:
    config = json.load(json_file)

save_loc = config['save_loc']

# Analysis variables
plot_traditional_analysis = True
plot_NN_analysis = False
plot_CDF_ranging_traditional = True
plot_CDF_eucl_dist = True
plot_CDF_x_error = True
plot_CDF_y_error = True
plot_CDF_z_error = True
plot_heatmap_eucl = True
plot_heatmap_x_error = True
plot_heatmap_y_error = True
plot_heatmap_z_error = True
plot_all_CDF_one = True
plot_CDF_allinone = True

n_anchors_used = config['number_used_anchors_pos'][0] #TODO fix to array for IPIN   n_mics is changed to this
print(n_anchors_used, ' anchor nodes were used')
anchor_locs = np.load(save_loc+'Sim_data\\anchor_positions.npy')[:,0:3]
mn_loc_all = np.load(save_loc+'Sim_data\\positions_mobile_node.npy')[:,0:3]
print('\nTotal amount of mobile node positions: ', np.size(mn_loc_all, axis=0))

# Base only on test set to compare: filter out test set
out_counted = np.load(save_loc+'Sim_data\\outcounted.npy')
# Calculate amount of test set grid points (for non-shoebox not n_x_test * n_y_test * n_z_test)
n_test_set_positions = (config['n_x_test'] * config['n_y_test'] * config['n_z_test'])-out_counted
mn_loc_test_set = mn_loc_all[:n_test_set_positions, :]
print('\nAmount of test set mobile node positions: {}'.format(n_test_set_positions))

path_output_data = save_loc+'ML_Results_Journal\\'
path_output_data_all_combined = save_loc+'ML_Results_Journal\\Result_figs\\'
path_estimation_data = save_loc+'ML_Results_Journal\\3d_pos_data_results\\'

#LPF models location
LPF_ANN_loc = path_output_data+'ANN\\LPF\\'
LPF_notonehot_ANN_loc = path_output_data+'ANN\\LPF\\not_onehot_encoded\\'
LPF_CCNN_loc = path_output_data+'CCNN\\LPF\\'
LPF_CNN_loc = path_output_data+'CNN\\LPF\\'
LPF_GNN_loc = path_output_data+'GNN\\LPF\\'

#AUDIO models location
audio_ANN_loc = path_output_data+'ANN\\Audio\\'
audio_CCNN_loc = path_output_data+'CCNN\\Audio\\'
audio_CCNN_smalltobig_loc = path_output_data+'CCNN\\Audio\\small-to-big\\'
audio_GNN_loc = path_output_data+'GNN\\Audio\\'

#Correlation models location
correlation_ANN_loc = path_output_data+'ANN\\Correlation\\'
correlation_CCNN_loc = path_output_data+'CCNN\\Correlation\\'
correlation_CCNN_smalltobig_loc = path_output_data+'CCNN\\Correlation\\small-to-big\\'
correlation_GNN_loc = path_output_data+'GNN\\Correlation\\'

#Spectrogram models location
spect_CCNN1_loc = path_output_data+'CCNN\\Spectrogram\\ccnn1\\'
spect_CCNN2_loc = path_output_data+'CCNN\\Spectrogram\\ccnn2\\'
spect_CCNN_t_loc = path_output_data+'CCNN\\Spectrogram\\ccnn_test\\'
spect_GNN_loc = path_output_data+'GNN\\Spectrogram\\'

#IPIN2024 data
IPIN2024_data_loc = path_output_data+'All_anchors_used_ipin2024\\'
model_based_path = path_output_data+'modelbased\\'

# LOAD DATA
#IPIN2024 data
ALL_anchors_IPIN2024_test = np.load(IPIN2024_data_loc+'estimation_data_all_selected_test.npy', allow_pickle=True)
ALL_anchors_IPIN2024_dev = np.load(IPIN2024_data_loc+'estimation_data_all_selected_dev.npy', allow_pickle=True)
ALL_anchors_IPIN2024_train = np.load(IPIN2024_data_loc+'estimation_data_all_selected_train.npy', allow_pickle=True)

IPIN2024_6Selected_CCNN_test = np.load(IPIN2024_data_loc+'estimation_data_6_selected_ordered_test.npy', allow_pickle=True)
IPIN2024_6Selected_CCNN_dev = np.load(IPIN2024_data_loc+'estimation_data_6_selected_ordered_dev.npy', allow_pickle=True)
IPIN2024_6Selected_CCNN_train = np.load(IPIN2024_data_loc+'estimation_data_6_selected_ordered_train.npy', allow_pickle=True)

model_based = np.load(IPIN2024_data_loc+'estimation_data_dicts_modelbased.npy', allow_pickle=True)

#LPF models
LPF_ANN_test = np.load(path_estimation_data+'LPF_ANN_test.npy', allow_pickle=True)
LPF_ANN_dev = np.load(path_estimation_data+'LPF_ANN_dev.npy', allow_pickle=True)
LPF_ANN_train = np.load(path_estimation_data+'LPF_ANN_train.npy', allow_pickle=True)

LPF_ANN_notonehot_test = np.load(path_estimation_data+'LPF_ANN_notOnehot_test.npy', allow_pickle=True)
LPF_ANN_notonehot_dev = np.load(path_estimation_data+'LPF_ANN_notOnehot_dev.npy', allow_pickle=True)
LPF_ANN_notonehot_train = np.load(path_estimation_data+'LPF_ANN_notOnehot_train.npy', allow_pickle=True)

LPF_CCNN_test = np.load(path_estimation_data+'LPF_CCNN_test.npy', allow_pickle=True)
LPF_CCNN_dev = np.load(path_estimation_data+'LPF_CCNN_dev.npy', allow_pickle=True)
LPF_CCNN_train = np.load(path_estimation_data+'LPF_CCNN_train.npy', allow_pickle=True)

LPF_CNN_test = np.load(path_estimation_data+'LPF_CNN_test.npy', allow_pickle=True)
LPF_CNN_dev = np.load(path_estimation_data+'LPF_CNN_dev.npy', allow_pickle=True)
LPF_CNN_train = np.load(path_estimation_data+'LPF_CNN_train.npy', allow_pickle=True)

LPF_GNN_test = np.load(path_estimation_data+'LPF_GNN_test.npy', allow_pickle=True)
LPF_GNN_dev = np.load(path_estimation_data+'LPF_GNN_dev.npy', allow_pickle=True)
LPF_GNN_train = np.load(path_estimation_data+'LPF_GNN_train.npy', allow_pickle=True)

#audio
AUDIO_ANN_test = np.load(path_estimation_data+'Audio_ANN_test.npy', allow_pickle=True)
AUDIO_ANN_dev = np.load(path_estimation_data+'Audio_ANN_dev.npy', allow_pickle=True)
AUDIO_ANN_train = np.load(path_estimation_data+'Audio_ANN_train.npy', allow_pickle=True)

AUDIO_CCNN_test = np.load(path_estimation_data+'Audio_CCNN_test.npy', allow_pickle=True)
AUDIO_CCNN_dev = np.load(path_estimation_data+'Audio_CCNN_dev.npy', allow_pickle=True)
AUDIO_CCNN_train = np.load(path_estimation_data+'Audio_CCNN_train.npy', allow_pickle=True)

AUDIO_CCNN_smalltobig_test = np.load(path_estimation_data+'Audio_CCNN_smalltobig_test.npy', allow_pickle=True)
AUDIO_CCNN_smalltobig_dev = np.load(path_estimation_data+'Audio_CCNN_smalltobig_dev.npy', allow_pickle=True)
AUDIO_CCNN_smalltobig_train = np.load(path_estimation_data+'Audio_CCNN_smalltobig_train.npy', allow_pickle=True)

AUDIO_GNN_test = np.load(path_estimation_data+'Audio_GNN_test.npy', allow_pickle=True)
AUDIO_GNN_dev = np.load(path_estimation_data+'Audio_GNN_dev.npy', allow_pickle=True)
AUDIO_GNN_train = np.load(path_estimation_data+'Audio_GNN_train.npy', allow_pickle=True)

#Correlation
CORRELATION_ANN_test = np.load(path_estimation_data+'Correlation_ANN_test.npy', allow_pickle=True)
CORRELATION_ANN_dev = np.load(path_estimation_data+'Correlation_ANN_dev.npy', allow_pickle=True)
CORRELATION_ANN_train = np.load(path_estimation_data+'Correlation_ANN_train.npy', allow_pickle=True)

CORRELATION_CCNN_test = np.load(path_estimation_data+'Correlation_CCNN_test.npy', allow_pickle=True)
CORRELATION_CCNN_dev = np.load(path_estimation_data+'Correlation_CCNN_dev.npy', allow_pickle=True)
CORRELATION_CCNN_train = np.load(path_estimation_data+'Correlation_CCNN_train.npy', allow_pickle=True)

CORRELATION_GNN_test = np.load(path_estimation_data+'Correlation_GNN_test.npy', allow_pickle=True)
CORRELATION_GNN_dev = np.load(path_estimation_data+'Correlation_GNN_dev.npy', allow_pickle=True)
CORRELATION_GNN_train = np.load(path_estimation_data+'Correlation_GNN_train.npy', allow_pickle=True)

if not config['shoebox']:
    CORRELATION_CCNN_smalltobig_test = np.load(path_estimation_data+'Correlation_CCNN_smalltobig_test.npy', allow_pickle=True)
    CORRELATION_CCNN_smalltobig_dev = np.load(path_estimation_data+'Correlation_CCNN_smalltobig_dev.npy', allow_pickle=True)
    CORRELATION_CCNN_smalltobig_train = np.load(path_estimation_data+'Correlation_CCNN_smalltobig_train.npy', allow_pickle=True)

# ranging_faults_all = np.load(path_output_data+'rangingfault_all.npy')

#Spectrogram
SPECT_CCNN1_test = np.load(path_estimation_data+'Spectrogram_CCNN_1_test.npy', allow_pickle=True)
SPECT_CCNN1_dev = np.load(path_estimation_data+'Spectrogram_CCNN_1_dev.npy', allow_pickle=True)
SPECT_CCNN1_train = np.load(path_estimation_data+'Spectrogram_CCNN_1_train.npy', allow_pickle=True)

# SPECT_CCNN2_test = np.load(path_estimation_data+'Spectrogram_CCNN2_test.npy', allow_pickle=True)
# SPECT_CCNN2_dev = np.load(path_estimation_data+'Spectrogram_CCNN2_dev.npy', allow_pickle=True)
# SPECT_CCNN2_train = np.load(path_estimation_data+'Spectrogram_CCNN2_train.npy', allow_pickle=True)
#
# SPECT_CCNN_t_test = np.load(path_estimation_data+'Spectrogram_CCNN_test_test.npy', allow_pickle=True)
# SPECT_CCNN_t_dev = np.load(path_estimation_data+'Spectrogram_CCNN_test_dev.npy', allow_pickle=True)
# SPECT_CCNN_t_train = np.load(path_estimation_data+'Spectrogram_CCNN_test_train.npy', allow_pickle=True)

SPECT_GNN_test = np.load(path_estimation_data+'Spectrogram_GNN_test.npy', allow_pickle=True)
SPECT_GNN_dev = np.load(path_estimation_data+'Spectrogram_GNN_dev.npy', allow_pickle=True)
SPECT_GNN_train = np.load(path_estimation_data+'Spectrogram_GNN_train.npy', allow_pickle=True)

def analyse(estimation_data):
    """
    Analyses the data
    :param estimation_data: all data given by dict
    :return: sorted_vals, error_vals, speaker_loc_all
    """
    mn_locs_from_data = [estimation_data[pos_nr]['mn_loc'] for pos_nr in range(len(estimation_data))]
    pos_estimate_all = [estimation_data[pos_nr]['pos_estimate'] for pos_nr in range(len(estimation_data))]
    euclid_dist_error_all = [estimation_data[pos_nr]['eucl_dist_error'] for pos_nr in range(len(estimation_data))]
    euclid_dist_error_all_2D = [estimation_data[pos_nr]['eucl_dist_error2D'] for pos_nr in range(len(estimation_data))]
    x_error_all = [estimation_data[pos_nr]['x_error'] for pos_nr in range(len(estimation_data))]
    y_error_all = [estimation_data[pos_nr]['y_error'] for pos_nr in range(len(estimation_data))]
    z_error_all = [estimation_data[pos_nr]['z_error'] for pos_nr in range(len(estimation_data))]

    x_error_all = np.asarray(x_error_all).ravel()
    y_error_all = np.asarray(y_error_all).ravel()
    z_error_all = np.asarray(z_error_all).ravel()
    error_vals = [euclid_dist_error_all, euclid_dist_error_all_2D, x_error_all, y_error_all, z_error_all]

    sorted_eucl_dist = np.sort(np.abs(euclid_dist_error_all))
    sorted_eucl_dist_2D = np.sort(np.abs(euclid_dist_error_all_2D))
    sorted_x_error = np.sort(np.abs(x_error_all))
    sorted_y_error = np.sort(np.abs(y_error_all))
    sorted_z_error = np.sort(np.abs(z_error_all))
    sorted_vals = [sorted_eucl_dist, sorted_eucl_dist_2D, sorted_x_error, sorted_y_error, sorted_z_error]

    return sorted_vals, error_vals, mn_locs_from_data

def plot_figs(sorted_vals, error_vals, mn_locations, anchor_locs ,method, cmax, path_fig):
    if plot_CDF_eucl_dist:
        lf.plot_CDF_one(sorted_vals[0], 'CDF of 3D Euclidean distance errors ('+method+')', 'CDF_plot_eucl_dist_3D'+method, cmax, path_fig)

    if plot_CDF_eucl_dist:
        lf.plot_CDF_one(sorted_vals[1], 'CDF of 2D Euclidean distance errors ('+method+')', 'CDF_plot_eucl_dist_2D'+method, cmax, path_fig)

    if plot_CDF_x_error:
        lf.plot_CDF_one(sorted_vals[2], 'CDF of x errors ('+method+')', 'CDF_plot_x_error'+method, cmax, path_fig)

    if plot_CDF_y_error:
         lf.plot_CDF_one(sorted_vals[3], 'CDF of y errors ('+method+')', 'CDF_plot_y_error'+method, cmax, path_fig)

    if plot_CDF_z_error:
        lf.plot_CDF_one(sorted_vals[4], 'CDF of z errors ('+method+')', 'CDF_plot_z_error'+method, cmax, path_fig)

    if plot_CDF_allinone:
        lf.plot_multiple_CDF_oneTech(sorted_vals, 'CDF of position predictions ('+method+')', 'CDF_allinone' +method, cmax, path_fig)

    if plot_heatmap_eucl:
        lf.plot_room_errors(np.array(mn_locations), error_vals[0], anchor_locs, 'heatmap_Eucl_3D'+method,
                            '3D Euclidean distance error on position estimation ('+method+')', cmax, path_fig)

        lf.plot_room_errors(np.array(mn_locations), error_vals[1], anchor_locs, 'heatmap_Eucl_2D' + method,
                            '2D Euclidean distance error on position estimation (' + method + ')', cmax, path_fig)

    if plot_heatmap_x_error:
        lf.plot_room_errors(np.array(mn_locations), np.abs(error_vals[2]), anchor_locs, 'heatmap_x_error_'+method,
                            'X distance error on position estimation ('+method+')', cmax, path_fig)

    if plot_heatmap_y_error:
        lf.plot_room_errors(np.array(mn_locations), np.abs(error_vals[3]), anchor_locs, 'heatmap_y_error_' + method,
                        'Y distance error on position estimation ('+method+')', cmax, path_fig)

    if plot_heatmap_z_error:
        lf.plot_room_errors(np.array(mn_locations), np.abs(error_vals[4]), anchor_locs, 'heatmap_z_error_' + method,
                        'Z distance error on position estimation ('+method+')', cmax, path_fig)


if __name__ == '__main__':

    cmax = 1
    #LPF -----------------------------------------------------------------------------------------
    sorted_LPF_ANN_test, error_LPF_ANN_test, mn_loc_LPF_ANN_test = analyse(LPF_ANN_test)
    sorted_LPF_ANN_dev, error_LPF_ANN_dev, mn_loc_LPF_ANN_dev = analyse(LPF_ANN_dev)
    sorted_LPF_ANN_train, error_LPF_ANN_train, mn_loc_LPF_ANN_train = analyse(LPF_ANN_train)

    sorted_LPF_ANN_notonehot_test, error_LPF_ANN_notonehot_test, mn_loc_LPF_ANN_notonehot_test = analyse(LPF_ANN_notonehot_test)
    sorted_LPF_ANN_notonehot_dev, error_LPF_ANN_notonehot_dev, mn_loc_LPF_ANN_notonehot_dev = analyse(LPF_ANN_notonehot_dev)
    sorted_LPF_ANN_notonehot_train, error_LPF_ANN_notonehot_train, mn_loc_LPF_ANN_notonehot_train = analyse(LPF_ANN_notonehot_train)

    sorted_LPF_CCNN_test, error_LPF_CCNN_test, mn_loc_LPF_CCNN_test = analyse(LPF_CCNN_test)
    sorted_LPF_CCNN_dev, error_LPF_CCNN_dev, mn_loc_LPF_CCNN_dev = analyse(LPF_CCNN_dev)
    sorted_LPF_CCNN_train, error_LPF_CCNN_train, mn_loc_LPF_CCNN_train = analyse(LPF_CCNN_train)

    sorted_LPF_CNN_test, error_LPF_CNN_test, mn_loc_LPF_CNN_test = analyse(LPF_CNN_test)
    sorted_LPF_CNN_dev, error_LPF_CNN_dev, mn_loc_LPF_CNN_dev = analyse(LPF_CNN_dev)
    sorted_LPF_CNN_train, error_LPF_CNN_train, mn_loc_LPF_CNN_train = analyse(LPF_CNN_train)

    sorted_LPF_GNN_test, error_LPF_GNN_test, mn_loc_LPF_GNN_test = analyse(LPF_GNN_test)
    sorted_LPF_GNN_dev, error_LPF_GNN_dev, mn_loc_LPF_GNN_dev = analyse(LPF_GNN_dev)
    sorted_LPF_GNN_train, error_LPF_GNN_train, mn_loc_LPF_GNN_train = analyse(LPF_GNN_train)
    # audio-----------------------------------------------------------------------------------------
    sorted_AUDIO_ANN_test, error_AUDIO_ANN_test, mn_loc_AUDIO_ANN_test = analyse(AUDIO_ANN_test)
    sorted_AUDIO_ANN_dev, error_AUDIO_ANN_dev, mn_loc_AUDIO_ANN_dev = analyse(AUDIO_ANN_dev)
    sorted_AUDIO_ANN_train, error_AUDIO_ANN_train, mn_loc_AUDIO_ANN_train = analyse(AUDIO_ANN_train)

    sorted_AUDIO_CCNN_test, error_AUDIO_CCNN_test, mn_loc_AUDIO_CCNN_test = analyse(AUDIO_CCNN_test)
    sorted_AUDIO_CCNN_dev, error_AUDIO_CCNN_dev, mn_loc_AUDIO_CCNN_dev = analyse(AUDIO_CCNN_dev)
    sorted_AUDIO_CCNN_train, error_AUDIO_CCNN_train, mn_loc_AUDIO_CCNN_train = analyse(AUDIO_CCNN_train)

    sorted_AUDIO_CCNN_smalltobig_test, error_AUDIO_CCNN_smalltobig_test, mn_loc_AUDIO_CCNN_smalltobig_test = analyse(AUDIO_CCNN_smalltobig_test)
    sorted_AUDIO_CCNN_smalltobig_dev, error_AUDIO_CCNN_smalltobig_dev, mn_loc_AUDIO_CCNN_smalltobig_dev = analyse(AUDIO_CCNN_smalltobig_dev)
    sorted_AUDIO_CCNN_smalltobig_train, error_AUDIO_CCNN_smalltobig_train, mn_loc_AUDIO_CCNN_smalltobig_train = analyse(AUDIO_CCNN_smalltobig_train)

    sorted_AUDIO_GNN_test, error_AUDIO_GNN_test, mn_loc_AUDIO_GNN_test = analyse(AUDIO_GNN_test)
    sorted_AUDIO_GNN_dev, error_AUDIO_GNN_dev, mn_loc_AUDIO_GNN_dev = analyse(AUDIO_GNN_dev)
    sorted_AUDIO_GNN_train, error_AUDIO_GNN_train, mn_loc_AUDIO_GNN_train = analyse(AUDIO_GNN_train)
    # Correlation ----------------------------------------------------------------------------------
    sorted_CORRELATION_ANN_test, error_CORRELATION_ANN_test, mn_loc_CORRELATION_ANN_test = analyse(CORRELATION_ANN_test)
    sorted_CORRELATION_ANN_dev, error_CORRELATION_ANN_dev, mn_loc_CORRELATION_ANN_dev = analyse(CORRELATION_ANN_dev)
    sorted_CORRELATION_ANN_train, error_CORRELATION_ANN_train, mn_loc_CORRELATION_ANN_train = analyse(CORRELATION_ANN_train)

    sorted_CORRELATION_CCNN_test, error_CORRELATION_CCNN_test, mn_loc_CORRELATION_CCNN_test = analyse(CORRELATION_CCNN_test)
    sorted_CORRELATION_CCNN_dev, error_CORRELATION_CCNN_dev, mn_loc_CORRELATION_CCNN_dev = analyse(CORRELATION_CCNN_dev)
    sorted_CORRELATION_CCNN_train, error_CORRELATION_CCNN_train, mn_loc_CORRELATION_CCNN_train = analyse(CORRELATION_CCNN_train)

    sorted_CORRELATION_GNN_test, error_CORRELATION_GNN_test, mn_loc_CORRELATION_GNN_test = analyse(CORRELATION_GNN_test)
    sorted_CORRELATION_GNN_dev, error_CORRELATION_GNN_dev, mn_loc_CORRELATION_GNN_dev = analyse(CORRELATION_GNN_dev)
    sorted_CORRELATION_GNN_train, error_CORRELATION_GNN_train, mn_loc_CORRELATION_GNN_train = analyse(CORRELATION_GNN_train)

    if not config['shoebox']:
        sorted_CORRELATION_CCNN_smalltobig_test, error_CORRELATION_CCNN_smalltobig_test, mn_loc_CORRELATION_CCNN_smalltobig_test = analyse(CORRELATION_CCNN_smalltobig_test)
        sorted_CORRELATION_CCNN_smalltobig_dev, error_CORRELATION_CCNN_smalltobig_dev, mn_loc_CORRELATION_CCNN_smalltobig_dev = analyse(CORRELATION_CCNN_smalltobig_dev)
        sorted_CORRELATION_CCNN_smalltobig_train, error_CORRELATION_CCNN_smalltobig_train, mn_loc_CORRELATION_CCNN_smalltobig_train = analyse(CORRELATION_CCNN_smalltobig_train)

    sorted_SPECT_CCNN1_test, error_SPECT_CCNN1_test, mn_loc_SPECT_CCNN1_test = analyse(SPECT_CCNN1_test)
    sorted_SPECT_CCNN1_dev, error_SPECT_CCNN1_dev, mn_loc_SPECT_CCNN1_dev = analyse(SPECT_CCNN1_dev)
    sorted_SPECT_CCNN1_train, error_SPECT_CCNN1_train, mn_loc_SPECT_CCNN1_train = analyse(SPECT_CCNN1_train)

    # sorted_SPECT_CCNN2_test, error_SPECT_CCNN2_test, mn_loc_SPECT_CCNN2_test = analyse(SPECT_CCNN2_test)
    # sorted_SPECT_CCNN2_dev, error_SPECT_CCNN2_dev, mn_loc_SPECT_CCNN2_dev = analyse(SPECT_CCNN2_dev)
    # sorted_SPECT_CCNN2_train, error_SPECT_CCNN2_train, mn_loc_SPECT_CCNN2_train = analyse(SPECT_CCNN2_train)
    #
    # sorted_SPECT_CCNN_t_test, error_SPECT_CCNN_t_test, mn_loc_SPECT_CCNN_t_test = analyse(SPECT_CCNN_t_test)
    # sorted_SPECT_CCNN_t_dev, error_SPECT_CCNN_t_dev, mn_loc_SPECT_CCNN_t_dev = analyse(SPECT_CCNN_t_dev)
    # sorted_SPECT_CCNN_t_train, error_SPECT_CCNN_t_train, mn_loc_SPECT_CCNN_t_train = analyse(SPECT_CCNN_t_train)

    sorted_SPECT_GNN_test, error_SPECT_GNN_test, mn_loc_SPECT_GNN_test = analyse(SPECT_GNN_test)
    sorted_SPECT_GNN_dev, error_SPECT_GNN_dev, mn_loc_SPECT_GNN_dev = analyse(SPECT_GNN_dev)
    sorted_SPECT_GNN_train, error_SPECT_GNN_train, mn_loc_SPECT_GNN_train = analyse(SPECT_GNN_train)


    # IPIN2024 DATA
    sorted_IPIN2024_all_anchors_test, error_IPIN2024_all_anchors_test, mn_loc_IPIN2024_all_anchors_test = analyse(ALL_anchors_IPIN2024_test)
    sorted_IPIN2024_all_anchors_dev, error_IPIN2024_all_anchors_dev, mn_loc_IPIN2024_all_anchors_dev = analyse(ALL_anchors_IPIN2024_dev)
    sorted_IPIN2024_all_anchors_train, error_IPIN2024_all_anchors_train, mn_loc_IPIN2024_all_anchors_train = analyse(ALL_anchors_IPIN2024_train)

    sorted_IPIN2024_6_anchors_test, error_IPIN2024_6_anchors_test, mn_loc_IPIN2024_6_anchors_test = analyse(IPIN2024_6Selected_CCNN_test)
    sorted_IPIN2024_6_anchors_dev, error_IPIN2024_6_anchors_dev, mn_loc_IPIN2024_6_anchors_dev = analyse(IPIN2024_6Selected_CCNN_dev)
    sorted_IPIN2024_6_anchors_train, error_IPIN2024_6_anchors_train, mn_loc_IPIN2024_6_anchors_train = analyse(IPIN2024_6Selected_CCNN_train)

    sorted_model_based, error_model_based, mn_loc_model_based = analyse(model_based)

    # # # #Plot single figs per use case:
    # #IPIN2024 Data
    # plot_figs(sorted_IPIN2024_all_anchors_test, error_IPIN2024_all_anchors_test, mn_loc_IPIN2024_all_anchors_test, anchor_locs, 'ALL Anchors CCNN IPIN2024 (test set)', cmax=cmax, path_fig=IPIN2024_data_loc)
    # plot_figs(sorted_IPIN2024_all_anchors_dev, error_IPIN2024_all_anchors_dev, mn_loc_IPIN2024_all_anchors_dev, anchor_locs, 'ALL Anchors CCNN IPIN2024 (dev set)', cmax=cmax, path_fig=IPIN2024_data_loc)
    # plot_figs(sorted_IPIN2024_all_anchors_train, error_IPIN2024_all_anchors_train, mn_loc_IPIN2024_all_anchors_train, anchor_locs, 'ALL Anchors CCNN IPIN2024 (train set)', cmax=cmax, path_fig=IPIN2024_data_loc)
    #
    # plot_figs(sorted_IPIN2024_6_anchors_test, error_IPIN2024_6_anchors_test, mn_loc_IPIN2024_6_anchors_test, anchor_locs, '6 Anchors RMS CCNN IPIN2024 (test set)', cmax=cmax, path_fig=IPIN2024_data_loc)
    # plot_figs(sorted_IPIN2024_6_anchors_dev, error_IPIN2024_6_anchors_dev, mn_loc_IPIN2024_6_anchors_dev, anchor_locs, '6 Anchors RMS CCNN IPIN2024 (dev set)', cmax=cmax, path_fig=IPIN2024_data_loc)
    # plot_figs(sorted_IPIN2024_6_anchors_train, error_IPIN2024_6_anchors_train, mn_loc_IPIN2024_6_anchors_train, anchor_locs, '6 Anchors RMS CCNN IPIN2024 (train set)', cmax=cmax, path_fig=IPIN2024_data_loc)

    #plot_figs(sorted_model_based, error_model_based, mn_loc_model_based, anchor_locs, 'Model-based RMS 6 Anchors IPIN2024', cmax=0.6, path_fig=model_based_path)

    #
    # #LPF
    # plot_figs(sorted_LPF_ANN_test, error_LPF_ANN_test, mn_loc_LPF_ANN_test, anchor_locs, 'LPF MLP 6 Anchors RMS (test set)', cmax=cmax, path_fig=LPF_ANN_loc)
    # plot_figs(sorted_LPF_ANN_dev, error_LPF_ANN_dev, mn_loc_LPF_ANN_dev, anchor_locs, 'LPF MLP 6 Anchors RMS (dev set)', cmax=cmax, path_fig= LPF_ANN_loc)
    # plot_figs(sorted_LPF_ANN_train, error_LPF_ANN_train, mn_loc_LPF_ANN_train, anchor_locs, 'LPF MLP 6 Anchors RMS (train set)', cmax=cmax, path_fig= LPF_ANN_loc)
    #
    # plot_figs(sorted_LPF_ANN_notonehot_test, error_LPF_ANN_notonehot_test, mn_loc_LPF_ANN_notonehot_test, anchor_locs, 'LPF MLP 6 Anchors RMS no one-hot (test set)', cmax=cmax, path_fig=LPF_notonehot_ANN_loc)
    # plot_figs(sorted_LPF_ANN_notonehot_dev, error_LPF_ANN_notonehot_dev, mn_loc_LPF_ANN_notonehot_dev, anchor_locs, 'LPF MLP 6 Anchors RMS no one-hot (dev set)', cmax=cmax, path_fig= LPF_notonehot_ANN_loc)
    # plot_figs(sorted_LPF_ANN_notonehot_train, error_LPF_ANN_notonehot_train, mn_loc_LPF_ANN_notonehot_train, anchor_locs, 'LPF MLP 6 Anchors RMS no one-hot (train set)', cmax=cmax, path_fig= LPF_notonehot_ANN_loc)
    #
    # plot_figs(sorted_LPF_CCNN_test, error_LPF_CCNN_test, mn_loc_LPF_CCNN_test, anchor_locs, 'LPF CCNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=LPF_CCNN_loc)
    # plot_figs(sorted_LPF_CCNN_dev, error_LPF_CCNN_dev, mn_loc_LPF_CCNN_dev, anchor_locs, 'LPF CCNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig= LPF_CCNN_loc)
    # plot_figs(sorted_LPF_CCNN_train, error_LPF_CCNN_train, mn_loc_LPF_CCNN_train, anchor_locs, 'LPF CCNN 6 Anchors RMS (train set)', cmax=cmax, path_fig= LPF_CCNN_loc)
    #
    # plot_figs(sorted_LPF_CNN_test, error_LPF_CNN_test, mn_loc_LPF_CNN_test, anchor_locs, 'LPF CNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=LPF_CNN_loc)
    # plot_figs(sorted_LPF_CNN_dev, error_LPF_CNN_dev, mn_loc_LPF_CNN_dev, anchor_locs, 'LPF CNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig= LPF_CNN_loc)
    # plot_figs(sorted_LPF_CNN_train, error_LPF_CNN_train, mn_loc_LPF_CNN_train, anchor_locs, 'LPF CNN 6 Anchors RMS (train set)', cmax=cmax, path_fig= LPF_CNN_loc)
    #
    # plot_figs(sorted_LPF_GNN_test, error_LPF_GNN_test, mn_loc_LPF_GNN_test, anchor_locs, 'LPF GNN 6 Anchors RMS (test set)', cmax=0.6, path_fig=LPF_GNN_loc)
    # plot_figs(sorted_LPF_GNN_dev, error_LPF_GNN_dev, mn_loc_LPF_GNN_dev, anchor_locs, 'LPF GNN 6 Anchors RMS (dev set)', cmax=0.6, path_fig= LPF_GNN_loc)
    # plot_figs(sorted_LPF_GNN_train, error_LPF_GNN_train, mn_loc_LPF_GNN_train, anchor_locs, 'LPF GNN 6 Anchors RMS (train set)', cmax=0.6, path_fig= LPF_GNN_loc)

    # #AUDIO
    # plot_figs(sorted_AUDIO_ANN_test, error_AUDIO_ANN_test, mn_loc_AUDIO_ANN_test, anchor_locs, 'AUDIO MLP 6 Anchors RMS (test set)', cmax=cmax, path_fig=audio_ANN_loc)
    # plot_figs(sorted_AUDIO_ANN_dev, error_AUDIO_ANN_dev, mn_loc_AUDIO_ANN_dev, anchor_locs, 'AUDIO MLP 6 Anchors RMS (dev set)', cmax=cmax, path_fig= audio_ANN_loc)
    # plot_figs(sorted_AUDIO_ANN_train, error_AUDIO_ANN_train, mn_loc_AUDIO_ANN_train, anchor_locs, 'AUDIO MLP 6 Anchors RMS (train set)', cmax=cmax, path_fig= audio_ANN_loc)
    #
    # plot_figs(sorted_AUDIO_CCNN_test, error_AUDIO_CCNN_test, mn_loc_AUDIO_CCNN_test, anchor_locs, 'AUDIO CCNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=audio_CCNN_loc)
    # plot_figs(sorted_AUDIO_CCNN_dev, error_AUDIO_CCNN_dev, mn_loc_AUDIO_CCNN_dev, anchor_locs, 'AUDIO CCNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig= audio_CCNN_loc)
    # plot_figs(sorted_AUDIO_CCNN_train, error_AUDIO_CCNN_train, mn_loc_AUDIO_CCNN_train, anchor_locs, 'AUDIO CCNN 6 Anchors RMS (train set)', cmax=cmax, path_fig= audio_CCNN_loc)
    #
    # plot_figs(sorted_AUDIO_CCNN_smalltobig_test, error_AUDIO_CCNN_smalltobig_test, mn_loc_AUDIO_CCNN_smalltobig_test, anchor_locs, 'AUDIO CCNN 6 Anchors RMS small-to-big (test set)', cmax=cmax, path_fig=audio_CCNN_smalltobig_loc)
    # plot_figs(sorted_AUDIO_CCNN_smalltobig_dev, error_AUDIO_CCNN_smalltobig_dev, mn_loc_AUDIO_CCNN_smalltobig_dev, anchor_locs, 'AUDIO CCNN 6 Anchors RMS small-to-big (dev set)', cmax=cmax, path_fig= audio_CCNN_smalltobig_loc)
    # plot_figs(sorted_AUDIO_CCNN_smalltobig_train, error_AUDIO_CCNN_smalltobig_train, mn_loc_AUDIO_CCNN_smalltobig_train, anchor_locs, 'AUDIO CCNN 6 Anchors RMS small-to-big (train set)', cmax=cmax, path_fig= audio_CCNN_smalltobig_loc)
    #
    # plot_figs(sorted_AUDIO_GNN_test, error_AUDIO_GNN_test, mn_loc_AUDIO_GNN_test, anchor_locs, 'AUDIO GNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=audio_GNN_loc)
    # plot_figs(sorted_AUDIO_GNN_dev, error_AUDIO_GNN_dev, mn_loc_AUDIO_GNN_dev, anchor_locs, 'AUDIO GNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig= audio_GNN_loc)
    # plot_figs(sorted_AUDIO_GNN_train, error_AUDIO_GNN_train, mn_loc_AUDIO_GNN_train, anchor_locs, 'AUDIO GNN 6 Anchors RMS (train set)', cmax=cmax, path_fig= audio_GNN_loc)


    # #CORRELATION
    # plot_figs(sorted_CORRELATION_ANN_test, error_CORRELATION_ANN_test, mn_loc_CORRELATION_ANN_test, anchor_locs, 'CORRELATION MLP 6 Anchors RMS (test set)', cmax=cmax, path_fig=correlation_ANN_loc)
    # plot_figs(sorted_CORRELATION_ANN_dev, error_CORRELATION_ANN_dev, mn_loc_CORRELATION_ANN_dev, anchor_locs, 'CORRELATION MLP 6 Anchors RMS (dev set)', cmax=cmax, path_fig=correlation_ANN_loc)
    # plot_figs(sorted_CORRELATION_ANN_train, error_CORRELATION_ANN_train, mn_loc_CORRELATION_ANN_train, anchor_locs, 'CORRELATION MLP 6 Anchors RMS (train set)', cmax=cmax, path_fig=correlation_ANN_loc)
    #
    # plot_figs(sorted_CORRELATION_CCNN_test, error_CORRELATION_CCNN_test, mn_loc_CORRELATION_CCNN_test, anchor_locs, 'CORRELATION CCNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=correlation_CCNN_loc)
    # plot_figs(sorted_CORRELATION_CCNN_dev, error_CORRELATION_CCNN_dev, mn_loc_CORRELATION_CCNN_dev, anchor_locs, 'CORRELATION CCNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig=correlation_CCNN_loc)
    # plot_figs(sorted_CORRELATION_CCNN_train, error_CORRELATION_CCNN_train, mn_loc_CORRELATION_CCNN_train, anchor_locs, 'CORRELATION CCNN 6 Anchors RMS (train set)', cmax=cmax, path_fig=correlation_CCNN_loc)
    #
    # if not config['shoebox']:
    #     plot_figs(sorted_CORRELATION_CCNN_smalltobig_test, error_CORRELATION_CCNN_smalltobig_test, mn_loc_CORRELATION_CCNN_smalltobig_test, anchor_locs, 'CORRELATION CCNN 6 Anchors RMS small-to-big (test set)', cmax=cmax, path_fig=correlation_CCNN_smalltobig_loc)
    #     plot_figs(sorted_CORRELATION_CCNN_smalltobig_dev, error_CORRELATION_CCNN_smalltobig_dev, mn_loc_CORRELATION_CCNN_smalltobig_dev, anchor_locs, 'CORRELATION CCNN 6 Anchors RMS small-to-big (dev set)', cmax=cmax, path_fig=correlation_CCNN_smalltobig_loc)
    #     plot_figs(sorted_CORRELATION_CCNN_smalltobig_train, error_CORRELATION_CCNN_smalltobig_train, mn_loc_CORRELATION_CCNN_smalltobig_train, anchor_locs, 'CORRELATION CCNN 6 Anchors RMS small-to-big (train set)', cmax=cmax, path_fig=correlation_CCNN_smalltobig_loc)

    # plot_figs(sorted_CORRELATION_GNN_test, error_CORRELATION_GNN_test, mn_loc_CORRELATION_GNN_test, anchor_locs, 'CORRELATION GNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=correlation_GNN_loc)
    # plot_figs(sorted_CORRELATION_GNN_dev, error_CORRELATION_GNN_dev, mn_loc_CORRELATION_GNN_dev, anchor_locs, 'CORRELATION GNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig=correlation_GNN_loc)
    # plot_figs(sorted_CORRELATION_GNN_train, error_CORRELATION_GNN_train, mn_loc_CORRELATION_GNN_train, anchor_locs, 'CORRELATION GNN 6 Anchors RMS (train set)', cmax=cmax, path_fig=correlation_GNN_loc)

    # #SPECTROGRAMS

    # plot_figs(sorted_SPECT_GNN_test, error_SPECT_GNN_test, mn_loc_SPECT_GNN_test, anchor_locs, 'Spectrogram GNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=spect_GNN_loc)
    # plot_figs(sorted_SPECT_GNN_dev, error_SPECT_GNN_dev, mn_loc_SPECT_GNN_dev, anchor_locs, 'Spectrogram GNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig=spect_GNN_loc)
    # plot_figs(sorted_SPECT_GNN_train, error_SPECT_GNN_train, mn_loc_SPECT_GNN_train, anchor_locs, 'Spectrogram GNN 6 Anchors RMS (train set)', cmax=cmax, path_fig=spect_GNN_loc)
    #
    # plot_figs(sorted_SPECT_CCNN1_test, error_SPECT_CCNN1_test, mn_loc_SPECT_CCNN1_test, anchor_locs, 'Spectrogram CCNN 6 Anchors RMS (test set)', cmax=cmax, path_fig=spect_CCNN1_loc)
    # plot_figs(sorted_SPECT_CCNN1_dev, error_SPECT_CCNN1_dev, mn_loc_SPECT_CCNN1_dev, anchor_locs, 'Spectrogram CCNN 6 Anchors RMS (dev set)', cmax=cmax, path_fig=spect_CCNN1_loc)
    # plot_figs(sorted_SPECT_CCNN1_train, error_SPECT_CCNN1_train, mn_loc_SPECT_CCNN1_train, anchor_locs, 'Spectrogram CCNN 6 Anchors RMS (train set)', cmax=cmax, path_fig=spect_CCNN1_loc)


    #---------------------- PLOT FIGURES FOR PAPER -------------------------------#
    # Plot 3D eucl dist CDF's for all options at the same graph
    def plot_CDF_matplotlib_LPF(mb, a, b, c, d, e, f,g, title, filename):
        p = 1. * np.arange(len(a)) / (len(a) - 1)

        P95_MB = lf.return_eval_values(mb)[3]
        # P95_IPIN2024_all = lf.return_eval_values(a)[3]
        # P95_IPIN2024_6 = lf.return_eval_values(b)[3]
        P95_CCNN = lf.return_eval_values(c)[3]
        P95_CNN = lf.return_eval_values(d)[3]
        P95_MLP = lf.return_eval_values(e)[3]
        P95_MLP_notonehot = lf.return_eval_values(f)[3]
        P95_GNN = lf.return_eval_values(g)[3]

        plt.plot(mb, p, color='#e41a1c', linestyle='dashed', label='Model-based, LPF, RMS 6 Anchors'),
        # plt.plot(a, p, color='#B4869F', linestyle='dotted', label='CCNN, LPF, All Anchors, IPIN2024'),
        # plt.plot(b, p, color='#632A50', linestyle='dotted', label='CCNN, LPF, RMS 6 Anchors, IPIN2024'),
        plt.plot(c, p, color='#247BA0', linestyle='dotted', label='CCNN, LPF, RMS 6 Anchors'),
        plt.plot(d, p, color='#a65628', linestyle=(0, (1, 1)), label='CNN, LPF, RMS 6 Anchors'),
        plt.plot(e, p, color='#7FB069', linestyle='dashdot', label='MLP, LPF, RMS 6 Anchors'),
        plt.plot(f, p, color='#053225', linestyle='dashdot', label='MLP, LPF, RMS 6 Anchors, not one-hot'), #5F6D5D E1CEAD
        plt.plot(g, p, color='#E6AA68', label='GNN, LPF, RMS 6 Anchors'), #5F6D5D E1CEAD

#ff7f00

        # plt.title(title)
        plt.xlabel('RMSE (m)')
        plt.ylabel('CDF')
        plt.legend()
        plt.xlim(0, 1.5)
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.3)

        # plt.vlines(P95_IPIN2024_all, -0.05, 0.95, color='#808080', linestyles='dashdot')
        # plt.vlines(P95_IPIN2024_6, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_CCNN, -0.05, 0.95, color='#808080', linestyles='dashdot')
        plt.vlines(P95_CNN, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_MLP, -0.05, 0.95, color='#808080', linestyles='dashdot')
        plt.vlines(P95_MLP_notonehot, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_GNN, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_MB, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.hlines(0.95, 0, 2.5, color='#808080', linestyles='dashdot')

        plt.savefig(filename+".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(filename+".svg")
        tikzplotlib.save(filename+".tex")
        plt.show()

    def plot_CDF_matplotlib_Correlation_Lshape(mb, a, b, c, d, e,f, title, filename):
        p = 1. * np.arange(len(a)) / (len(a) - 1)

        P95_MB = lf.return_eval_values(mb)[3]
        P95_IPIN2024_all = lf.return_eval_values(a)[3]
        P95_IPIN2024_6 = lf.return_eval_values(b)[3]
        P95_CCNN = lf.return_eval_values(c)[3]
        P95_CCNN2 = lf.return_eval_values(d)[3]
        P95_MLP = lf.return_eval_values(e)[3]
        P95_GNN = lf.return_eval_values(f)[3]


        plt.plot(mb, p, color='#e41a1c', linestyle='dashed', label='Model-based, LPF, RMS 6 Anchors'),
        # plt.plot(a, p, color='#B4869F', linestyle='dotted', label='CCNN, LPF, All Anchors, IPIN2024'),
        # plt.plot(b, p, color='#632A50', linestyle='dotted', label='CCNN, LPF, RMS 6 Anchors, IPIN2024'),
        plt.plot(c, p, color='#247BA0', linestyle='dotted', label='CCNN, Correlation, RMS 6 Anchors'),
        #plt.plot(d, p, color='#247BA0', linestyle='dotted', label='CCNN, Correlation, RMS 6 Anchors'), #small-to-big #CCNN2
        plt.plot(e, p, color='#7FB069', linestyle='dashdot', label='MLP, Correlation, RMS 6 Anchors'),
        plt.plot(f, p, color='#E6AA68', label='GNN, Correlation, RMS 6 Anchors'), #5F6D5D E1CEAD


        # plt.title(title)
        plt.xlabel('RMSE (m)')
        plt.ylabel('CDF')
        plt.legend()
        plt.xlim(0, 1.5)
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.3)

        # plt.vlines(P95_IPIN2024_all, -0.05, 0.95, color='#808080', linestyles='dashdot')
        # plt.vlines(P95_IPIN2024_6, -0.05, 0.95, color='#808080', linestyles='dashdot')

        # plt.vlines(P95_CCNN, -0.05, 0.95, color='#808080', linestyles='dashdot')
        plt.vlines(P95_CCNN2, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_MLP, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_GNN, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_MB, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.hlines(0.95, 0, 2.5, color='#808080', linestyles='dashdot')

        plt.savefig(filename+".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(filename+".svg")
        tikzplotlib.save(filename+".tex")
        plt.show()

    def plot_CDF_matplotlib_Correlation_Shoebox(mb, a, b, c, d, e, title, filename):
        p = 1. * np.arange(len(a)) / (len(a) - 1)

        P95_MB = lf.return_eval_values(mb)[3]
        # P95_IPIN2024_all = lf.return_eval_values(a)[3]
        # P95_IPIN2024_6 = lf.return_eval_values(b)[3]
        P95_CCNN = lf.return_eval_values(c)[3]
        P95_MLP = lf.return_eval_values(d)[3]
        P95_GNN = lf.return_eval_values(e)[3]


        plt.plot(mb, p, color='#e41a1c', linestyle='dashed', label='Model-based, LPF, RMS 6 Anchors'),
        # plt.plot(a, p, color='#B4869F', linestyle='dotted', label='CCNN, LPF, All Anchors, IPIN2024'),
        # plt.plot(b, p, color='#632A50', linestyle='dotted', label='CCNN, LPF, RMS 6 Anchors, IPIN2024'),
        plt.plot(c, p, color='#247BA0', linestyle='dotted', label='CCNN, Correlation, RMS 6 Anchors'),
        plt.plot(d, p, color='#7FB069', linestyle='dashdot', label='MLP, Correlation, RMS 6 Anchors'),
        plt.plot(e, p, color='#E6AA68', label='GNN, Correlation, RMS 6 Anchors'), #5F6D5D E1CEAD



        # plt.title(title)
        plt.xlabel('RMSE (m)')
        plt.ylabel('CDF')
        plt.legend()
        plt.xlim(0, 1.5)
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.3)

        # plt.vlines(P95_IPIN2024_all, -0.05, 0.95, color='#808080', linestyles='dashdot')
        # plt.vlines(P95_IPIN2024_6, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_CCNN, -0.05, 0.95, color='#808080', linestyles='dashdot')
        plt.vlines(P95_MLP, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_GNN, -0.05, 0.95, color='#808080', linestyles='dashdot')


        plt.vlines(P95_MB, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.hlines(0.95, 0, 2.5, color='#808080', linestyles='dashdot')

        plt.savefig(filename+".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(filename+".svg")
        tikzplotlib.save(filename+".tex")
        plt.show()

    def plot_CDF_matplotlib_Audio(mb, a, b, c, d, e,f, title, filename):
        p = 1. * np.arange(len(a)) / (len(a) - 1)

        P95_MB = lf.return_eval_values(mb)[3]
        # P95_IPIN2024_all = lf.return_eval_values(a)[3]
        # P95_IPIN2024_6 = lf.return_eval_values(b)[3]
        P95_CCNN = lf.return_eval_values(c)[3]
        P95_CCNN2 = lf.return_eval_values(d)[3]
        P95_MLP = lf.return_eval_values(e)[3]
        P95_GNN = lf.return_eval_values(f)[3]

        plt.plot(mb, p, color='#e41a1c', linestyle='dashed', label='Model-based, LPF, RMS 6 Anchors'),
        # plt.plot(a, p, color='#B4869F', linestyle='dotted', label='CCNN, LPF, All Anchors, IPIN2024'),
        # plt.plot(b, p, color='#632A50', linestyle='dotted', label='CCNN, LPF, RMS 6 Anchors, IPIN2024'),
        plt.plot(c, p, color='#247BA0', linestyle='dotted', label='CCNN, Audio, RMS 6 Anchors'), #CCNN1
        #plt.plot(d, p, color='#ff7f00', label='CCNN2, Audio, RMS 6 Anchors'), #small-to-big
        plt.plot(e, p, color='#7FB069', linestyle='dashdot', label='MLP, Audio, RMS 6 Anchors'),
        plt.plot(f, p, color='#E6AA68', label='GNN, Audio, RMS 6 Anchors'), #5F6D5D E1CEAD


        # plt.title(title)
        plt.xlabel('RMSE (m)')
        plt.ylabel('CDF')
        plt.legend()
        plt.xlim(0, 1.5)
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.3)
        #
        # plt.vlines(P95_IPIN2024_all, -0.05, 0.95, color='#808080', linestyles='dashdot')
        # plt.vlines(P95_IPIN2024_6, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_CCNN, -0.05, 0.95, color='#808080', linestyles='dashdot')
        # plt.vlines(P95_CCNN2, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_MLP, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_GNN, -0.05, 0.95, color='#808080', linestyles='dashdot')


        plt.vlines(P95_MB, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.hlines(0.95, 0, 2.5, color='#808080', linestyles='dashdot')

        plt.savefig(filename+".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(filename+".svg")
        tikzplotlib.save(filename+".tex")
        plt.show()


    def plot_CDF_matplotlib_spectrogram(mb, a, b, c,d, title, filename):
        p = 1. * np.arange(len(a)) / (len(a) - 1)
        p2 = 1. * np.arange(len(c)) / (len(c) - 1)

        P95_MB = lf.return_eval_values(mb)[3]
        P95_IPIN2024_all = lf.return_eval_values(a)[3]
        P95_IPIN2024_6 = lf.return_eval_values(b)[3]
        P95_CCNN = lf.return_eval_values(c)[3]
        P95_GNN = lf.return_eval_values(d)[3]


        plt.plot(mb, p, color='#e41a1c', linestyle='dashed', label='Model-based, LPF, RMS 6 Anchors'),
        # plt.plot(a, p, color='#B4869F', linestyle='dotted', label='CCNN, LPF, All Anchors, IPIN2024'),
        # plt.plot(b, p, color='#632A50', linestyle='dotted', label='CCNN, LPF, RMS 6 Anchors, IPIN2024'),
        plt.plot(c, p2, color='#247BA0', linestyle='dotted', label='CCNN, Spectrogram, RMS 6 Anchors'),
        plt.plot(d, p, color='#E6AA68', label='GNN, Spectrogram, RMS 6 Anchors'), #5F6D5D E1CEAD


        # plt.title(title)
        plt.xlabel('RMSE (m)')
        plt.ylabel('CDF')
        plt.legend()
        plt.xlim(0, 1.5)
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.3)

        # plt.vlines(P95_IPIN2024_all, -0.05, 0.95, color='#808080', linestyles='dashdot')
        # plt.vlines(P95_IPIN2024_6, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_CCNN, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.vlines(P95_GNN, -0.05, 0.95, color='#808080', linestyles='dashdot')


        plt.vlines(P95_MB, -0.05, 0.95, color='#808080', linestyles='dashdot')

        plt.hlines(0.95, 0, 2.5, color='#808080', linestyles='dashdot')

        plt.savefig(filename+".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(filename+".svg")
        tikzplotlib.save(filename+".tex")
        plt.show()

    # #use  path_output_data_all_combined

    # # Plot LPF part
    # if config['shoebox']:
    #     name = 'shoebox'
    # else:
    #     name = 'lshape'
    #
    # plot_CDF_matplotlib_LPF(sorted_model_based[0], sorted_IPIN2024_all_anchors_test[0], sorted_IPIN2024_6_anchors_test[0], sorted_LPF_CCNN_test[0],
    #                     sorted_LPF_CNN_test[0], sorted_LPF_ANN_test[0], sorted_LPF_ANN_notonehot_test[0],  sorted_LPF_GNN_test[0], 'CDFs LPF Input Data', path_output_data_all_combined+'LPF_models_CDF_journal'+name)
    #
    #
    # if not config['shoebox']:
    #     plot_CDF_matplotlib_Correlation_Lshape(sorted_model_based[0], sorted_IPIN2024_all_anchors_test[0],
    #                                     sorted_IPIN2024_6_anchors_test[0], sorted_CORRELATION_CCNN_test[0],
    #                                     sorted_CORRELATION_CCNN_smalltobig_test[0], sorted_CORRELATION_ANN_test[0], sorted_CORRELATION_GNN_test[0],
    #                                     'CDFs Correlation Input Data',
    #                                     path_output_data_all_combined + 'Correlation_models_CDF_journal'+name)
    # else:
    #     plot_CDF_matplotlib_Correlation_Shoebox(sorted_model_based[0], sorted_IPIN2024_all_anchors_test[0],
    #                                     sorted_IPIN2024_6_anchors_test[0], sorted_CORRELATION_CCNN_test[0],sorted_CORRELATION_ANN_test[0], sorted_CORRELATION_GNN_test[0],
    #                                     'CDFs Correlation Input Data',
    #                                     path_output_data_all_combined + 'Correlation_models_CDF_journal'+name)
    #
    # plot_CDF_matplotlib_Audio(sorted_model_based[0], sorted_IPIN2024_all_anchors_test[0],
    #                                 sorted_IPIN2024_6_anchors_test[0], sorted_AUDIO_CCNN_test[0],
    #                                 sorted_AUDIO_CCNN_smalltobig_test[0], sorted_AUDIO_ANN_test[0], sorted_AUDIO_GNN_test[0],
    #                                 'CDFs Audio Input Data',
    #                                 path_output_data_all_combined + 'Audio_models_CDF_journal'+name)
    #
    # plot_CDF_matplotlib_spectrogram(sorted_model_based[0], sorted_IPIN2024_all_anchors_test[0],
    #                                 sorted_IPIN2024_6_anchors_test[0],sorted_SPECT_CCNN1_test[0], sorted_SPECT_GNN_test[0], 'CDFs Spectrogram Input Data', path_output_data_all_combined + 'Spectrogram_models_CDF_journal'+name)
    #
    #
    # print('ACCURACY IN CM!!')
    # print('------------ANN LPF NOT ONEHOT-------------------------------------')
    # score_train = np.mean(sorted_LPF_ANN_notonehot_train[0])*100
    # # print('\nTrain loss :', score_train**2)
    # print('Train accuracy :', score_train)
    #
    # score_dev = np.mean(sorted_LPF_ANN_notonehot_dev[0])*100
    # # print('\nDev loss :', score_dev**2)
    # print('Dev accuracy :', score_dev)
    #
    # score_test = np.mean(sorted_LPF_ANN_notonehot_test[0])*100
    # # print('\nTest loss :', score_test**2)
    # print('Test accuracy :', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('-------------ANN LPF------------------------------------')
    # score_train = np.mean(sorted_LPF_ANN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_LPF_ANN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_LPF_ANN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('-------------CCNN LPF------------------------------------')
    # score_train = np.mean(sorted_LPF_CCNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_LPF_CCNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_LPF_CCNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('-------------CNN LPF------------------------------------')
    # score_train = np.mean(sorted_LPF_CNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_LPF_CNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_LPF_CNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('-------------GNN LPF------------------------------------')
    # score_train = np.mean(sorted_LPF_GNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_LPF_GNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_LPF_GNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    #
    # print('###############################################################')
    # print('-------------ANN audio------------------------------------')
    # score_train = np.mean(sorted_AUDIO_ANN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_AUDIO_ANN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_AUDIO_ANN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    #
    # print('-------------CCNN audio------------------------------------')
    # score_train = np.mean(sorted_AUDIO_CCNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_AUDIO_CCNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_AUDIO_CCNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('-------------GNN audio------------------------------------')
    # score_train = np.mean(sorted_AUDIO_GNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_AUDIO_GNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_AUDIO_GNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('###############################################################')
    # print('-------------ANN Correlation------------------------------------')
    # score_train = np.mean(sorted_CORRELATION_ANN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_CORRELATION_ANN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_CORRELATION_ANN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    #
    # print('-------------CCNN Correlation------------------------------------')
    # score_train = np.mean(sorted_CORRELATION_CCNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_CORRELATION_CCNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_CORRELATION_CCNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('-------------GNN Correlation------------------------------------')
    # score_train = np.mean(sorted_CORRELATION_GNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_CORRELATION_GNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_CORRELATION_GNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('###############################################################')
    #
    # print('-------------CCNN Spectrogram------------------------------------')
    # score_train = np.mean(sorted_SPECT_CCNN1_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_SPECT_CCNN1_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_SPECT_CCNN1_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # print('-------------GNN Spectrogram------------------------------------')
    # score_train = np.mean(sorted_SPECT_GNN_train[0])*100
    # print('Train accuracy:', score_train)
    #
    # score_dev = np.mean(sorted_SPECT_GNN_dev[0])*100
    # print('Dev accuracy:', score_dev)
    #
    # score_test = np.mean(sorted_SPECT_GNN_test[0])*100
    # print('Test accuracy:', score_test)
    # print('\n-------------------------------------------------')
    #
    # # print('-------------------------------------------------')
    # # score_train = np.mean(sorted_SPECT_GNN_train[0])
    # # print('\nTrain loss GNN spectrogram:', score_train**2)
    # # print('Train accuracy GNN spectrogram:', score_train)
    # #
    # # score_dev = np.mean(sorted_SPECT_GNN_dev[0])
    # # print('\nDev loss GNN spectrogram:', score_dev**2)
    # # print('Dev accuracy GNN spectrogram:', score_dev)
    # #
    # # score_test = np.mean(sorted_SPECT_GNN_test[0])
    # # print('\nTest loss GNN spectrogram:', score_test**2)
    # # print('Test accuracy GNN spectrogram:', score_test)
    # # print('\n-------------------------------------------------')
    #
    # # # # Evaluation Values
    # # print('________________ TEST SET ________________')
    # # print('mean, P50, P90, P95')
    # # print('MB: \n', lf.return_eval_values(sorted_model_based[0]))
    # # print('CCNN: \n', lf.return_eval_values(sorted_LPF_CCNN_test[0]))
    # # print('CNN: \n', lf.return_eval_values(sorted_LPF_CNN_test[0]))
    # # print('MLP one hot: \n', lf.return_eval_values(sorted_LPF_ANN_test[0]))
    # # print('MLP not one hot: \n', lf.return_eval_values(sorted_LPF_ANN_notonehot_test[0]))
    # # print('GNN: \n', lf.return_eval_values(sorted_LPF_GNN_test[0]))

