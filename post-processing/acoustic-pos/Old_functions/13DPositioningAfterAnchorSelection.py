import numpy as np
import localFunctions as lf
from scipy.signal import *
import librosa as lbr
import json
from tqdm import tqdm

with open('config.json') as json_file:
    config = json.load(json_file)

save_loc = config['save_loc']
n_anch_positions = config['number_used_anchors_pos'][0]
anchor_selection_tech = config['anchor_selection_tech']

if anchor_selection_tech=="All":
    print('Select All anchors')
    extra_save_path = save_loc + 'anchor_selection_results\\All\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
elif anchor_selection_tech== "RMS":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on biggest RMS values')
    extra_save_path = save_loc+'anchor_selection_results\\RMS\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'
elif anchor_selection_tech=="Max_CorrPeak_height":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on max correlation peak height')
    extra_save_path = save_loc + 'anchor_selection_results\\Max_CorrPeak_height\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'
elif anchor_selection_tech== "Picked_CorrPeak_height":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on picked correlation peak height')
    extra_save_path = save_loc + 'anchor_selection_results\\Picked_CorrPeak_height\\' + str(
        config['number_used_anchors_pos'][0]) + 'anchors\\'
elif anchor_selection_tech== "PP_value":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based PP values')
    extra_save_path = save_loc + 'anchor_selection_results\\PP_value\\'+str(config['number_used_anchors_pos'][0])+'anchors\\'
elif anchor_selection_tech== "N_sig_peaks":
    print('Select ' + str(config['number_used_anchors_pos'][0]) + ' anchors based on amount of significant peaks')
    extra_save_path = save_loc + 'anchor_selection_results\\N_sig_peaks\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
elif anchor_selection_tech== "RMS_diff_dist":
    extra_save_path = save_loc + 'anchor_selection_results\\RMS_diff_dist\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
elif anchor_selection_tech== "Sincsplit":
    extra_save_path = save_loc + 'anchor_selection_results\\Sincsplit\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
elif anchor_selection_tech== "Sincsplit_dist":
    extra_save_path = save_loc + 'anchor_selection_results\\Sincsplit_dist\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
elif anchor_selection_tech == 'STMR': # side lobe to main lobe ration (STMR), max2nd peak / max1st peak --> hoe kleiner hoe beter
    extra_save_path = save_loc + 'anchor_selection_results\\STMR\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
elif anchor_selection_tech== "ML":
    extra_save_path = save_loc + 'anchor_selection_results\\ML\\' + str(config['number_used_anchors_pos'][0]) + 'anchors\\'
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
#
# Base only on test set to compare: filter out test set
out_counted = np.load(save_loc+'Sim_data\\outcounted.npy')
# Calculate amount of test set grid points (for non-shoebox not n_x_test * n_y_test * n_z_test)
# n_test_set_positions = (config['n_x_test'] * config['n_y_test'] * config['n_z_test'])-out_counted
# mn_loc_test_set = mn_loc_all[:n_test_set_positions, :]

n_test_set_positions = config['n_test_set_positions']
mn_loc_test_set = mn_loc_all[:n_test_set_positions, :]
print('Amount of test set mobile node positions: {}'.format(n_test_set_positions))

d_meas_matrix = np.load(path_output_data+'d_meas_matrix.npy')
anchor_selection_matrix = np.load(extra_save_path+'anchor_selection_matrix.npy')
################################################
# ------------- 3D positioning -----------------
################################################
estimation_data_all = np.array([])
first = True
for mn_position in tqdm(range(0, n_test_set_positions)):  #mn_position = mobile node position = row in d_meas_matrix
    # Determine 3D position
    d_meas_one_position = d_meas_matrix[mn_position, anchor_selection_matrix[mn_position, :]]
    #pos_estimate = lf.simple_inter_xyz(anch_x_coords, anch_y_coords, anch_z_coords, d_meas_one_position)
    x0 = np.array([1, 1, 1])
    anchor_locs_selected = anchor_loc_all[anchor_selection_matrix[mn_position,:],:]

    if first:
        selected_anchor_locs = anchor_locs_selected
        first = False
    else:
        selected_anchor_locs = np.dstack((selected_anchor_locs, anchor_locs_selected))


    pos_estimate = lf.LS_positioning(anchor_locs_selected, d_meas_one_position, x0)

    # Get real position
    real_pos = mn_loc_test_set[mn_position, :]

    # Determine euclidean distance error
    euclidean_dist_error = np.linalg.norm(real_pos - pos_estimate)

    # Eucl. distance 2D
    euclidian_dist_2D = np.linalg.norm(real_pos[0:2] - pos_estimate[0:2])

    # Coördinate errors
    x_coord_diff = real_pos[0] - pos_estimate[0]
    y_coord_diff = real_pos[1] - pos_estimate[1]
    z_coord_diff = real_pos[2] - pos_estimate[2]

    # safe all distance error values in array [mn_position_nr, n_anchors (used for positioning),
    # mn_loc (real theoretical location), pos_estimate (estimated location), euclidean_distance_error, x_error, y_errror, z_error]
    estimation_data = dict({'mn_position_nr': mn_position,              # number to index the mobile node position
                            'n_anchors': n_anch_positions,              # amount of anchors used to estimate position
                            'mn_loc': real_pos,                         # real/theoretical position
                            'pos_estimate': pos_estimate,               # estimated position
                            'eucl_dist_error': euclidean_dist_error,    # euclidean distance error of 3D position
                            'eucl_dist_error2D': euclidian_dist_2D,
                            'x_error': x_coord_diff,                    # x error
                            'y_error': y_coord_diff,                    # y error
                            'z_error': z_coord_diff})                   # z error

    estimation_data_all = np.concatenate((estimation_data_all, np.array([estimation_data])))


# save date for analysis
np.save(extra_save_path+'selected_anchor_locs.npy', selected_anchor_locs)
np.save(extra_save_path+'estimation_data_dicts.npy', estimation_data_all)