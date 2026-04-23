import numpy as np
import matplotlib.pyplot as plt
import localFunctions as lf
from scipy.signal import *
import json
from datetime import datetime
import tikzplotlib
import os
from tqdm import tqdm
import torch
from model import GNNmodel, MLPmodel

# ------------------------------------------------
#           Configurations
# ------------------------------------------------
with open('C:\\Users\\DaanDelabie\\Documents\\GitHub\\Acoustic-Simulator-full\\config.json') as json_file:
    config = json.load(json_file)

save_loc = config['save_loc']
n_anchors = config['number_used_anchors_pos'][0]  # amount of anchors used (first element in list of config file!!)
n_speakers_simultaneous = config['n_speakers_simultaneous_in_simulation']    #amount of speakers DURING ONE position simulation
n_onehots = 20
out_counted = np.load(save_loc+'Sim_data\\outcounted.npy')

path_ML_data_sim = save_loc+'ML_Data\\Techtile_sim_data\\'

class DaanData(torch.utils.data.Dataset):
    def __init__(self, lpf_data, one_hot_data, labels, device='cuda'):
        """
        :param lpf_data: (bs x nr_a x nr_s)
        :param one_hot_data: (bs x nr_a x 20)
        :params labels: (bs x 3)
        """
        self.lpf_data = torch.from_numpy(lpf_data).to(torch.float32).to(device)
        self.one_hot_data = torch.from_numpy(one_hot_data).to(torch.float32).to(device)
        self.labels = torch.from_numpy(labels).to(torch.float32).to(device)


    def __len__(self):
        return self.lpf_data.shape[0]

    def __getitem__(self, idx):
        data = self.lpf_data[idx, :, :]
        ids = self.one_hot_data[idx, :, :]
        labels = self.labels[idx, :]
        return data, ids, labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if GPU is available

#TODO -----------------  ADJUST FOR NOT LPF ----------------------------------------------------------------------
input_feature_type = 'LPF' #'LPF', 'Audio', 'Correlation', 'Spectrogram'
path_model = save_loc+'ML_models\\Techtile_sim_data\\modelGNN_sim'

# LOAD PREPROCESSED DATA IN RIGHT FORMAT
data_batch_last = np.load(path_ML_data_sim+"ML_dataset_sorted_anchor_nr_LPF_onehot_simulation.npy")

# load labels
labels = np.load(save_loc + 'Sim_data\\positions_mobile_node.npy')[:, 0:3]

in_size_a = 20 # size of initial anchor features (20 for one hot)
in_size_edge = data_batch_last.shape[1] - in_size_a  # size of intial edge features
data = np.moveaxis(data_batch_last, -1, 0)  # switch first and last dimension new shape (bs x nr_a x nr_s)
print(f'{data.shape}')
one_hot_ids = data[:, :, 0:20]
lpf_data = data[:, :, 20:]

# create datasets
# calculate amount of positions integrated
Nr_test = config['n_test_set_positions']
Nr_train = config['n_train_set_positions']
Nr_val = config['n_dev_set_positions']


print('Test, Train, Dev #: ', Nr_test, Nr_train, Nr_val)

test_set = DaanData(lpf_data[0:Nr_test], one_hot_ids[0:Nr_test], labels[0:Nr_test], device=device)
val_set = DaanData(lpf_data[Nr_test:Nr_test + Nr_val], one_hot_ids[Nr_test:Nr_test + Nr_val],
                   labels[Nr_test:Nr_test + Nr_val], device=device)
train_set = DaanData(lpf_data[Nr_test + Nr_val:Nr_train + Nr_val + Nr_test],
                     one_hot_ids[Nr_test + Nr_val:Nr_train + Nr_val + Nr_test],
                     labels[Nr_test + Nr_val:Nr_train + Nr_val + Nr_test], device=device)

#TODO ------------------------------------------------------------------------------------------------------------

save_path = save_loc+'anchor_selection_results_GNN\\RMS\\6anchors\\'
save_model_path = save_loc+'ML_models\\Techtile_sim_data\\'


def get_model(location):
    model = torch.load(location, weights_only=False)

    return model

def generate_pos_est_information_dict(real_pos, predicted_pos, n_anchors):
    """
    Generate a dictionarry with all necessary data
    :param real_pos: real positions
    :param predicted_pos: estimated positions
    :param n_anchors: amount of used anchors
    :return:
    """
    # Create dict with estimation information
    estimation_data_all = np.array([])
    for pos_nr in range(0, np.size(real_pos, axis=0)):
        real_loc = real_pos[pos_nr,:]
        estimate_loc = predicted_pos[pos_nr, :]
        # Euclidian distance 3D
        euclidian_dist = np.linalg.norm(real_loc - estimate_loc)
        # Eucl. distance 2D
        euclidian_dist_2D = np.linalg.norm(real_loc[0:2]-estimate_loc[0:2])

        # Error x y and z position per position
        x_coord_diff = real_loc[0]-estimate_loc[0]
        y_coord_diff = real_loc[1] - estimate_loc[1]
        z_coord_diff = real_loc[2] - estimate_loc[2]

        estimation_data = dict({'mn_position_nr': pos_nr,
                                'n_anchors': n_anchors,
                                'mn_loc': real_loc,
                                'pos_estimate': estimate_loc,
                                'eucl_dist_error': euclidian_dist,
                                'eucl_dist_error2D': euclidian_dist_2D,
                                'x_error': x_coord_diff,
                                'y_error': y_coord_diff,
                                'z_error': z_coord_diff})

        estimation_data_all = np.concatenate((estimation_data_all, np.array([estimation_data])))

    return estimation_data_all


if __name__ == '__main__':
    nr_anchors = config['number_used_anchors_pos'][0]  # amount of anchors used (first element in list of config file!!)

    batch_size = 1
    lr = 10**-3
    nr_epochs = 10
    nr_u = 1 # nr users
    in_size_u = 3 # size of initial user features
    nr_hidden_feat = 256 # nr features (512)
    nr_h_layers = 2 # nr hidden layers (8)
    model = GNNmodel(nr_anchors, in_size_a, in_size_u, in_size_edge, nr_hidden_feat, nr_h_layers).to(device)
    model.load_state_dict(torch.load(path_model))
    model.eval()

    # get loss function
    loss_fn = torch.nn.MSELoss()
    #-------------------------------------------------
    #              Evaluation
    #-------------------------------------------------
    print('-------------------------------------------------')
    train_data_val, train_ids_val, train_labels_val = train_set.lpf_data, train_set.one_hot_data, train_set.labels
    pos_init = torch.ones((train_labels_val.shape[0], 1, 3)).to(device)

    # forward pass
    outputs_train_val = model(train_data_val, train_ids_val, pos_init)
    outputs_train_val_smaldim = torch.squeeze(outputs_train_val)

    # compute loss
    loss_train = loss_fn(outputs_train_val_smaldim, train_labels_val)
    acc_train = torch.sqrt(loss_train)

    print('\nTrain loss:', loss_train.item())
    print('Train accuracy:', acc_train.item())

    # -------------------------------------------------------
    dev_data, dev_ids, dev_labels = val_set.lpf_data, val_set.one_hot_data, val_set.labels
    pos_init_dev = torch.ones((dev_labels.shape[0], 1, 3)).to(device)

    # forward pass
    outputs_dev_val = model(dev_data, dev_ids, pos_init_dev)
    outputs_dev_val_smaldim = torch.squeeze(outputs_dev_val)

    # compute loss
    loss_dev = loss_fn(outputs_dev_val_smaldim, dev_labels)
    acc_dev = torch.sqrt(loss_dev)

    print('\nDev loss:', loss_dev.item())
    print('Dev accuracy:', acc_dev.item())

    # -------------------------------------------------------
    test_data, test_ids, test_labels = test_set.lpf_data, test_set.one_hot_data, test_set.labels
    pos_init_test = torch.ones((test_labels.shape[0], 1, 3)).to(device)

    # forward pass
    outputs_test_val = model(test_data, test_ids, pos_init_test)
    outputs_test_val_smaldim = torch.squeeze(outputs_test_val)

    # compute loss
    loss_test = loss_fn(outputs_test_val_smaldim, test_labels)
    acc_test = torch.sqrt(loss_test)

    print('\nTest loss:', loss_test.item())
    print('Test accuracy:', acc_test.item())

    estimation_data_all_train = generate_pos_est_information_dict(train_labels_val.cpu().detach().numpy(), outputs_train_val_smaldim.cpu().detach().numpy(), n_anchors)
    np.save(save_path + 'LPF_GNN_train.npy', estimation_data_all_train)
    estimation_data_all_devset = generate_pos_est_information_dict(dev_labels.cpu().detach().numpy(), outputs_dev_val_smaldim.cpu().detach().numpy(), n_anchors)
    np.save(save_path + 'LPF_GNN_dev.npy', estimation_data_all_devset)
    estimation_data_all_testset = generate_pos_est_information_dict(test_labels.cpu().detach().numpy(), outputs_test_val_smaldim.cpu().detach().numpy(), n_anchors)
    np.save(save_path + 'LPF_GNN_test.npy', estimation_data_all_testset)





