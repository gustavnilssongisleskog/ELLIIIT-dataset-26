import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import GNNmodel, MLPmodel
import json
from datetime import datetime
import tikzplotlib

with open('C:\\Users\\DaanDelabie\\Documents\\GitHub\\Acoustic-Simulator-full\\config.json') as json_file:
    config = json.load(json_file)

save_loc = config['save_loc']
nr_anchors = config['number_used_anchors_pos'][0]  # amount of anchors used (first element in list of config file!!)
n_speakers_simultaneous = config['n_speakers_simultaneous_in_simulation']    #amount of speakers DURING ONE position simulation
in_size_a = 20 # size of initial anchor features (20 for one hot)
out_counted = np.load(save_loc + 'Sim_data\\outcounted.npy')

path_ML_data = save_loc+'ML_Data\\Techtile_sim_data\\'
#TODO adjust if needed
save_model_path = save_loc+'ML_models\\Techtile_sim_data\\'


timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path="checkpoint.pt", verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time the validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model checkpoint.
            verbose (bool): If True, prints a message each time validation loss improves.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

if __name__=='__main__':
    # set GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if GPU is available
    #device = torch.device("cpu")  # check if GPU is available
    print(f'device: {device}')

    # todo set training params
    batch_size = 32
    lr = 10**-3
    nr_epochs = 1500
    nr_u = 1 # nr users
    in_size_u = 3 # size of initial user features
    nr_hidden_feat = 256 # nr features (512)
    nr_h_layers = 2 # nr hidden layers (8)
    early_stopping_patience = 150
    reduce_lr_patience = 100

    # load data
    # LOAD PREPROCESSED DATA IN RIGHT FORMAT
    data_batch_last = np.load(path_ML_data + "ML_dataset_sorted_anchor_nr_LPF_onehot_simulation.npy")

    in_size_edge = data_batch_last.shape[1] - in_size_a  # size of intial edge features
    print(f'{data_batch_last.shape}') #shape: (nr_a x nr_s x bs)
    data = np.moveaxis(data_batch_last, -1, 0) #switch first and last dimension new shape (bs x nr_a x nr_s)
    print(f'{data.shape}')
    one_hot_ids = data[:, :, 0:20]
    lpf_data = data[:, :, 20:]

    for i in range(6):
        plt.plot(lpf_data[0, i, :], label=f'anchor {i}')
    plt.legend()
    plt.show()

    # load labels
    labels = np.load(save_loc + 'Sim_data\\positions_mobile_node.npy')[:, 0:3]

    # calculate amount of positions integrated
    n_sets = np.size(labels, axis=0)  # amount of measured positions = amount of training examples

    print('\nTotal number of mobile node positions: ', n_sets)
    print('\nLabels/positions: \n', labels)
    print('\nLable size: ', labels.shape)
    print('\nData size: ', data.shape)

    #todo set correct split
    Nr_test = config['n_test_set_positions']
    Nr_train = config['n_train_set_positions']
    Nr_val = config['n_dev_set_positions']

    print('Test, Train, Dev #: ', Nr_test, Nr_train, Nr_val)


    #todo normalize the data for better training??? between 0, 1 => check plot!!

    # create datasets
    test_set = DaanData(lpf_data[0:Nr_test], one_hot_ids[0:Nr_test], labels[0:Nr_test], device=device)
    val_set = DaanData(lpf_data[Nr_test:Nr_test+Nr_val], one_hot_ids[Nr_test:Nr_test+Nr_val],
                       labels[Nr_test:Nr_test+Nr_val], device=device)
    train_set = DaanData(lpf_data[Nr_test+Nr_val:Nr_train+Nr_val+Nr_test],
                        one_hot_ids[Nr_test+Nr_val:Nr_train+Nr_val+Nr_test],
                        labels[Nr_test+Nr_val:Nr_train+Nr_val+Nr_test], device=device)

    # train_set = DaanData(lpf_data[0:Nr_train], one_hot_ids[0:Nr_train], labels[0:Nr_train], device=device)
    # val_set = DaanData(lpf_data[Nr_train:Nr_train+Nr_val], one_hot_ids[Nr_train:Nr_train+Nr_val],
    #                    labels[Nr_train:Nr_train+Nr_val], device=device)
    # test_set = DaanData(lpf_data[Nr_train+Nr_val:Nr_train+Nr_val+Nr_test],
    #                     one_hot_ids[Nr_train+Nr_val:Nr_train+Nr_val+Nr_test],
    #                     labels[Nr_train+Nr_val:Nr_train+Nr_val+Nr_test], device=device)

    # create data loaders
    training_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # get model
    model = GNNmodel(nr_anchors, in_size_a, in_size_u, in_size_edge, nr_hidden_feat, nr_h_layers).to(device)
    print(model)

    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # get loss function
    loss_fn = torch.nn.MSELoss()

    # Early stopping instance
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    # ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # Mode "min" because we're tracking validation loss (we want it to decrease)
        factor=0.1,  # Multiplicative factor to reduce the learning rate by (e.g., 0.1 reduces LR by 10x)
        patience=reduce_lr_patience,  # Number of epochs to wait before reducing the LR
        threshold=0.0001,  # Minimum change to qualify as an improvement
        min_lr=1e-8,  # Lower bound on the learning rate
        verbose=True  # Prints messages when the learning rate is reduced
    )

    # containers to store loss
    loss_history = []
    vloss_history = []
    running_loss = 0
    best_vloss = 100
    last_loss = 0
    for epoch in range(nr_epochs):
        with tqdm(training_dataloader, unit='batch') as tqdmbatch:
            for i, batch in enumerate(tqdmbatch):
                data, ids, labels = batch  # data: bs x Nr_a x Nr_s, ids: bs x Nr_a x 20, labels: bs x 3
                pos_init = torch.ones((batch_size, 1, 3)).to(device)  # bs x 1 x 3, zeros as initial input estimate for position
                #ids = torch.zeros((batch_size, 1, 20)).to(device)
                #todo could later be replaced with some sort of prior knowledge (e.g., last known position)

                # set accumulated grads to zero
                optimizer.zero_grad()

                # forward pass
                outputs = model(data, ids, pos_init)
                outputs = torch.squeeze(outputs)

                # compute loss
                loss = loss_fn(outputs, labels)

                # backprop
                loss.backward()

                # take gradient descent step
                optimizer.step()

                # gather data and report
                running_loss += loss.item()
                if i % 100 == 99:
                    last_loss = running_loss / 100

                    # log some values
                    tqdmbatch.set_postfix(loss=last_loss)
                    running_loss = 0

            # save training loss after each epoch
            loss_history.append(last_loss)

            # todo cumpute validation loss after each epoch
            # validation loss
            model.eval()
            with torch.no_grad():
                running_vloss = 0
                pos_init = torch.ones((batch_size, 1, 3)).to(device)
                for i, batch in enumerate(val_dataloader):
                    data_val, ids_val, labels_val = batch

                    # forward pass
                    outputs_val =  torch.squeeze(model(data_val, ids_val, pos_init))

                    # compute loss
                    vloss = loss_fn(outputs_val, labels_val)
                    running_vloss += vloss.item()

            # log the validation loss
            avg_vloss = running_vloss / (i + 1)
            print(f'\navg vallidation loss: {avg_vloss}')
            vloss_history.append(avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                path = os.path.join(save_model_path, 'modelGNN')
                torch.save(model.state_dict(), path)

            # Step the scheduler with the validation loss
            scheduler.step(avg_vloss)

            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

            # Early stopping check
            early_stopping(avg_vloss, model)

            # If early stopping triggered, exit training
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # print epoch nr
            print(f'epoch: {epoch}')

    print('Best val loss after training: ', best_vloss)

    # plot training loss
    plt.plot(loss_history, label='training loss')
    plt.plot(vloss_history, label='validation loss')
    plt.legend()

    curve_name = save_model_path+'learning_curve'
    plt.savefig(curve_name + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(curve_name + ".svg")
    tikzplotlib.save(curve_name + ".tex")
    plt.show()

    #After training test model and output values:
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

    #-------------------------------------------------------
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

    #-------------------------------------------------------
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
    print('\n-------------------------------------------------')