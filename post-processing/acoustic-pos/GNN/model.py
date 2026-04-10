import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GNN_layer(nn.Module):
    def __init__(self, nr_anchors, input_feature_size_anchors, input_feature_size_user, input_feature_size_edges,
                 output_feature_size, output_layer=False):
        super().__init__()

        self.nr_anchors = nr_anchors
        self.output_layer = output_layer

        # set input feature sizes
        self.dl_anchors = input_feature_size_anchors
        self.dl_user = input_feature_size_user
        self.dl_edges = input_feature_size_edges

        # set output feature size
        self.dl_out = output_feature_size


        # define trainable weights & instantiate
        self.Wedge = nn.Parameter(torch.zeros(self.dl_edges, self.dl_out))
        nn.init.xavier_uniform_(self.Wedge)

        self.Wa = nn.Parameter(torch.zeros(self.dl_anchors, self.dl_out))
        nn.init.xavier_uniform_(self.Wa)

        self.Wu = nn.Parameter(torch.zeros(self.dl_user, self.dl_out))
        nn.init.xavier_uniform_(self.Wu)

        self.Wself_a = nn.Parameter(torch.zeros(self.dl_anchors, self.dl_out))
        nn.init.xavier_uniform_(self.Wself_a)

        self.Wneigh_a = nn.Parameter(torch.zeros(self.dl_out, self.dl_out))
        nn.init.xavier_uniform_(self.Wneigh_a)

        self.Wself_u = nn.Parameter(torch.zeros(self.dl_user, self.dl_out))
        nn.init.xavier_uniform_(self.Wself_u)

        self.Wneigh_u = nn.Parameter(torch.zeros(self.dl_out, self.dl_out))
        nn.init.xavier_uniform_(self.Wneigh_u)

    def forward(self, z_a, z_au, z_u):
        """
        :param z_a: bs x nr_anchors x input_feature_size
        :param z_au: bs x nr_anchors x input_feature_size
        :param z_u: bs x nr_user x input_feature_size

        in first layer: z_a = one_hot data, size: bs x nr_anchors x onehotlength (=20)
                        z_au = lpf_data, size: bs x nr_anchors x nr_samples
                        z_u = initial guess for position: size 3
                        (set to zeros now, can be prior knowledge later if we have an intial estimate)

        in output layer:
                        z_u = [\hat{x}, \hat{y}, \hat{z}]
                        we don't care about the other outputs
        :return:
        """

        """ Update edge features """
        # multiply with weight matrices
        Wezau = z_au @ self.Wedge # bs x nr_anchors x output_feature_size
        Waza = z_a @ self.Wa # bs x nr_anchors x output_feature_size
        Wuzu = z_u @ self.Wu # bs x nr_user x output_feature_size #todo now nr_user=1 if we want to generalize to
        #multiple users then we need to change this!!

        # expand dims to bs x nr_anchors x output_feature size for easy summation
        Wuzu_expanded = torch.tile(Wuzu, (1, self.nr_anchors, 1))
        #Wuzu_expanded = torch.repeat_interleave(Wuzu, repeats=self.nr_anchors, dim=1)


        # sum + relu
        z_au_updated = F.leaky_relu(Wezau + Waza + Wuzu_expanded) # bs x nr_anchors x output_feature_size

        """ Message passing """
        message_Nu = torch.mean(z_au_updated, dim=1, keepdim=True) # take mean over anchors => bs x 1 xoutput_feature_size

        """ Update anchor nodes """
        Wsaza = z_a @ self.Wself_a # bs x nr_anchors x output_feature_size
        Wnazau = z_au_updated @ self.Wneigh_a # bs x nr_anchors x output_feature_size
        z_a_updated = F.leaky_relu(Wsaza + Wnazau)

        """ Update user node """
        Wsuzu = z_u @ self.Wself_u # bs x nr_users x output_feature_size
        Wnumnu = message_Nu @ self.Wneigh_u # bs x nr_users x output_feature_size
        if self.output_layer: # no activation in output layer
            z_u_updated = Wsuzu + Wnumnu
        else:
            z_u_updated = F.leaky_relu(Wsuzu + Wnumnu)

        # todo for output layer no activation

        return z_a_updated, z_au_updated, z_u_updated

class GNNmodel(torch.nn.Module):
    def __init__(self, Nr_anchors, input_feature_size_anchors, input_feature_size_user, input_feature_size_edges,
                 nr_hidden_features, nr_hidden_layers):
        torch.nn.Module.__init__(self)
        self.nr_anchors = Nr_anchors
        self.nr_hidden_features = nr_hidden_features
        self.in_features_anchors = input_feature_size_anchors
        self.in_features_user = input_feature_size_user
        self.in_features_edges = input_feature_size_edges
        self.nr_hidden_layers = nr_hidden_layers
        self.output_size = 3 #x,y,z

        # define layers of GNN
        self.input_layer = GNN_layer(self.nr_anchors, self.in_features_anchors, self.in_features_user,
                                     self.in_features_edges, self.nr_hidden_features)

        self.hidden_layers = nn.ModuleList()
        for l in range(nr_hidden_layers):
            self.hidden_layers.append(GNN_layer(self.nr_anchors, self.nr_hidden_features, self.nr_hidden_features,
                                                self.nr_hidden_features, self.nr_hidden_features))
        self.output_layer = GNN_layer(self.nr_anchors, self.nr_hidden_features, self.nr_hidden_features,
                                                self.nr_hidden_features, self.output_size, output_layer=True)

    def forward(self, data, ids, init_pos):
        """
        :param data: lpf_data, size: bs x nr_anchors x nr_samples
        :param ids: one_hot data, size: bs x nr_anchors x onehotlength (=20)
        :param init_pos: initial guess for position: size 3
                        (set to zeros now, can be prior knowledge later if we have an intial estimate)
        :return: z_u (user node output features) position estimate (x,y,z) shape: bs x 3
        """

        # forward pass
        z_a, z_au, z_u = self.input_layer(ids, data, init_pos) # input layer
        for layer in self.hidden_layers: # hidden layers
            z_a, z_au, z_u = layer(z_a, z_au, z_u)
        z_a, z_au, z_u = self.output_layer(z_a, z_au, z_u) # output layer

        return z_u

class MLPmodel(torch.nn.Module):
    def __init__(self, nr_neurons):
        torch.nn.Module.__init__(self)
        self.nr_neurons = nr_neurons

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2520, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, 3)
        )

    def forward(self, data, ids, init_pos):
        """
        :param data: lpf_data, size: bs x nr_anchors x nr_samples
        :param ids: one_hot data, size: bs x nr_anchors x onehotlength (=20)
        """

        # flatten the input data
        data_flat = torch.flatten(data, start_dim=1)
        ids_flat = torch.flatten(ids, start_dim=1)
        input_concat = torch.cat((data_flat, ids_flat), dim=-1).float()

        # pass throug network
        output = self.layers(input_concat)

        return output




"""
# testing
bs = 2
nr_a = 6
nr_u = 1
in_size_a = 20
in_size_u = 3
in_size_edge = 400
nr_hidden_feat = 25
nr_h_layers = 4
GNN_model = GNNmodel(nr_a, in_size_a, in_size_u, in_size_edge, nr_hidden_feat, nr_h_layers)


#  generate dummy data
data = torch.randn((bs, nr_a, in_size_edge))
ids = torch.randn((bs, nr_a, in_size_a))
init_pos = torch.randn((bs, nr_u, in_size_u))

# test model
output = GNN_model(data, ids, init_pos)
print(output)
"""


