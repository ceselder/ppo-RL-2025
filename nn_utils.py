import torch.optim
from torch import nn
import numpy as np
import torch

class GenericNeuralNetwork(nn.Module):
    def __init__(self, params: dict = None):
        """

        Args:
            params:
        """
        super().__init__()

        if params is None:
            params = dict(input_size=2,
                          layers=[(64, 'relu', 0.0)],
                          output_size=2,
                          activation_final='tanh')

        self.parameter_dict = params
        self.training_data_size = None

        self.network = nn.Sequential(*make_network(network_params=params))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        predicted_next_state = self.network(x)  # [T_r_k+1, u_phys_k]
        return predicted_next_state

    @torch.no_grad()
    def predict(self, x):
        # x = torch.tensor(x, dtype=torch.float32)
        x = x.clone().detach()
        o_k_1 = self.forward(x)
        o_k_1 = (o_k_1.data.numpy())

        return o_k_1


def make_network(network_params: dict = None):
    """
    Args:
        network_params:

    Returns:
        network: nn.Module
    """

    if len(network_params['layers']) == 0:
        network = [fc_module([network_params['input_size'],
                              network_params['output_size']],
                             activation=network_params['activation_final'],
                             dropout_rate=0.0)]
    else:
        network = [fc_module([network_params['input_size'],
                              network_params['layers'][0][0]],
                             activation=network_params['layers'][0][1],
                             dropout_rate=network_params['layers'][0][2])]

        for l_i in range(len(network_params['layers'][:-1])):
            network += [fc_module([network_params['layers'][l_i][0],
                                   network_params['layers'][l_i + 1][0]],
                                  activation=network_params['layers'][l_i + 1][1],
                                  dropout_rate=network_params['layers'][l_i + 1][2])]

        network += [fc_module([network_params['layers'][-1][0],
                               network_params['output_size']],
                              activation=network_params['activation_final'],
                              dropout_rate=0.0)]
    return network


class fc_module(nn.Module):
    """
    Fully connected module of a neural network
    """

    def __init__(self, layer_params: list = None,
                 activation: str = 'tanh',
                 dropout_rate: float = 0.0):
        """

        Args:
            layer_params:
            activation:
            dropout_rate:
        """

        super(fc_module, self).__init__()

        if activation == 'linear':
            self.fc_module = nn.Sequential(
                nn.Linear(layer_params[0], layer_params[1]),
                nn.Dropout(p=dropout_rate)
            )
        else:
            if activation == 'tanh':
                activation_function = nn.Tanh

            elif activation == 'relu':
                activation_function = nn.ReLU

            elif activation == 'sigmoid':
                activation_function = nn.Sigmoid

            elif activation == 'softmax':
                activation_function = nn.Softmax

            else:
                raise ValueError(f"Unknown Activation function: {activation}")

            if activation != 'softmax':
                self.fc_module = nn.Sequential(
                    nn.Linear(layer_params[0], layer_params[1]),
                    activation_function(),
                    nn.Dropout(p=dropout_rate)
                )
            else:
                self.fc_module = nn.Sequential(
                    nn.Linear(layer_params[0], layer_params[1]),
                    activation_function(dim=-1),
                    nn.Dropout(p=dropout_rate)
                )

    def forward(self, x):
        return self.fc_module(x)