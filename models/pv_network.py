import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class PolicyValueNet(nn.Module):
    '''
    A neural network that outputs both policy logits and a value estimate.
    The network has a shared body and two heads: one for policy and one for value.
    The policy head outputs logits for 12 independent action coefficients, each with n_actions possible actions.
    The value head outputs a single scalar value estimate.
    '''

    # input_dim: dimension of the flattened input state
    # hidden_dim: dimension of the hidden layers in the shared body ( number of neurons in each hidden layer )
    def __init__(self, input_dim=config.INPUT_DIM,
                 hidden_dim=config.HIDDEN_DIM,
                 n_heads=config.N_HEADS,
                 n_actions=3): # n_actions for u, v, w (-1, 0, 1)
        super(PolicyValueNet, self).__init__()

        # --- shared body ---
        # nn.Linear : applies a linear transformation to the incoming data: y = xA^T + b
        # three fully connected layers with ReLU activations
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # --- policy head ---
        # outputs logits for each of the 12 coefficients independently
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_actions) for _ in range(n_heads)
        ])

        # --- value head ---
        # outputs a single scalar value
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x is the flattened tensor state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        body_out = F.relu(self.fc3(x))

        # policy : list of 12 tensors, each of shape (batch_size, n_actions)
        policy_logits = [head(body_out) for head in self.policy_heads]

        # value : scalar tensor of shape (batch_size, 1) - tanhed to be in [-1, 1]
        value = torch.tanh(self.value_head(body_out))

        return policy_logits, value