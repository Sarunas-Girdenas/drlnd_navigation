import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,
                 fc1_units=128, fc2_units=64, ddqn=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.ddqn = ddqn
        
        self.fc1 = nn.utils.weight_norm(nn.Linear(state_size, fc1_units), name='weight')
        nn.init.xavier_uniform_(self.fc1.weight)
        
        self.fc2 = nn.utils.weight_norm(nn.Linear(fc1_units, fc2_units), name='weight')
        nn.init.xavier_uniform_(self.fc2.weight)
        
        self.fc3 = nn.utils.weight_norm(nn.Linear(fc2_units, action_size), name='weight')
        nn.init.xavier_uniform_(self.fc3.weight)
        
        # value function
        self.value_function = nn.Linear(fc2_units, 1)
        
        # advantage function
        self.adv_function = nn.Linear(fc2_units, action_size)
        
        return None
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.ddqn:
            out = self.value_function(x) + (self.adv_function(x) - self.adv_function(x).mean())
        else:
            out = self.fc3(x)
        
        return out