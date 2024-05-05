import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.fc2_adv = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2_val = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3_adv = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc3_val = nn.Linear(self.fc2_dims, 1)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.dropout = nn.Dropout(0.25)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)

        x = F.relu6(self.bn1(self.fc1(state)))
        x = self.dropout(x)

        adv = F.relu6(self.fc2_adv(x))
        val = F.relu6(self.fc2_val(x))
        adv = F.relu6(self.fc3_adv(adv))
        val = F.relu6(self.fc3_val(val))

        adv = self.dropout(adv)
        
        adv_mean = adv.mean(dim=1, keepdim=True)
        actions = self.fc4(val + adv - adv_mean)

        return actions
