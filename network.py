import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

class ActorCriticFFNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorCriticFFNetwork, self).__init__()
        # Shared siamese layer
        self.h_s_flat = nn.Linear(8192, 512) 
        self.h_t_flat = nn.Linear(8192, 512)

        # Shared fusion layer
        self.h_fc2 = nn.Linear(1024, 512)

        self.h_fc3 = nn.Linear(512, 512)

        self.actor = nn.Linear(512, action_size)
        self.critic = nn.Linear(512, 1)
       
        self.train()

    def forward(self, observation, target):
        # Transform input to torch tensor
        s = torch.tensor(observation, dtype=torch.float)
        t = torch.tensor(target, dtype=torch.float)
        
        # Flatten input
        obs_flat = s.reshape(-1, 8192)
        target_flat = t.reshape(-1, 8192)
        
        # Siamese Layer
        h_s_flat = F.relu(self.h_s_flat(obs_flat))
        h_t_flat = F.relu(self.h_s_flat(target_flat))
        h_fc1 = torch.cat((h_s_flat, h_t_flat), 1) 
        
        # Shared fusion layer
        h_fc2 = F.relu(self.h_fc2(h_fc1))

        # Scene-specific adaption layer
        h_fc3 = F.relu(self.h_fc3(h_fc2))
        
        return self.actor(h_fc3), self.critic(h_fc3)
   
    '''
    def get_action_probs(self, observation, target):
        h_fc3 = self(observation, target)
        action_probs = F.softmax(self.actor(h_fc3), dim=1)
        return action_probs[0]
    
    def get_state_value(self, observation, target):
        h_fc3 = self(observation, target)
        state_value = self.critic(h_fc3)
        return state_value[0]
    
    def evaluate_actions(self, observation, target):
        h_fc3 = self(observation, target)
        action_probs = F.softmax(self.actor(h_fc3), dim=1)
        state_values = self.critic(h_fc3)
        return action_probs[0], state_values[0]
    '''
