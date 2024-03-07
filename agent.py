import torch
import numpy as np
from replay_buffer import ReplayBuffer
from env import Multiroller
from model import Model

class actorCriticAgent:
    def __init__(self, input_size, output_size, hidden_size, num_layers, normalization, replay_buffer, device):
        self.actor = Model(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization, softmax=True)
        self.critic = Model(input_size=input_size + 2 * output_size, output_size=1, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization)
        self.replay_buffer = replay_buffer
        self.device = device
        self.actor.to(self.device)

    def chooseAction(self, state, h=None, c=None):
        stateTensor = torch.tensor(state).to(self.device)
        self.actor(stateTensor)
        
        if h is None or c is None:
            lstm_state = None
        else:
            lstm_state = (h,c)
        output, (h,c) = self.actor(stateTensor, lstm_state)
        return torch.argmax(output), (h,c)
    
    def store(self):
        pass

    def train(self):
        pass






        