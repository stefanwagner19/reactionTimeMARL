import torch
import numpy as np
from replay_buffer import ReplayBuffer
from env import Multiroller
from model import Model

class actorCriticAgent:
    def __init__(self, input_size, output_size, hidden_size, num_layers, normalization):
        self.actor = Model(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization, softmax=True)
        self.critic = Model(input_size=input_size + 2 * output_size, output_size=1, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization)