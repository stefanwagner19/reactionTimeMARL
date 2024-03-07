import torch
import numpy as np
from torch import optim
from torch import nn
from replay_buffer import ReplayBuffer
from env import Multiroller
from model import Model

class actorCriticAgent:
    def __init__(self, input_size, output_size, hidden_size, num_layers, normalization, replay_buffer, device, gamma):
        self.actor = Model(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization, softmax=True)
        self.critic = Model(input_size=input_size + 2 * output_size, output_size=1, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization)
        self.replay_buffer = replay_buffer
        self.device = device
        self.actor.to(self.device)
        self.gamma = gamma
        self.lossFunc = nn.MSEloss()
        

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

    def train(self, batch_size):
        self.replay_buffer.initiate_batch(batch_size)
        actor_obs, h_actor, c_actor = self.replay_buffer.retrieve_actor(batch_size)
        actions, next_actions, state, next_state, reward, isTerminal, h_critic, c_critic = self.replay_buffer.retrieve_critic(batch_size)
        state_actions = np.stack(state, actions, axis =1)
        state_actions_tensor = torch.tensor(state_actions).to(self.device)
        h_critic_tensor = torch.tensor(h_critic).to(self.device)
        c_critic_tensor = torch.tensor(c_critic).to(self.device)
        value, (h,c) = self.critic(state_actions, tuple[h_critic_tensor, c_critic_tensor])

        next_state_actions = np.stack(next_state, next_actions, axis =1)
        next_state_actions_tensor = torch.tensor(next_state_actions).to(self.device)
        next_value, (h,c) = self.critic(state_actions, tuple[h, c])
        critic_target = reward + self.gamma * next_value * isTerminal






        






        