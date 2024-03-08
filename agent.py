import os
import copy
import torch
import numpy as np
from torch import optim
from torch import nn
from replay_buffer import ReplayBuffer
from env import Multiroller
from model import Model

class actorCriticAgent:
    def __init__(self, id, input_size, output_size, hidden_size, num_layers, normalization, replay_buffer, device, gamma, lr):
        self.id = id
        self.actor = Model(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization, softmax=True)
        self.critic = Model(input_size=input_size + 2 * output_size, output_size=1, hidden_size=hidden_size, num_layers=num_layers, normalization = normalization)
        self.replay_buffer = replay_buffer
        self.device = device
        self.actor.to(self.device, dtype=torch.float32)
        self.critic.to(self.device, dtype=torch.float32)
        self.gamma = gamma
        self.lossFunc = nn.MSELoss()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)


    def save_agent(self, model_dir, id, epoch):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        try:
            copied_normalization_actor = copy.deepcopy(self.actor._normalization)
            copied_normalization_critic = copy.deepcopy(self.critic._normalization)
            copied_normalization_actor.to(torch.device("cpu"), dtype=torch.float32)
            copied_normalization_critic.to(torch.device("cpu"), dtype=torch.float32)
            torch.save({
                "model_state_dict": self.actor.state_dict(),
                "optimizer_state_dict": self.actor_optim.state_dict(),
                "normalization": copied_normalization_actor
                }, os.path.join(model_dir, f"{epoch}_{id}_actor"))
            torch.save({
                "model_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.critic_optim.state_dict(),
                "normalization": copied_normalization_critic
                }, os.path.join(model_dir, f"{epoch}_{id}_critic"))
            print("### Save successful")
        except:
            print("XXX Save failed")

    def load_agent(self, model_dir, id, epoch):
        try:
            checkpoint_actor = torch.load(os.path.join(model_dir, f"{epoch}_{id}_actor"))
            self.actor.load_state_dict(checkpoint_actor["model_state_dict"])
            self.actor_optim.load_state_dict(checkpoint_actor["optimizer_state_dict"])
            self.actor._normalization(checkpoint_actor["normalization"])
            self.actor.to(self.device, dtype=torch.float32)

            checkpoint_critic = torch.load(os.path.join(model_dir, f"{epoch}_{id}_critic"))
            self.critic.load_state_dict(checkpoint_critic["model_state_dict"])
            self.critic_optim.load_state_dict(checkpoint_critic["optimizer_state_dict"])
            self.critic._normalization(checkpoint_critic["normalization"])
            self.critic.to(self.device, dtype=torch.float32)
            print("### Load successful")
        except:
            print("XXX Load failed")


    def chooseAction(self, state, h=None, c=None):
        self.actor.eval()
        stateTensor = torch.tensor(state.reshape(1, 1, -1)).to(self.device, dtype=torch.float32)
        
        if h is None or c is None:
            lstm_state = None
        else:
            hTensor = torch.tensor(h.reshape(h.shape[0], 1, -1)).to(self.device, dtype=torch.float32)
            cTensor = torch.tensor(c.reshape(c.shape[0], 1, -1)).to(self.device, dtype=torch.float32)
            lstm_state = (hTensor, cTensor)
        output, h_new, c_new = self.actor(stateTensor, lstm_state)
        dist = torch.distributions.Categorical(output)
        action = dist.sample()

        action_np = action.detach().cpu().numpy().squeeze()
        h_np = h_new.detach().cpu().numpy().squeeze()
        c_np = c_new.detach().cpu().numpy().squeeze()

        self.actor.train()

        return (action_np, h_np, c_np)

    def evaluateStateAction(self, actions, state, h=None, c=None):
        self.critic.eval()
        actions = actions.reshape(-1)
        state_actions = np.concatenate((state, actions))
        state_actions_tensor = torch.tensor(state_actions.reshape(1, 1, -1)).to(self.device, dtype=torch.float32)
        if h is None or c is None:
                    lstm_state = None
        else:
            hTensor = torch.tensor(h.reshape(h.shape[0], 1, -1)).to(self.device, dtype=torch.float32)
            cTensor = torch.tensor(c.reshape(c.shape[0], 1, -1)).to(self.device, dtype=torch.float32)
            lstm_state = (hTensor, cTensor)
        value, h_new, c_new = self.critic(state_actions_tensor, lstm_state)
        
        value_np = value.detach().cpu().numpy().squeeze()
        h_np = h_new.detach().cpu().numpy().squeeze()
        c_np = c_new.detach().cpu().numpy().squeeze()

        self.critic.train()

        return (value_np, h_np, c_np)

    def train(self, batch_size):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        self.replay_buffer.initiate_batch(batch_size)
        batch_size = self.replay_buffer.get_batch_size()

        #check point
        actor_actions, actor_obs, h_actor, c_actor = self.replay_buffer.retrieve_actor_info(self.id)
        actions, next_actions, state, next_state, reward, isTerminal, h_critic, c_critic = self.replay_buffer.retrieve_critic_info(self.id)
        state_actions = np.concatenate((state, actions), axis=1)
        state_actions_tensor = torch.tensor(state_actions.reshape(batch_size, 1, -1)).to(self.device, dtype=torch.float32)
        h_critic_tensor = torch.tensor(h_critic).to(self.device, dtype=torch.float32).contiguous()
        c_critic_tensor = torch.tensor(c_critic).to(self.device, dtype=torch.float32).contiguous()
        value, h, c = self.critic(state_actions_tensor, (h_critic_tensor, c_critic_tensor))

        next_state_actions = np.concatenate((next_state, next_actions), axis=1)
        next_state_actions_tensor = torch.tensor(next_state_actions.reshape(batch_size, 1, -1)).to(self.device, dtype=torch.float32)
        next_value, h, c = self.critic(next_state_actions_tensor, (h, c))
        critic_target = torch.tensor(reward.reshape(-1, 1, 1)).to(self.device, dtype=torch.float32) + \
            self.gamma * next_value.detach() * torch.tensor(1-isTerminal.reshape(-1, 1, 1)).to(self.device, dtype=torch.float32)

        critic_loss = self.lossFunc(value, critic_target)
        critic_loss.backward()
        self.critic_optim.step()

        actor_obs_tensor = torch.tensor(actor_obs.reshape(batch_size, 1, -1)).to(self.device, dtype=torch.float32)
        h_actor_tensor = torch.tensor(h_actor).to(self.device, dtype=torch.float32).contiguous()
        c_actor_tensor = torch.tensor(c_actor).to(self.device, dtype=torch.float32).contiguous()

        actor_output, _, _ = self.actor(actor_obs_tensor, (h_actor_tensor, c_actor_tensor))

        # check point
        dist = torch.distributions.Categorical(actor_output)
        actor_actions_tensor = torch.tensor(actor_actions).to(self.device, dtype=torch.float32)
        logprob = dist.log_prob(actor_actions_tensor)

        actor_loss = torch.mean((-logprob) * (critic_target - value.detach()))
        actor_loss.backward()
        self.actor_optim.step()

        return actor_loss.item(), critic_loss.item()












        






        