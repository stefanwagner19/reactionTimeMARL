from typing import Union
from typing_extensions import Self

import numpy as np
import numpy.typing as npt


class ReplayBuffer(object):

	def __init__(self, num_agents: int, max_len: int, num_actions: int, obs_size: int, hidden_size_actor: int, \
			hidden_size_critic: int, num_layers_actor: int, num_layers_critic: int, reaction_time: int = 0) -> Self:
		self.__n_agents = num_agents
		self.__max_len = max_len
		self.__num_actions = num_actions
		self.__obs_size = obs_size
		self.__reaction_time = reaction_time
		self.__counter = 0
		self.__total_counter = 0
		self.__episode_counter = 0
		self.__actions = np.zeros((self.__max_len, self.__n_agents, self.__num_actions), dtype=np.float)
		self.__state = np.zeros((self.__max_len, self.__n_agents, self.__obs_size), dtype=np.float)
		self.__rewards = np.zeros((self.__max_len, self.__n_agents), dtype=np.float)
		self.__isTerminal = np.zeros(self.__max_len, dtype=np.bool)
		self.__episode = np.zeros(self.__max_len, dtype=np.int)

		# network states saved
		self.__hidden_states_actor_shape = (num_layers_actor, hidden_size_actor)
		self.__c_states_actor_shape = (num_layers_actor, hidden_size_actor)
		
		self.__hidden_states_actor = np.zeros((num_layers_actor, self.__max_len, self.__n_agents, hidden_size_actor), dtype=np.float)
		self.__c_states_actor = np.zeros((num_layers_actor, self.__max_len, self.__n_agents, hidden_size_actor), dtype=np.float)

		self.__hidden_states_critic_shape = (num_layers_critic, hidden_size_critic)
		self.__c_states_critic_shape = (num_layers_critic, hidden_size_critic)

		self.__hidden_states_critic = np.zeros((num_layers_critic, self.__max_len, self.__n_agents, hidden_size_critic), dtype=np.float)
		self.__c_states_critic = np.zeros((num_layers_critic, self.__max_len, self.__n_agents, hidden_size_critic), dtype=np.float)

		# used for sampling
		self.__batch_indices = None
		self.__batch_size = 0

	def get_batch_size(self) -> int:
		return self.__batch_size

	def store_actor_info(self, agent: int, action: npt.NDArray[np.float32], obs: npt.NDArray[np.float32], \
			hidden_state: Union[npt.NDArray[np.float32], None], c_state: Union[npt.NDArray[np.float32], None]) -> None:
		self.__actions[self.__counter, agent] = action
		self.__state[self.__counter, agent] = obs
		self.__hidden_states_actor[:, self.__counter, agent] = hidden_state if hidden_state is not None \
			else np.zeros(self.__hidden_states_actor_shape, dtype=np.float32)
		self.__c_states_actor[:, self.__counter, agent] = c_state if c_state is not None \
			else np.zeros(self.__c_states_actor_shape, dtype=np.float32)

	def store_critic_info(self, agent: int, reward: npt.NDArray[np.float32], isTerminal: bool, \
			hidden_state: Union[npt.NDArray[np.float32], None], c_state: Union[npt.NDArray[np.float32], None]) -> None:
		self.__rewards[self.__counter, agent] = reward
		self.__isTerminal[self.__counter] = isTerminal
		self.__hidden_states_critic[:, self.__counter, agent] = hidden_state if hidden_state is not None \
			else np.zeros(self.__hidden_states_critic_shape, dtype=np.float32)
		self.__c_states_critic[:, self.__counter, agent] = c_state if c_state is not None \
			else np.zeros(self.__c_states_critic_shape, dtype=np.float32)

	# def store(self, agent: int, action: npt.NDArray[np.float32], state: npt.NDArray[np.float32], next_state: npt.NDArray[np.float32], \
	# 		reward: float, isTerminal: bool, h_state_actor :npt.NDArray[np.float32], c_state_actor: npt.NDArray[np.float32], \
	# 		h_state_critic: npt.NDArray[np.float32], c_state_critic: npt.NDArray[np.float32]) -> None:
	# 	self.__actions[self.__counter, agent] = action
	# 	self.__state[self.__counter, agent] = state
	# 	self.__next_state[self.__counter, agent] = next_state
	# 	self.__rewards[self.__counter, agent] = reward
	# 	self.__isTerminal[self.__counter] = isTerminal
	# 	self.__episode[self.__counter] = self.__episode_counter
	# 	self.__hidden_states_actor[self.__counter, agent] = h_state_actor
	# 	self.__c_states_actor[self.__counter, agent] = c_state_actor
	# 	self.__hidden_states_critic[self.__counter, agent] = h_state_critic
	# 	self.__c_states_critic[self.__counter, agent] = c_state_critic
		
	def increment_buffer(self) -> None:
		if self.__isTerminal[self.__counter]:
			self.__episode = (self.__episode + 1) % self.__max_len
		self.__counter = (self.__counter + 1) % self.__max_len
		self.__total_counter += 1

	def initiate_batch(self, batch_size: int) -> None:
		self.__batch_size = min(batch_size, self.__total_counter - 1)
		if self.__max_len > self.__total_counter - 1:
			potential_indices = np.arange(self.__total_counter - 1)
		else:
			potential_indices = np.concatenate((np.arange(0, self.__counter - 1), np.arange(self.__counter, self.__max_len))) 
		self.__batch_indices = np.random.choice(potential_indices, self.__batch_size)

	def retrieve_actor_info(self, agent: int) -> tuple[npt.NDArray]:
		actions = self.__actions[self.__batch_indices, agent]
		obs = self.__state[self.__batch_indices - self.__reaction_time, agent]
		different_episodes = self.__episode[self.__batch_indices] != self.__episode[self.__batch_indices - self.__reaction_time]
		obs[different_episodes] = 0
		hidden_state = self.__hidden_states_actor[:,  self.__batch_indices, agent]
		c_state = self.__c_states_actor[:, self.__batch_indices, agent]
		return (actions, obs, hidden_state, c_state)

	def retrieve_critic_info(self, agent: int) -> tuple[npt.NDArray]:
		actions = self.__actions[self.__batch_indices, :].reshape(self.__batch_size, -1)
		next_actions = self.__actions[(self.__batch_indices + 1)%self.__max_len, :].reshape(self.__batch_size, -1)
		state = self.__state[self.__batch_indices, agent].reshape(self.__batch_size, -1)
		next_state = self.__state[(self.__batch_indices + 1)%self.__max_len, agent].reshape(self.__batch_size, -1)
		reward = self.__rewards[self.__batch_indices, agent]
		isTerminal = self.__isTerminal[self.__batch_indices]
		hidden_state = self.__hidden_states_critic[:,  self.__batch_indices, agent]
		c_state = self.__c_states_critic[:,  self.__batch_indices, agent]
		return (actions, next_actions, state, next_state, reward, isTerminal, hidden_state, c_state)
		