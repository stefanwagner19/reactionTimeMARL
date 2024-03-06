from typing_extensions import Self

import numpy as np
import numpy.typing as npt


class ReplayBuffer(object):

	def __init__(self, agents: list[str], max_len: int, num_actions: int, obs_size: int, hidden_size_actor: int, \
			hidden_size_critic: int, reaction_time: int = 0) -> Self:
		self.__agents = {agent: i for i, agent in enumerate(agents)}
		self.__n_agents = len(self.__agents)
		self.__max_len = max_len
		self.__num_actions = num_actions
		self.__obs_size = obs_size
		self.__reaction_time = reaction_time
		self.__counter = 0
		self.__episode_counter = 0
		self.__actions = np.zeros((self.__max_len, self.__n_agents, self.__num_actions), dtype=np.float)
		self.__state = np.zeros((self.__max_len, self.__n_agents, self.__obs_size), dtype=np.float)
		self.__next_state = np.zeros((self.__max_len, self.__n_agents, self.__obs_size), dtype=np.float)
		self.__rewards = np.zeros((self.__max_len, self.__n_agents), dtype=np.float)
		self.__isTerminal = np.zeros(self.__max_len, dtype=np.bool)
		self.__episode = np.zeros(self.__max_len, dtype=np.int)

		# network states saved
		self.__hidden_size_actor, self.__c_size_actor = hidden_size_actor
		self.__hidden_states_actor = np.zeros((self.__max_len, self.__n_agents, self.__hidden_size_actor), dtype=np.float)
		self.__c_states_actor = np.zeros((self.__max_len, self.__n_agents, self.__hidden_size_actor), dtype=np.float)
		self.__hidden_size_critic, self.__c_size_critic = hidden_size_critic
		self.__hidden_states_critic = np.zeros((self.__max_len, self.__n_agents, self.__hidden_size_critic), dtype=np.float)
		self.__c_states_critic = np.zeros((self.__max_len, self.__n_agents, self.__hidden_size_critic), dtype=np.float)

		# used for sampling
		self.__batch_indices = None
		self.__batch_size = 0
		

	def store(self, agent: str, action: int, state: npt.NDArray[np.float32], next_state: npt.NDArray[np.float32], \
			reward: float, isTerminal: bool, h_state_actor :npt.NDArray[np.float32], c_state_actor: npt.NDArray[np.float32], \
			h_state_critic: npt.NDArray[np.float32], c_state_critic: npt.NDArray[np.float32]) -> None:
		action_one_hot = np.zeros(self.__num_actions, dtype=np.float)
		action_one_hot[action] = 1
		self.__actions[self.__counter, self.__agents[agent]] = action_one_hot
		self.__state[self.__counter, self.__agents[agent]] = state
		self.__next_state[self.__counter, self.__agents[agent]] = next_state
		self.__rewards[self.__counter, self.__agents[agent]] = reward
		self.__isTerminal[self.__counter] = isTerminal
		self.__episode[self.__counter] = self.__episode_counter
		self.__hidden_states_actor[self.__counter, self.__agents[agent]] = h_state_actor
		self.__c_states_actor[self.__counter, self.__agents[agent]] = c_state_actor
		self.__hidden_states_critic[self.__counter, self.__agents[agent]] = h_state_critic
		self.__c_states_critic[self.__counter, self.__agents[agent]] = c_state_critic
		
	def increment_buffer(self) -> None:
		if self.__isTerminal[self.__counter]:
			self.__episode = (self.__episode + 1) % self.__max_len
		self.__counter = (self.__counter + 1) % self.__max_len

	def initiate_batch(self, batch_size: int) -> None:
		self.__batch_size = batch_size
		self.__batch_indices = np.random.choice(np.arange(self.__max_len), self.__batch_size)

	def retrieve_actor(self, agent: str) -> tuple[npt.NDArray]:
		obs = self.__state[self.__agents[agent], self.__batch_indices - self.__reaction_time]
		different_episodes = self.__episode[self.__batch_indices] == self.__episode[self.__batch_indices - self.__reaction_time]
		obs[different_episodes] = 0
		hidden_state = self.__hidden_states_actor[self.__agents[agent], self.__batch_indices]
		c_state = self.__c_states_actor[self.__agents[agent], self.__batch_indices]
		return (obs, hidden_state, c_state)

	def retrieve_critic(self, agent: str) -> tuple[npt.NDArray]:
		actions = self.__actions[:, self.__batch_indices].reshape(self.__batch_size, -1)
		state = self.__state[self.__agents[agent], self.__batch_indices].reshape(self.__batch_size, -1)
		next_state = self.__next_state[self.__agents[agent], self.__batch_indices].reshape(self.__batch_size, -1)
		reward = self.__rewards[self.__agents[agent], self.__batch_indices]
		isTerminal = self.__isTerminal[self.__batch_indices]
		hidden_state = self.__hidden_states_critic[self.__agents[agent], self.__batch_indices]
		c_state = self.__c_states_critic[self.__agents[agent], self.__batch_indices]
		return (actions, state, next_state, reward, isTerminal, hidden_state, c_state)
		