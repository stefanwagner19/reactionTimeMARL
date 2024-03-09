import os
import argparse
import tqdm

import torch
import numpy as np

from env import Multiroller
from agent import actorCriticAgent
from replay_buffer import ReplayBuffer
from normalization import NoNormalization, RunningNormalization

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--epoch", type=int, required=True)
parser.add_argument("--run_for", type=int, default=1000)
parser.add_argument("--reaction_time", type=int, default=0)
parser.add_argument("--hidden_size", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--model_base_dir", type=str, default="models")
parser.add_argument("--port", type=int, default=5005)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--running_normalization", action="store_true")

try:
	buffer_len = 1000

	args = parser.parse_args()

	if args.running_normalization:
		normalization = RunningNormalization
	else:
		normalization = NoNormalization

	device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

	env = Multiroller(base_port=args.port)

	replay_buffer = ReplayBuffer(num_agents=env.get_num_agents(),
								 max_len=buffer_len,
								 num_actions=env.get_num_actions(),
								 obs_size=env.get_num_obs(),
								 hidden_size_actor=args.hidden_size,
								 hidden_size_critic=args.hidden_size,
								 num_layers_actor=args.num_layers,
								 num_layers_critic=args.num_layers,
								 reaction_time=args.reaction_time)

	agents = [actorCriticAgent(id=i,
							  input_size=env.get_num_obs(),
							  output_size=env.get_num_actions(),
							  hidden_size=args.hidden_size,
							  num_layers=args.num_layers,
							  normalization=normalization,
							  replay_buffer=replay_buffer,
							  device=device,
							  gamma=args.gamma,
							  lr=0.0001) for i in range(env.get_num_agents())]


	for i, agent in enumerate(agents):
		agent.load_agent(os.path.join(args.model_base_dir, args.exp_name), i, args.epoch)


	max_episode_length = buffer_len//2
	is_terminal = False

	for step in range(args.run_for):
		if step == 0 or is_terminal:
			
			episode_reward = 0
			reward_modifier = 1/args.gamma
			env.reset()
			curr_episode_length = 0
			h_actor = [None] * env.get_num_agents()
			c_actor = [None] * env.get_num_agents()
			h_critic = [None] * env.get_num_agents()
			c_critic = [None] * env.get_num_agents()
			delayed_obs = [[np.zeros(env.get_num_obs()) for j in range(args.reaction_time)] for i in range(env.get_num_agents())]
		
		obs = env.get_obs()
		actor_obs = np.zeros((env.get_num_agents(), env.get_num_obs()), dtype=np.float32)
		for i in range(obs.shape[0]):
			delayed_obs[i].append(obs[i])
			actor_obs[i] = delayed_obs[i].pop(0)

		actions = np.zeros((env.get_num_agents(), env.get_num_actions()), dtype=np.float32)

		# perform step for actor
		for i, agent in enumerate(agents):
			action, h_actor_new, c_actor_new = agent.chooseAction(actor_obs[i], h_actor[i], c_actor[i])
			actions[i, action] = 1
			h_actor[i] = h_actor_new
			c_actor[i] = c_actor_new

		discrete_actions = np.argmax(actions, axis=1)
		env.set_actions(discrete_actions)

		env.update()

		reward = env.get_reward()
		reward_modifier *= args.gamma
		episode_reward += reward_modifier * reward[0].squeeze()

		is_terminal = env.is_terminal() or (curr_episode_length >= max_episode_length)

		curr_episode_length += 1

	env.close()

except Exception as e:
	print(e)
	env.close()
