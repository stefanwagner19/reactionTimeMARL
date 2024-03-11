import abc
from typing import Mapping
from typing_extensions import Self

import os
import platform

import numpy as np
import numpy.typing as npt

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class Env(abc.ABC):

    def __init__(self, n_agents: int, n_actions: int, n_obs: int) -> Self:
        self._n_agents = n_agents
        self._n_actions = n_actions
        self._n_obs = n_obs

    def get_num_agents(self) -> int:
        return self._n_agents

    def get_num_actions(self) -> int:
        return self._n_actions

    def get_num_obs(self) -> int:
        return self._n_obs

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def update(self) -> None:
        pass

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abc.abstractmethod
    def get_obs(self) -> npt.NDArray[np.float32]:
        pass

    @abc.abstractmethod
    def get_reward(self) -> float:
        pass

    @abc.abstractmethod
    def set_actions(self, actions: Mapping[int, Mapping[int, float]]) -> None:
        pass


class Multiroller(Env):

    def __init__(self, **kwargs: dict[str, any]) -> Self:
        super(Multiroller, self).__init__(n_agents=2, n_actions=4, n_obs=9)
        if platform.system() == "Linux":
            version = "multiroller_LIN"
        elif platform.system() == "Windows":
            version = "multiroller_WIN"
        else:
            version = "multiroller_MAC"
        self.__env_path = os.path.join("envs", "unity", "build", version, "multiroller")
        self.__env= UnityEnvironment(file_name=self.__env_path, **kwargs)
        self.__env.reset()
        self.__isTerminal = False
        self.__agents = list(self.__env.behavior_specs)
        decision_steps, _ = self.__env.get_steps(self.__agents[0])
        self.__obs_shape = np.squeeze(decision_steps.obs[0]).shape
        self.__reward_shape = np.squeeze(decision_steps.reward[0]).shape
        self.__obs = np.empty((2, *self.__obs_shape), dtype=np.float32)
        self.__reward = np.empty((2, *self.__reward_shape), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.__isTerminal = False
        self.__env.reset()
        self.__update_obs_and_reward()

    def close(self) -> None:
        self.__env.close()

    def update(self) -> None:
        self.__env.step()
        self.__update_obs_and_reward()

    def is_terminal(self) -> bool:
        return self.__isTerminal

    def get_obs(self) -> npt.NDArray[np.float32]:
        return self.__obs

    def get_reward(self) -> npt.NDArray[np.float32]:
        return self.__reward

    def set_actions(self, actions: npt.NDArray[np.int32]) -> None:
        for i, action in enumerate(actions):
            discrete_action = action.reshape(1, -1)
            action_tuple = ActionTuple()
            action_tuple.add_discrete(discrete_action)
            self.__env.set_actions(self.__agents[i], action_tuple)

    def __update_obs_and_reward(self) -> npt.NDArray[np.float32]:
        for i, agent in enumerate(self.__agents):
            decision_steps, terminal_steps = self.__env.get_steps(agent)
            if len(terminal_steps) > 0:
                steps = terminal_steps
                self.__isTerminal = True
            else:
                steps = decision_steps
            self.__obs[i] = steps.obs[0]
            self.__reward[i] = steps.reward[0]


if __name__ == "__main__":

    import random

    env = Multiroller()
    env.reset()

    for i in range(1000):
        print(i)
        if env.is_terminal():
            env.reset()

        obs = env.get_obs()
        reward = env.get_reward()

        actions = np.array([random.randint(0,3), random.randint(0,3)], dtype=np.int32)
        env.set_actions(actions)
        env.update()

    env.close()
    quit()

    # ##############################

    # env_path = os.path.join("envs", "unity", "build", "multiroller_LIN", "multiroller")

    # env = UnityEnvironment(file_name=env_path)

    # env.reset()

    # agents = list(env.behavior_specs)

    # # d, t = env.get_steps(agents[0])
    # # print(d)
    # # print(agents[0])
    # #
    # # env.close()
    # # quit()

    # while True:

    #     for i, agent in enumerate(agents):
    #         decision_steps, terminal_steps = env.get_steps(agent)

    #         if len(terminal_steps) > 0:
    #             print(f"\n#### Agent {i+1} ####")
    #             print(f"Obersvations: {terminal_steps.obs}")
    #             print(f"Reward: {terminal_steps.reward}")
    #             print(f"OtherObs: {decision_steps.obs}")
    #             env.close()
    #             quit()
    #         #
    #         else:
    #         #     print(f"\n#### Agent {i+1} ####")
    #         #     print(f"Obersvations: {decision_steps.obs}")
    #         #     print(f"Reward: {decision_steps.reward}")

    #             try:
    #                 discrete_action = np.array([[random.randint(0,3)]], dtype=np.int32)
    #                 action_tuple = ActionTuple(discrete_action)
    #                 # action_tuple.add_discrete(discrete_action)
    #                 env.set_actions(agent, action_tuple)
    #             except Exception as e:
    #                 print(e)
    #                 env.close()
    #                 quit()
    #     env.step()

    # print("exiting")
    # env.close()
