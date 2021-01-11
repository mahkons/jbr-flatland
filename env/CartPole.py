import torch
import gym
from collections import defaultdict

class CartPole():
    def __init__(self, random_state):
        self.env = gym.make("CartPole-v1")
        self.env.seed(random_state)
        self.action_sz = self.env.action_space.n
        self.state_sz = self.env.observation_space.shape[0]

        self.n_agents = 1
        self.steps = 0
        self.total_reward = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action[0]))
        real_reward = reward
        self.total_reward += real_reward
        self.steps += 1

        return {0: obs}, {0:reward}, {0:done, '__all__':done}, info, real_reward

    def reset(self):
        self.steps, self.total_reward = 0, 0
        return {0: self.env.reset()}

    def render(self):
        self.env.render()

    def get_steps(self):
        return self.steps

    def get_total_reward(self):
        return self.total_reward

    def transform_action(self, handle, action):
        return action

    def greedy_action(self, handle):
        return None


# just several independent CartPoles
class MultiCartPole():
    def __init__(self, n_agent, random_state):
        self.envs = dict()
        for i in range(n_agent):
            self.envs[i] = gym.make("CartPole-v1")
            self.envs[i].seed(random_state + i)

        self.action_sz = self.envs[0].action_space.n
        self.state_sz = self.envs[0].observation_space.shape[0]

        self.n_agents = n_agent
        self.steps = 0
        self.total_reward = 0

    def step(self, action):
        obs_dict, reward_dict, done_dict = defaultdict(lambda: None), dict(), defaultdict(lambda: True)
        for i in action.keys():
            if self.prev_dones[i]:
                continue
            obs, reward, done, info = self.envs[i].step(int(action[i]))
            obs_dict[i] = obs
            reward_dict[i] = reward
            done_dict[i] = done

            real_reward = reward
            self.total_reward += real_reward
        self.steps += 1
        self.prev_dones = done_dict
        done_dict['__all__'] = all(done_dict.values())
        return obs_dict, reward_dict, done_dict, None, real_reward

    def reset(self):
        self.prev_dones = defaultdict(lambda: False)
        self.steps, self.total_reward = 0, 0
        obs_dict = dict()
        for i in range(self.n_agents):
            obs_dict[i] = self.envs[i].reset()
        return obs_dict

    def render(self):
        assert False

    def get_steps(self):
        return self.steps

    def get_total_reward(self):
        return self.total_reward

    def transform_action(self, handle, action):
        return action

    def greedy_action(self, handle):
        return None
