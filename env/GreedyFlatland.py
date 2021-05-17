import numpy as np
from collections import defaultdict

from env.Flatland import get_new_position, DelegatedAttribute
from flatland.envs.agent_utils import RailAgentStatus

_gamma = 0.999 # TODO pass it somehow

class GreedyFlatland():
    def __init__(self, env):
        self.env = env
        self.greedy_way = None

        self.switches = None
        self.switches_neighbors = None
        self.location_has_target = None
        self.target_neighbors = None


        self.n_cities = self.env.n_cities
        self.n_agents = self.env.n_agents
        self.action_sz = self.env.action_sz
        self.state_sz = self.env.state_sz
        self.max_time_steps = self.env.max_time_steps

        self.agents = DelegatedAttribute(self.env, "agents")
        self.rail = DelegatedAttribute(self.env, "rail")
        self.obs_builder = DelegatedAttribute(self.env, "obs_builder")
        self.distance_map = DelegatedAttribute(self.env, "distance_map")

        self.cur_env = 0

    def greedy_action(self, handle):
        return 0

    def step(self, action_dict):
        transformed_action_dict = dict(action_dict)
        for handle in range(self.env.n_agents):
            if self.obs_builder.greedy_checker.greedy_position(handle):
                transformed_action_dict[handle] = self.greedy_action(handle)
        action_dict = transformed_action_dict
        obs, reward, done, info, real_reward = self.env.step(action_dict)

        for handle in action_dict.keys():
            self.skipped_rewards[handle] += reward[handle] * (_gamma ** self.was_greedy[handle])
            if self.obs_builder.greedy_checker.greedy_position(handle): # next state makes action greedy
                self.was_greedy[handle] += 1
            else:
                reward[handle] = self.skipped_rewards[handle]
                self.skipped_rewards[handle] = 0
                self.was_greedy[handle] = 0

        return obs, reward, done, info, real_reward

    def reset(self):
        obs = self.env.reset()
        self.was_greedy = [0] * self.n_agents
        self.skipped_rewards = [0] * self.n_agents
        return obs

    def render(self):
        self.env.render()

    def get_steps(self):
        return self.env.get_steps()

    def get_total_reward(self):
        return self.env.get_total_reward()

    def get_available_actions(self, handle):
        return self.env.get_available_actions(handle)

    def transform_action(self, handle, action):
        return self.env.transform_action(handle, action)
