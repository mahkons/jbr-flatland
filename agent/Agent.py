import numpy as np
from collections import namedtuple, defaultdict
from itertools import count
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F

from flatland.envs.agent_utils import RailAgentStatus

class Agent:
    def __init__(self, env, controller, device):
        self.env = env
        self.controller = controller
        self.device = device

    def select_actions(self, state, done, train):
        action_dict = dict()
        action_info = dict()
        for i in state.keys(): # not blind
            if self.env.agents[i].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                    and not self.env.obs_builder.deadlock_checker.is_deadlocked(i):
                action_dict[i], action_info[i] = self.controller.select_action(
                        handle=i,
                        state=torch.tensor(state[i], dtype=torch.float, device=self.device),
                        train=train
                )
        return action_dict, action_info

    def save_transitions(self, state, action_dict, action_info, next_state, reward, done):
        for i in next_state.keys(): # all not blind
            if not i in self.prev_valid_state: # just departed
                continue 
            self.controller.push_in_memory(
                    i,
                    self.prev_valid_state[i],
                    self.prev_valid_action[i],
                    self.prev_valid_action_info[i],
                    next_state[i],
                    reward[i],
                    done[i]
            )

    def rollout(self, train, show=False, step_by_step=False):
        state = self.env.reset()
        done = defaultdict(lambda: False)
        
        self.prev_valid_state = state
        self.prev_valid_action = dict()
        self.prev_valid_action_info = dict()

        while True:
            if show:
                self.env.render()
                #sleep(1)

            action_dict, action_info = self.select_actions(state, done, train)
            self.prev_valid_action.update(action_dict)
            self.prev_valid_action_info.update(action_info)
            self.prev_valid_state.update(state)
            if step_by_step:
                print(action_dict)
                input("Press any key to continue...")
            next_state, reward, done, info, _ = self.env.step(action_dict)

            if train:
                self.save_transitions(state, action_dict, action_info, next_state, reward, done)
                self.controller.optimize()

            state = next_state
            if done['__all__']:
                break;

        return self.env.get_total_reward(), self.env.get_steps()

    def steps_done(self):
        return self.controller.steps_done

    def save_controller(self, path):
        self.controller.save_controller(path)

