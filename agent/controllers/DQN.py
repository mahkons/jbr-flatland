import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import math
import random

from agent.memory.ReplayMemory import Transition
from networks.DQN import DQNNet


class DQN(nn.Module):
    def __init__(self, env, params, device):
        super(DQN, self).__init__()
        self.state_sz = env.state_sz
        self.action_sz = env.action_sz
        self.params = params
        self.device = device

        self.eps_start = params.eps_start
        self.eps_end = params.eps_end
        self.eps_decay = params.eps_decay
        self.batch_size = params.batch_size
        self.gamma = params.gamma
        self.target_net_update_steps = params.target_net_update_steps
        self.optimizer_config = params.optimizer_config

        self.net = DQNNet(self.state_sz, self.action_sz, params.layers_sz, device).to(device)
        self.target_net = DQNNet(self.state_sz, self.action_sz, params.layers_sz, device).to(device)

        self.memory = params.memory_config.create_memory()
        self.optimizer = self.optimizer_config.create_optimizer(self.net.parameters())

        self.steps_done = 0

    def select_action(self, handle, state, train):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        if train and random.random() < eps_threshold:
            return random.randrange(self.action_sz), None
        else:
            with torch.no_grad():
                return self.net(state.unsqueeze(0).to(self.device)).max(1)[1].item(), None

    def hard_update(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def all_to_device(self, xs):
        return map(lambda x: x.to(self.device), xs)

    def calc_loss(self):
        state, action, next_state, reward, done = self.all_to_device(self.memory.sample(self.batch_size))

        state_action_values = self.net(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_values = self.target_net(next_state).max(1)[0]
            expected_state_action_values = (next_values * self.gamma) * (1 - done) + reward

        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

        return loss

    def optimize(self):
        self.steps_done += 1
        if self.steps_done % self.target_net_update_steps == 0:
            self.hard_update()

        if len(self.memory) < self.batch_size:
            return

        loss = self.calc_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def push_in_memory(self, handle, state, action, action_info, next_state, reward, done):
        self.memory.push(handle, state, action, next_state, reward, done)

    def load_controller(self, path):
        pass

    def save_controller(self, path):
        pass
