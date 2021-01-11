import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim
import math
import random
from collections import defaultdict
import os

from agent.controllers.Controller import Controller
from agent.memory.Memory import Transition
from agent.memory.OneStepMemory import OneStepMemory
from networks.ActorCritic import ActorNet, CriticNet
    

class ActorCritic(Controller):
    def __init__(self, env, params, device):
        super(ActorCritic, self).__init__()
        self.state_sz = env.state_sz
        self.action_sz = env.action_sz

        self.params = params
        self.device = device

        self.gamma = params.gamma
        self.entropy_coeff = params.entropy_coeff

        self.actor_net = ActorNet(self.state_sz, self.action_sz, params.actor_layers_sz).to(device)
        self.actor_optimizer = params.actor_optimizer_config.create_optimizer(self.actor_net.parameters())
        self.critic_net = CriticNet(self.state_sz, self.action_sz, params.critic_layers_sz).to(device)
        self.critic_optimizer = params.critic_optimizer_config.create_optimizer(self.critic_net.parameters())

        self.memory = OneStepMemory()
        self.steps_done = 0


    def select_action(self, handle, state, train=True):
        # TODO batch action_selection
        with torch.no_grad():
            action_distribution = Categorical(logits=self.actor_net(state))

        if not train:
            action = action_distribution.probs.argmax().item()
            return action, None

        action = action_distribution.sample()
        #  print(handle, action_distribution.probs)
        return action.item(), None

    def all_to_device(self, xs):
        return map(lambda x: x.to(self.device), xs)

    # calculates loss for all agents at once
    def calc_loss_batch(self, state, action, next_state, reward, done):
        state_values = self.critic_net(state).squeeze(1)
        next_values = self.critic_net(next_state).squeeze(1)
        expected_state_values = (next_values * self.gamma * (1 - done)) + reward
        critic_loss = F.smooth_l1_loss(state_values, expected_state_values)

        with torch.no_grad():
            advantage = expected_state_values - state_values
    
        actor_loss = 0
        logits = self.actor_net(state) # Aaargh not same as in action sample =(
        for i in range(len(state)):
            action_distribution = Categorical(logits=logits[i])
            actor_loss -= action_distribution.log_prob(action[i]) * advantage[i];
            actor_loss -= self.entropy_coeff * action_distribution.entropy()
        actor_loss /= len(state)

        return actor_loss, critic_loss

    def optimize(self):
        self.steps_done += 1

        transitions = self.memory.get_all_transitions()
        if transitions is None:
            return
        state, action, next_state, reward, done = self.all_to_device(transitions)

        actor_loss, critic_loss = self.calc_loss_batch(state, action, next_state, reward, done)
        self.memory.clean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def push_in_memory(self, handle, state, action, action_info, next_state, reward, done):
        self.memory.push(handle, state, action, next_state, reward, done)
    
    def load_controller(self, path):
        model = torch.load(path)
        self.actor_net.load_state_dict(model['actor'])
        self.critic_net.load_state_dict(model['critic'])

    def save_controller(self, dirpath):
        torch.save(self.params, os.path.join(dirpath, "params.torch"))
        torch.save({'actor': self.actor_net.state_dict(), 'critic': self.critic_net.state_dict()},
                os.path.join(dirpath, "controller.torch"))
