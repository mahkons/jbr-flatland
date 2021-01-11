import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from flatland.envs.agent_utils import RailAgentStatus

from agent.judge.JudgeNetwork import JudgeNetwork
from agent.judge.JudgeFeatures import JudgeFeatures
from agent.judge.ThresholdBandit import ThresholdBandit

from logger import log

class Judge():
    def __init__(self, window_size_generator, device):
        self.obs_builder = JudgeFeatures()
        self.window_size_generator = window_size_generator
        self.device = device

        self.net = JudgeNetwork(self.obs_builder.state_sz).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        self.cur_threshold = 0.97
        self.threshold_optimizer = ThresholdBandit()

        self.batch_size = 8
        self.epochs = 3

    def reset(self, env):
        self.env = env
        self.obs_builder.reset(env)
        self.window_size = self.window_size_generator(env)
        self.ready_to_depart = [0] * len(self.env.agents)
        self.send_more = self.window_size

        self.sent_priorities = torch.empty(len(self.env.agents))
        self.sent_states = torch.empty((len(self.env.agents), self.obs_builder.state_sz))

        self.cur_threshold = 0.95
        self.timer = 0
        self.prev_check = -1e9
        self.end_time = torch.empty(len(self.env.agents))
        self.all_out = False
        self.first_launch = True
        self.active_agents = set()

        self.priority_timer = 0
        self.last_priorities_update = -1e9
        self.priorities = torch.zeros(len(self.env.agents), dtype=torch.float)
        self.observations = torch.zeros((len(self.env.agents), self.obs_builder.state_sz), dtype=torch.float)

    def get_rollout(self):
        used_handles = [handle for handle in range(len(self.env.agents)) if self.ready_to_depart[handle] != 0]
        finished_handles = [handle for handle in range(len(self.env.agents)) if self.ready_to_depart[handle] == 2]
        target = torch.tensor([1. if self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED else 0. for handle in used_handles])

        # not perfect place for cur_threshold update

        reward = torch.sum(self.env._max_episode_steps - self.end_time[finished_handles])
        reward /= len(self.env.agents) * self.env._max_episode_steps
        #  self.threshold_optimizer.step(self.cur_threshold, reward)
        #  print("Threshold: {}".format(self.cur_threshold))
        #  self.cur_threshold = self.threshold_optimizer.choose_threshold()

        probs = self.sent_priorities[used_handles]
        states = self.sent_states[used_handles]
        return target, states

    def optimize(self, rollout):
        target, states = rollout
        n = len(target)

        target, states = target.to(self.device).repeat(self.epochs+1), states.to(self.device).repeat((self.epochs+1,1))
        sum_loss = 0
        for l in range(0, n*self.epochs, self.batch_size):
            r = min(len(target), l + self.batch_size)
            probs = self.net(states[l:r]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(probs, target[l:r])
            sum_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Judge Loss: {}".format(sum_loss))
        return {"loss": sum_loss}


    def get_batch(self, handles):
        with torch.no_grad():
            observations = self.obs_builder.get_many(handles)
            priorities = self.net(observations).squeeze(1)
        return observations, priorities

    def _get_observations(self, handles):
        return self.obs_builder.get_many(handles)

    def _get_priorities(self, observations):
        with torch.no_grad():
            priorities = self.net(observations).squeeze(1)
        return priorities

    def _get_expected(self, priority, observation):
        return -priority #+ observation[0]/(5 * (self.env.width + self.env.height))# or what?
        prob = F.sigmoid(priority)
        remaining_time = self.env._max_episode_steps - self.env._elapsed_steps
        return prob * min(observation[0], remaining_time) + (1 - prob) * remaining_time

    def calc_priorities(self, handles):
        self.priority_timer += 1
        if self.priority_timer > self.last_priorities_update or len(self.env.agents) < 100: # yeah
            self.observations[handles] = self._get_observations(handles)
            self.priorities[handles] = self._get_priorities(self.observations[handles])
            self.last_priorities_update = self.priority_timer
        return self.priorities[handles], self.observations[handles]


    def update(self):
        self.timer += 1
        #  self.cur_threshold -= 0.1 / self.env._max_episode_steps
        self.cur_threshold -= 1e-5
        self.update_finished()
        if self.timer - self.prev_check < 10 and len(self.env.agents) > 100: # optimizations
            return
        self.prev_check = self.timer
        self.priority_timer += 100
        self.any_changes = False

        start_time = time.time()
        if self.first_launch:
            start_time += 200
        self.first_launch = False

        self.obs_builder.update_begin(self.active_agents)

        # send_more stands for minimal number of active agents
        # judge will send agent if probability of arriving is big enough
        while not self.all_out and time.time() - start_time < 5:
            #  noise = torch.empty_like(priorities).data.normal_(0, 0.2) # exploration noise
            #  priorities += noise
            valid_handles = [handle for handle in range(len(self.env.agents)) if self.ready_to_depart[handle] == 0]
            priorities, observations = self.calc_priorities(valid_handles)

            best_handle, best_priority, best_obs = -1, -1, None
            for handle, priority, obs in zip(valid_handles, priorities, observations):
                if best_handle == -1 or \
                        self._get_expected(best_priority, best_obs) > self._get_expected(priority, obs):
                    best_handle, best_priority, best_obs = handle, priority, obs

            if best_handle == -1:
                self.all_out = True
                break

            if self.send_more <= 0 and torch.sigmoid(best_priority) < self.cur_threshold:
                # involves calculation of priorities on every iteration
                # might be slow
                break

            self.sent_priorities[best_handle] = best_priority
            self.sent_states[best_handle] = best_obs
            self._start_agent(best_handle)
            self.obs_builder.update_begin([best_handle])

        self.obs_builder.update_end(self.active_agents)

    def update_finished(self):
        for handle in range(len(self.env.agents)):
            if (self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED \
                    or self.env.obs_builder.deadlock_checker.is_deadlocked(handle))\
                and self.ready_to_depart[handle] == 1:

                self._finish_agent(handle)
                self.any_changes = True

    def is_ready(self, handle):
        return self.ready_to_depart[handle] != 0

    def _start_agent(self, handle):
        self.send_more -= 1
        self.ready_to_depart[handle] = 1
        self.active_agents.add(handle)
        self.obs_builder._start_agent(handle)

    def _finish_agent(self, handle):
        self.send_more += 1
        self.ready_to_depart[handle] = 2
        if self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED:
            self.end_time[handle] = self.timer
        else:
            self.end_time[handle] = self.env._max_episode_steps
        self.active_agents.remove(handle)
        self.obs_builder._finish_agent(handle)

    def update_net_params(self, net_params):
        self.net.load_state_dict(net_params)

    def get_net_params(self, device=None):
        state_dict = self.net.state_dict()
        if device is not None and device != self.device:
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
        return state_dict

    def load_judge(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)

    def save_judge(self, dirpath, name="judge.torch"):
        state_dict = self.get_net_params(device=torch.device("cpu"))
        torch.save(state_dict, os.path.join(dirpath, name))



# simple optimization
# naive
# make to steps and move in direction with better reward
class ThresholdOptimizer():
    def __init__(self):
        self.type = 0
        self.prev_reward, self.prev_threshold = None, None

        self.lr = 1e-3

    def get_threshold(self, threshold, reward):
        if self.type == 0:
            self.prev_threshold, self.prev_reward = threshold, reward
            new_threshold = threshold + 0.01
        else:
            delta_reward = reward - self.prev_reward
            new_threshold = self.prev_threshold + np.sign(delta_reward) * self.lr
            
        self.type = 1 - self.type
        return new_threshold

