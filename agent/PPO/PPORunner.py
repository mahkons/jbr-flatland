import ray
import torch
from collections import namedtuple, defaultdict
from itertools import count, chain

from flatland.envs.agent_utils import RailAgentStatus

from agent.PPO.PPORollout import PPORollout, PPOTransition
from env.Contradictions import Contradictions

class PPORunner():
    def _select_actions(self, state):
        valid_handles = list()
        internal_state = dict()
        for handle in range(self.env.n_agents):
            if self.env.agents[handle].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                    and not self.env.obs_builder.deadlock_checker.is_deadlocked(handle):
                if handle in state:
                    internal_state[handle] = state[handle]
                else:
                    internal_state[handle] = torch.tensor(self.env.obs_builder._get_internal(handle), dtype=torch.float)

        for handle in state.keys(): # not blind
            if self.env.agents[handle].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                    and not self.env.obs_builder.deadlock_checker.is_deadlocked(handle):
                valid_handles.append(handle)
                self.neighbours_state[handle].clear()
                for opp_handle in self.env.obs_builder.encountered[handle]:
                    if opp_handle in internal_state:
                        self.neighbours_state[handle].append(internal_state[opp_handle])
                    else:
                        # still zeros? there is attention now...
                        self.neighbours_state[handle].append(torch.zeros(self.env.obs_builder.state_sz)) 

        action_dict, log_probs = self.controller.select_action_batch(valid_handles, state, self.neighbours_state, train=True)
        return action_dict, log_probs


    def _save_transitions(self, state, action_dict, log_probs, next_state, reward, done, step):
        return
        self.prev_valid_state.update(state)
        self.prev_valid_action.update(action_dict)
        self.prev_valid_action_log_prob.update(log_probs)
        for handle in state.keys():
            self.prev_step[handle] = step

        for handle in next_state.keys(): # all not blind
            if not handle in self.prev_valid_state: # just departed
                continue 
            self.rollout[handle].append_transition(PPOTransition(
                self.prev_valid_state[handle],
                self.prev_valid_action[handle],
                self.prev_valid_action_log_prob[handle],
                next_state[handle],
                reward[handle],
                done[handle],
                torch.stack(self.neighbours_state[handle]),
                step + 1 - self.prev_step[handle],
            ))

    def _wrap(self, d, dtype=torch.float):
        for key, value in d.items():
            d[key] = torch.tensor(value, dtype=dtype)
        return d

    # samples one episode
    def run(self, env, controller):
        self.env = env
        self.controller = controller

        state = self._wrap(self.env.reset())
        self.prev_valid_state = state
        self.prev_valid_action = dict()
        self.rollout = defaultdict(PPORollout)
        self.prev_valid_action_log_prob = dict()
        self.neighbours_state = defaultdict(list)
        self.prev_step = torch.zeros(len(self.env.agents), dtype=torch.long)

        steps_done = 0

        while True:
            action_dict, log_probs = self._select_actions(state)

            next_state, reward, done, info, _ = self.env.step(action_dict)
            next_state, reward, done = self._wrap(next_state), self._wrap(reward), self._wrap(done)
            self._save_transitions(state, action_dict, log_probs, next_state, reward, done, steps_done)

            state = next_state
            steps_done += 1
            if done['__all__']:
                break;

        percent_done = sum([1 for agent in self.env.agents if agent.status == RailAgentStatus.DONE_REMOVED]) / self.env.n_agents
        return self.rollout, { "reward": self.env.get_total_reward(),
                               "percent_done": percent_done,
                               "steps_done": steps_done}



