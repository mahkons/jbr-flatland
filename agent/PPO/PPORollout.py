import torch
from collections import namedtuple

PPOTransition = namedtuple('PPOTransition', ('state', 'action', 'log_prob', 'next_state', 'reward', 
    'done', 'neighbours_states', 'actual_len'))

class PPORollout():
    def __init__(self):
        self.transitions = list()
        self.gae = None

    def append_transition(self, transition):
        self.transitions.append(transition)

    def unzip_transitions(self, device=torch.device("cpu")):
        batch = PPOTransition(*zip(*self.transitions))
        state = torch.stack(batch.state).to(device)
        action = torch.stack(batch.action).to(device)
        log_prob = torch.stack(batch.log_prob).to(device)
        reward = torch.stack(batch.reward).to(device)
        next_state = torch.stack(batch.next_state).to(device)
        done = torch.stack(batch.done).to(device)
        neighbours_states = torch.stack(batch.neighbours_states).to(device)
        actual_len = torch.stack(batch.actual_len).to(device)
        return state, action, log_prob, reward, next_state, done, neighbours_states, actual_len

    def is_empty(self):
        return not self.transitions

    @staticmethod
    def combine_rollouts(rollouts):
        combined = PPORollout()
        combined.transitions = sum((rollout.transitions for rollout in rollouts), [])
        combined.gae = torch.cat(tuple(rollout.gae for rollout in rollouts))
        return combined
