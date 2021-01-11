import torch

from agent.memory.Memory import Memory, Transition
from collections import defaultdict

# saves last transition for every agent
class OneStepMemory(Memory):
    def __init__(self):
        self.last_transition = dict()

    def push(self, id, state, action, next_state, reward, done):
        self.last_transition[id] = (
            torch.tensor(state, dtype=torch.float),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float),
            torch.tensor(reward, dtype=torch.float),
            torch.tensor(done, dtype=torch.float),
        )

    def get_transition(self, id):
        return self.last_transition[id]

    def get_all_transitions(self):
        return self.get_transitions(self.last_transition.keys())

    def get_transitions(self, ids):
        if not ids:
            return None
        transitions = [self.last_transition[id] for id in ids]

        batch = Transition(*zip(*transitions))
        state = torch.stack(batch.state)
        action = torch.stack(batch.action)
        reward = torch.stack(batch.reward)
        next_state = torch.stack(batch.next_state)
        done = torch.stack(batch.done)

        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.last_transition.clear()
