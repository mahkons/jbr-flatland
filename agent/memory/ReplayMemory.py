import torch
import random

from agent.memory.Memory import Memory, Transition

class ReplayMemory(Memory):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, handle, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(
            torch.tensor(state, dtype=torch.float),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float),
            torch.tensor(reward, dtype=torch.float),
            torch.tensor(done, dtype=torch.float),
        )
        
        self.position += 1
        if self.position == self.capacity:
            self.position = 0
            
    def sample(self, batch_size):
        return self.get_transitions(self.sample_positions(batch_size))

    def get_transitions(self, positions):
        transitions = [self.memory[pos] for pos in positions]

        batch = Transition(*zip(*transitions))
        state = torch.stack(batch.state)
        action = torch.stack(batch.action)
        reward = torch.stack(batch.reward)
        next_state = torch.stack(batch.next_state)
        done = torch.stack(batch.done)

        return state, action, next_state, reward, done

    def sample_positions(self, batch_size):
        return random.sample(range(len(self.memory)), batch_size)

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.memory = list()
        self.position = 0
