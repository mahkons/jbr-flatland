from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Memory:
    def push(self, id, state, action, next_state, reward, done):
        pass

    def __len__(self):
        pass

    def clean(self):
        pass

