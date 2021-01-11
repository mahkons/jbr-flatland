import torch.nn as nn

class Controller(nn.Module):
    def select_action(self, id, state, train):
        pass

    def optimize(self):
        pass

    def push_in_memory(self, id, state, action, next_state, reward, done):
        pass

    @staticmethod
    def load_controller(self, path):
        pass

    def save_controller(self, path):
        pass

