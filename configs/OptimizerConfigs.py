import torch.optim

from configs.Config import Config

class OptimizerConfig(Config):
    def __init__(self):
        pass

    def create_optimizer(self):
        raise NotImplementedError()


class AdamConfig(OptimizerConfig):
    def __init__(self, lr: float):
        super(AdamConfig, self).__init__()
        self.lr = lr

    def create_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr)
