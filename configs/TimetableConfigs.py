from configs.Config import Config

import torch
from agent.judge.Judge import Judge

class TimeTableConfig(Config):
    def __init__(self):
        pass

    def create_timetable(self):
        pass

class JudgeConfig(TimeTableConfig):
    def __init__(self, window_size_generator, lr, batch_size, optimization_epochs):
        self.window_size_generator = window_size_generator
        self.lr = lr
        self.batch_size = batch_size
        self.optimization_epochs = optimization_epochs

    def create_timetable(self):
        return Judge(self.window_size_generator, self.lr, self.batch_size, self.optimization_epochs, torch.device("cpu"))
