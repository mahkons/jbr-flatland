from configs.Config import Config

import torch
from agent.judge.Judge import Judge

class TimeTableConfig(Config):
    def __init__(self):
        pass

    def create_timetable(self):
        pass

class JudgeConfig(TimeTableConfig):
    def __init__(self, window_size_generator):
        self.window_size_generator = window_size_generator

    def create_timetable(self):
        return Judge(self.window_size_generator, torch.device("cpu"))
