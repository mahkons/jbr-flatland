from agent.memory.ReplayMemory import ReplayMemory

from configs.Config import Config

class MemoryConfig(Config):
    def __init__(self):
        pass

    def create_memory(self):
        pass


class ReplayMemoryConfig(MemoryConfig):
    def __init__(
            self,
            memory_size
        ):
        super(ReplayMemoryConfig, self).__init__()
        self.memory_size = memory_size

    def create_memory(self):
        return ReplayMemory(self.memory_size)
    
