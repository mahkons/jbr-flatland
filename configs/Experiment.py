from configs.Config import Config

class Experiment(Config):
    def __init__(
            self,
            opt_steps,
            episodes,
            device,
            logname,
            random_seed,
            env_config,
            controller_config,
        ):
        super(Experiment, self).__init__()
        self.opt_steps = opt_steps
        self.episodes = episodes
        self.device = device 
        self.logname = logname 
        self.random_seed = random_seed 
        self.env_config = env_config
        self.controller_config = controller_config

    @staticmethod
    def from_dict(params):
        pass
