from agent.PPO.PPOController import PPOController

from configs.Config import Config

class ControllerConfig(Config):
    def __init__(self):
        pass

    def create_controller(self, env, memory, device):
        pass


class PPOConfig(ControllerConfig):
    def __init__(
            self,
            state_sz,
            action_sz,
            neighbours_depth,
            optimizer_config,
            batch_size,
            gae_horizon,
            epochs_update,
            gamma,
            lam,
            clip_eps,
            value_loss_coeff,
            entropy_coeff,
            actor_layers_sz,
            critic_layers_sz,
    ):
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.neighbours_depth = neighbours_depth
        self.optimizer_config = optimizer_config
        self.batch_size = batch_size
        self.gae_horizon = gae_horizon
        self.epochs_update = epochs_update
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.actor_layers_sz = actor_layers_sz
        self.critic_layers_sz = critic_layers_sz

    def create_controller(self, device):
        return PPOController(self, device)
