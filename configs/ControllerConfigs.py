from agent.controllers.ActorCritic import ActorCritic
from agent.controllers.DQN import DQN
from agent.PPO.PPOController import PPOController

from configs.Config import Config

class ControllerConfig(Config):
    def __init__(self):
        pass

    def create_controller(self, env, memory, device):
        pass


class ActorCriticConfig(ControllerConfig):
    def __init__(
            self,
            actor_optimizer_config,
            critic_optimizer_config,
            gamma,
            entropy_coeff,
            actor_layers_sz,
            critic_layers_sz,
        ):
        super(ActorCriticConfig, self).__init__()
        self.actor_optimizer_config = actor_optimizer_config
        self.critic_optimizer_config = critic_optimizer_config
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.actor_layers_sz = actor_layers_sz
        self.critic_layers_sz = critic_layers_sz

    def create_controller(self, env, device):
        return ActorCritic(env, self, device)

class DQNConfig(ControllerConfig):
    def __init__(
            self,
            memory_config,
            optimizer_config,
            batch_size,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            target_net_update_steps,
            layers_sz,
        ):
        super(DQNConfig, self).__init__()
        self.memory_config = memory_config
        self.optimizer_config = optimizer_config
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_net_update_steps = target_net_update_steps
        self.layers_sz = layers_sz

    def create_controller(self, env, device):
        return DQN(env, self, device)


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
