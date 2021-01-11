from collections import namedtuple
import math
import copy

from configs import ActorCriticConfig, ReplayMemoryConfig, AdamConfig, DQNConfig, PPOConfig, \
        FlatlandConfig, SimpleObservationConfig, SimpleRewardConfig, SparseRewardConfig


def PPOParams():
    return PPOConfig(
        state_sz = 205, # TODO pass approprietly
        action_sz = 3,
        neighbours_depth = 3,
        optimizer_config = AdamConfig(lr=1e-5),
        batch_size = 32,
        gae_horizon = 16,
        epochs_update = 3,
        gamma = 0.995,
        lam = 0.95,
        clip_eps = 0.2,
        value_loss_coeff = 0.5,
        entropy_coeff = 0.01,
        actor_layers_sz=[256, 128],
        critic_layers_sz=[256, 128],
    )

def DQNConfigParams():
    return DQNConfig(
        memory_config = ReplayMemoryConfig(200000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 20000,
        target_net_update_steps = 2500,
        layers_sz = [256, 128],
    )


# env configs

def FewAgents(random_seed):
    return FlatlandConfig(
    )

def ActorCriticParams():
    return ActorCriticConfig(
        actor_optimizer_config = AdamConfig(lr=2e-4),
        critic_optimizer_config = AdamConfig(lr=2e-4),
        gamma=0.99,
        entropy_coeff=2,
        actor_layers_sz=[256, 128],
        critic_layers_sz=[256, 128],
    )

def DQNConfigParams():
    return DQNConfig(
        memory_config = ReplayMemoryConfig(200000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 20000,
        target_net_update_steps = 2500,
        layers_sz = [256, 128],
    )


# env configs

def FewAgents(random_seed):
    return FlatlandConfig(
            height=35,
            width=25,
            n_agents=2,
            n_cities=2,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./100,
            random_seed=random_seed,
            greedy=True,
        )


def SeveralAgents(random_seed):
    return FlatlandConfig(
            height=35,
            width=35,
            n_agents=5,
            n_cities=3,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./200,
            random_seed=random_seed,
            greedy=True,
        )


def PackOfAgents(random_seed):
    return FlatlandConfig(
            height=35,
            width=35,
            n_agents=10,
            n_cities=4,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./300,
            random_seed=random_seed,
            greedy=True,
        )

def LotsOfAgents(random_seed):
    return FlatlandConfig(
            height=40,
            width=60,
            n_agents=20,
            n_cities=6,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./400,
            random_seed=random_seed,
            greedy=True,
        )

def HordeOfAgents(random_seed):
    return FlatlandConfig(
            height=120,
            width=80,
            n_agents=50,
            n_cities=10,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./1000,
            random_seed=random_seed,
            greedy=True,
        )


def env6(random_seed):
    return FlatlandConfig(
            height=40,
            width=60,
            n_agents=80,
            n_cities=9,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./800,
            random_seed=random_seed,
            greedy=True,
        )


def env7(random_seed):
    return FlatlandConfig(
            height=60,
            width=40,
            n_agents=80,
            n_cities=13,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./800,
            random_seed=random_seed,
            greedy=True,
        )


def env8(random_seed):
    return FlatlandConfig(
            height=60,
            width=60,
            n_agents=80,
            n_cities=17,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./800,
            random_seed=random_seed,
            greedy=True,
        )

def env9(random_seed):
    return FlatlandConfig(
            height=80,
            width=120,
            n_agents=100,
            n_cities=21,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./1000,
            random_seed=random_seed,
            greedy=True,
        )

def env10(random_seed):
    return FlatlandConfig(
            height=100,
            width=80,
            n_agents=100,
            n_cities=25,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./1000,
            random_seed=random_seed,
            greedy=True,
        )

def env11(random_seed):
    return FlatlandConfig(
            height=100,
            width=100,
            n_agents=200,
            n_cities=29,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./2000,
            random_seed=random_seed,
            greedy=True,
        )

def env12(random_seed):
    return FlatlandConfig(
            height=150,
            width=150,
            n_agents=200,
            n_cities=33,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./2000,
            random_seed=random_seed,
            greedy=True,
        )

def env13(random_seed):
    return FlatlandConfig(
            height=150,
            width=150,
            n_agents=400,
            n_cities=37,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./4000,
            random_seed=random_seed,
            greedy=True,
        )

_test_env = list()
for i in range(50):
    if not i:
        n_agents = 1
        n_cities = 2
        x_dim = 25
        y_dim = 25
    else:
        n_agents = n_agents + math.ceil(10 ** (len(str(n_agents)) - 1)* 0.75)
        n_cities = n_agents//10 + 2
        x_dim = math.ceil(math.sqrt(150 * n_cities)) + 7
        y_dim = x_dim

    _test_env.append(FlatlandConfig(
        height=x_dim,
        width=y_dim,
        n_agents=n_agents,
        n_cities=n_cities,
        grid_distribution_of_cities=False,
        max_rails_between_cities=2,
        max_rail_in_cities=4,
        observation_builder_config=None,
        reward_config=None,
        malfunction_rate=1./1500,
        random_seed=None,
        greedy=True,
    ))


def test_env(random_seed, i):
    env = _test_env[i]
    env.random_seed = random_seed
    return env
