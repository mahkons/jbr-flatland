import torch
from multiprocessing import Pool
from functools import partial
import argparse
from copy import deepcopy
import random

from configs import Experiment, FlatlandConfig, AdamConfig, ActorCriticConfig, DQNConfig

def generate_experiments():
    exp_list = list()

    obs_config = SimpleObservationConfig(max_depth=2)

    env_config = FlatlandConfig(
        height=35,
        width=35,
        n_agents=1,
        n_cities=3,
        grid_distribution_of_cities=False,
        max_rails_between_cities=2,
        max_rail_in_cities=4,
        observation_builder_config=obs_config,
        random_seed=RANDOM_SEED,
    )

    controller_config = ActorCriticConfig(
        actor_optimizer_config = AdamConfig(lr=1e-4),
        critic_optimizer_config = AdamConfig(lr=1e-4),
        gamma=0.99,
        entropy_coeff=0.01,
        actor_layers_sz=[256, 128],
        critic_layers_sz=[256, 128],
    )


    torch.save(exp_list, "generated/exp_list")

if __name__ == "__main__":
    generate_experiments()
