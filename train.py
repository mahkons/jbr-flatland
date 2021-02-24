import torch
import numpy as np
import random

from env.observations import SimpleObservation

from agent.PPO.PPOLearner import PPOLearner

from configs import Experiment, AdamConfig, FlatlandConfig, \
    SimpleObservationConfig, EnvCurriculumConfig, \
    EnvCurriculumSampleConfig, SimpleRewardConfig, SparseRewardConfig, NearRewardConfig, \
    DeadlockPunishmentConfig,  RewardsComposerConfig, \
    NotStopShaperConfig, FinishRewardConfig, JudgeConfig
from env.Flatland import Flatland
from agent.judge.Judge import ConstWindowSizeGenerator, LinearOnAgentNumberSizeGenerator
from logger import log, init_logger

from params import PPOParams
from params import test_env, PackOfAgents

def init_random_seeds(RANDOM_SEED, cuda_determenistic):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    if cuda_determenistic:
        torch.backends.cudnn.deterministic = cuda_determenistic
        torch.backends.cudnn.benchmark = cuda_determenistic

def train_ppo(exp, n_workers):
    init_random_seeds(exp.random_seed, cuda_determenistic=False)
    log().update_params(exp)

    learner = PPOLearner(exp.env_config, exp.controller_config, n_workers, exp.device)
    learner.rollouts(max_opt_steps=exp.opt_steps, max_episodes=exp.episodes)
    learner.controller.save_controller(log().get_log_path())


if __name__ == "__main__":
    RANDOM_SEED = 23
    torch.set_printoptions(precision=6, sci_mode=False)
    logname = "tmp"
    init_logger("logdir", logname, use_wandb=False)

    timetable_config = JudgeConfig(LinearOnAgentNumberSizeGenerator(0.0, 2))
    obs_builder_config = SimpleObservationConfig(max_depth=3, neighbours_depth=3, timetable_config=timetable_config)
    reward_config = RewardsComposerConfig((
        FinishRewardConfig(finish_value=10),
        NearRewardConfig(coeff=0.01),
        DeadlockPunishmentConfig(value=-5),
        NotStopShaperConfig(on_switch_value=0, other_value=0),
    ))
    envs = [(PackOfAgents(RANDOM_SEED), 1)]

    workers = 1
    exp = Experiment(
        opt_steps=10**10,
        episodes=100000,
        device=torch.device("cuda"),
        logname=logname,
        random_seed=RANDOM_SEED,
        env_config = EnvCurriculumSampleConfig(*zip(*envs),
        obs_builder_config=obs_builder_config,
        reward_config=reward_config),
        controller_config = PPOParams(),
    )

    train_ppo(exp, workers)


