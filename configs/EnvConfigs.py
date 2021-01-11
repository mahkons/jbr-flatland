from env.Flatland import Flatland, FlatlandWrapper
from env.GreedyFlatland import GreedyFlatland
from env.CartPole import MultiCartPole
from env.LunarLander import MultiLunarLander
from env.EnvCurriculum import EnvCurriculum, EnvCurriculumSample

from configs.Config import Config

class EnvConfig(Config):
    def __init__(self):
        pass

    def create_env(self):
        pass


class FlatlandConfig(EnvConfig):
    def __init__(
            self,
            height,
            width,
            n_agents,
            n_cities,
            grid_distribution_of_cities,
            max_rails_between_cities,
            max_rail_in_cities,
            observation_builder_config,
            reward_config,
            malfunction_rate,
            greedy,
            random_seed,
        ):
        super(FlatlandConfig, self).__init__()
        self.height = height
        self.width = width
        self.n_agents = n_agents
        self.n_cities = n_cities
        self.grid_distribution_of_cities = grid_distribution_of_cities
        self.max_rails_between_cities = max_rails_between_cities
        self.max_rail_in_cities = max_rail_in_cities
        self.observation_builder_config = observation_builder_config
        self.reward_config = reward_config
        self.malfunction_rate = malfunction_rate
        self.random_seed = random_seed
        self.greedy = greedy

    def update_random_seed(self):
        self.random_seed += 1

    def set_obs_builder_config(self, obs_builder_config):
        self.observation_builder_config = obs_builder_config

    def set_reward_config(self, reward_config):
        self.reward_config = reward_config

    def create_env(self):
        obs_builder = self.observation_builder_config.create_builder()
        reward_shaper = self.reward_config.create_reward_shaper()
        rail_env = FlatlandWrapper(Flatland(
            height=self.height,
            width=self.width,
            n_agents=self.n_agents,
            n_cities=self.n_cities,
            grid_distribution_of_cities=self.grid_distribution_of_cities,
            max_rails_between_cities=self.max_rails_between_cities,
            max_rail_in_cities=self.max_rail_in_cities,
            observation_builder=obs_builder,
            malfunction_rate=self.malfunction_rate,
            random_seed=self.random_seed,
        ), reward_shaper=reward_shaper)
        if self.greedy:
            rail_env = GreedyFlatland(rail_env)
        return rail_env


class EnvCurriculumConfig(EnvConfig):
    def __init__(self, env_configs, env_episodes, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_episodes = env_episodes

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculum(self.env_configs, self.env_episodes)


class EnvCurriculumSampleConfig(EnvConfig):
    def __init__(self, env_configs, env_probs, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_probs = env_probs

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculumSample(self.env_configs, self.env_probs)


class CartPoleConfig(EnvConfig):
    def __init__(self, n_agents, random_seed):
        self.n_agents = n_agents
        self.random_seed = random_seed

    def update_random_seed(self):
        self.random_seed += 1

    def create_env(self):
        return MultiCartPole(self.n_agents, self.random_seed)


class LunarLanderConfig(EnvConfig):
    def __init__(self, n_agents, random_seed):
        self.n_agents = n_agents
        self.random_seed = random_seed

    def update_random_seed(self):
        self.random_seed += 1

    def create_env(self):
        return MultiLunarLander(self.n_agents, self.random_seed)
