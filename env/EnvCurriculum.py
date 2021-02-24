import numpy as np

# In Multiagent every worker goes through curriculum
class EnvCurriculum():
    def __init__(self, env_configs, env_episodes):
        self.env_configs = env_configs
        self.env_episodes = env_episodes
        self._pos_env = -1
        self.env = self._start_next()


    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        return getattr(self.env, name)


    def _start_next(self):
        if self._pos_env + 1 < len(self.env_configs): # keep launching last env_episode
            self._pos_env += 1
        self._cur_env_episode = 0
        return self.env_configs[self._pos_env].create_env()


    def _reset(self):
        if self._cur_env_episode == self.env_episodes[self._pos_env]:
            self.env = self._start_next()
        self._cur_env_episode += 1
        return self.env.reset()


class EnvCurriculumSample():
    def __init__(self, env_configs, env_probs):
        self.envs = [config.create_env() for config in env_configs]
        self.env_probs = np.array(env_probs, dtype=np.float)
        self.env_probs /= np.sum(self.env_probs)

        self.env = self._start_next()

    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        return getattr(self.env, name)

    def _start_next(self):
        self._pos_env = np.random.choice(len(self.envs), p=self.env_probs)
        return self.envs[self._pos_env]

    def _reset(self):
        self.env = self._start_next()
        return self.env.reset()
