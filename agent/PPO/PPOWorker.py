import ray
import torch

from agent.PPO.PPORunner import PPORunner

@ray.remote
class PPOWorker():
    def __init__(self, worker_handle, env_config, controller_config):
        self.worker_handle = worker_handle
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller(torch.device("cpu"))
        self.gamma = controller_config.gamma
        self.lam = controller_config.lam
        self.gae_horizon = controller_config.gae_horizon

        self.runner = PPORunner()

    def run(self, ppo_net_params, judge_net_params):
        self.controller.update_net_params(ppo_net_params)
        self.env.obs_builder.timetable.update_net_params(judge_net_params)

        rollout_dict, info = self.runner.run(self.env, self.controller)
        info["handle"] = self.worker_handle
        info["shaped_reward"] = 0
        info["env"] = self.env.cur_env
        info["judge_threshold"] = self.env.obs_builder.timetable.cur_threshold

        for handle, rollout in rollout_dict.items():
            if rollout.is_empty():
                continue
            state, action, log_prob, reward, next_state, done, neighbours_states, actual_len = rollout.unzip_transitions()
            rollout.gae = self._calc_gae(state, next_state, reward, done, actual_len)
            info["shaped_reward"] += torch.sum(reward).item()
        return rollout_dict, self.env.obs_builder.timetable.get_rollout(), info

    def _calc_gae(self, state, next_state, reward, done, actual_len):
        # moving this code to worker saves learner time
        with torch.no_grad():
            state_values = self.controller.critic_net(state).squeeze(1)
            next_values = self.controller.critic_net(next_state).squeeze(1) # TODO almost same calculation in prev line
            expected_state_values = (next_values * torch.pow(self.gamma, actual_len.float()) * (1 - done)) + reward
            gae = expected_state_values - state_values
            gae_copy = gae.clone()
            for i in reversed(range(len(gae) - 1)):
                gae[i] += gae[i + 1] * self.lam * self.gamma
                if i + self.gae_horizon < len(gae):
                    gae[i] -= gae_copy[i + self.gae_horizon] * (self.lam * self.gamma) ** self.gae_horizon
        return gae
