import ray
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import namedtuple
from itertools import count, chain
import time
from copy import deepcopy

from agent.PPO.PPOWorker import PPOWorker
from agent.PPO.PPORunner import PPORunner
from agent.PPO.PPORollout import PPORollout
from agent.PPO.PPOLosses import value_loss, policy_loss, value_loss_with_IS
from agent.judge.Judge import Judge
from logger import log

class PPOLearner():
    def __init__(self, env_config, controller_config, n_workers, device):
        self.n_workers = n_workers
        self.controller = controller_config.create_controller(device)
        self.judge = Judge(None, device)
        self.device = device

        self.controller.load_controller("logdir/MoreNeighbours/controller.torch")

        num_gpus = 0
        if device == torch.device("cuda"):
            num_gpus = 1

        ray.init(num_gpus=num_gpus)

        self.agents = [None] * n_workers
        for runner_handle in range(n_workers):
            self.agents[runner_handle] = PPOWorker.remote(runner_handle, env_config, controller_config)
            env_config.update_random_seed()


        self.batch_size = controller_config.batch_size
        self.gae_horizon = controller_config.gae_horizon
        self.value_loss_coeff = controller_config.value_loss_coeff
        self.entropy_coeff = controller_config.entropy_coeff
        self.lam = controller_config.lam
        self.gamma = controller_config.gamma
        self.epochs_update = controller_config.epochs_update
        self.clip_eps = controller_config.clip_eps
        self.optimizer = controller_config.optimizer_config.create_optimizer(
                chain(self.controller.critic_net.parameters(), self.controller.actor_net.parameters()))

        self.train_state = LearnerState(train_iters=2000, exploit_iters=500)

    # TODO different updates for target_actor/actor
    def _calc_loss(self, state, action, old_log_prob, reward, next_state, done, gae, neighbours_states, actual_len):
        state_values = self.controller.critic_net(state).squeeze(1)
        with torch.no_grad():
            next_state_values = self.controller.critic_net(next_state).squeeze(1)

        logits = self.controller.actor_net._make_logits(states, neighbours_states)

        action_distribution = Categorical(logits=logits)
        new_log_prob = action_distribution.log_prob(action)

        critic_loss = value_loss_with_IS(state_values, next_state_values, new_log_prob, old_log_prob, reward, done, self.gamma, actual_len)
        actor_loss = policy_loss(gae, new_log_prob, old_log_prob, self.clip_eps)
        entropy_loss = -action_distribution.entropy().mean()

        return critic_loss * self.value_loss_coeff + actor_loss + entropy_loss * self.entropy_coeff


    def _optimize(self, rollout_dict):
        # all agents rollouts combined
        rollouts = [rollout for rollout in rollout_dict.values() if not rollout.is_empty()]
        if not rollouts:
            return
        combined_rollout = PPORollout.combine_rollouts(rollouts)

        state, action, log_prob, reward, next_state, done, neighbours_states, actual_len = combined_rollout.unzip_transitions(self.device)
        gae = combined_rollout.gae.to(self.device)

        for _ in range(self.epochs_update):
            state, action, log_prob, reward, next_state, done, gae, neighbours_states, actual_len = \
                    _permute_all([state, action, log_prob, reward, next_state, done, gae, neighbours_states, actual_len])
            for l in range(0, len(state), self.batch_size):
                r = min(l + self.batch_size, len(state))
                loss = self._calc_loss(state[l:r], action[l:r], log_prob[l:r],
                        reward[l:r], next_state[l:r], done[l:r], gae[l:r], neighbours_states[l:r], actual_len[l:r])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.controller.soft_update(tau=0.05)


    def rollouts(self, max_opt_steps=10**10, max_episodes=10**10):
        log().add_plot("reward", ("train_episode", "train_steps", "reward", "env"))
        log().add_plot("shaped_reward", ("train_episode", "train_steps", "reward", "env"))
        log().add_plot("percent_done", ("train_episode", "train_steps", "percent_done", "env"))
        log().add_plot("time", ("train_episode", "train_steps", "time", "env"))
        log().add_plot("judge_loss", ("episode", "train_steps", "judge_loss", "env"))
        log().add_plot("judge_threshold", ("episode", "train_steps", "judge_threshold", "env"))

        controller_params, judge_params = self.controller.get_net_params(device=torch.device("cpu")), \
                self.judge.get_net_params(device=torch.device("cpu"))
        rollouts_list = [agent.run.remote(controller_params, judge_params) for agent in self.agents]
        cur_steps, cur_episode = 0, 0
        while True:
            done_id, rollouts_list = ray.wait(rollouts_list)
            rollout, judge_rollout, info = ray.get(done_id)[0]

            cur_steps += info["steps_done"]
            cur_episode += 1

            print(cur_episode, info["reward"], info["shaped_reward"], info["env"])
            log().add_plot_point("reward", (cur_episode, cur_steps, info["reward"], info["env"]))
            log().add_plot_point("shaped_reward", (cur_episode, cur_steps, info["shaped_reward"], info["env"]))
            log().add_plot_point("percent_done", (cur_episode, cur_steps, info["percent_done"], info["env"]))
            log().add_plot_point("time", (cur_episode, cur_steps, time.time(), info["env"]))

            if cur_episode % 100 == 0:
                log().save_logs()
            if cur_episode % 250 == 0:
                self.controller.save_controller(log().get_log_path(), "final_controller.torch")
                self.judge.save_judge(log().get_log_path(), "final_judge.torch")
            if self.train_state.is_training():
                #  self._optimize(rollout)
                judge_info = self.judge.optimize(judge_rollout)
                log().add_plot_point("judge_loss", (cur_episode, cur_steps, judge_info["loss"], info["env"]))
                log().add_plot_point("judge_threshold", (cur_episode, cur_steps, info["judge_threshold"], info["env"]))
            

            self.train_state.step(self.controller, self.judge, info["reward"])
            if cur_steps >= max_opt_steps or cur_episode >= max_episodes:
                break
                
            controller_params, judge_params = self.controller.get_net_params(device=torch.device("cpu")), \
                    self.judge.get_net_params(device=torch.device("cpu"))
            rollouts_list.extend([self.agents[info["handle"]].run.remote(controller_params, judge_params)])

        log().save_logs()
        return

def _permute_all(tensors):
    permutation = torch.randperm(len(tensors[0]))
    for tensor in tensors:
        assert len(tensor) == len(permutation)
    return (tensor[permutation] for tensor in tensors)


class LearnerState():
    def __init__(self, train_iters, exploit_iters):
        self.train_iters = train_iters
        self.exploit_iters = exploit_iters

        self.best_exploit_reward = -np.inf
        self.cur_exploit_reward = 0

        self.cur_steps = 0
        self.train = (train_iters != 0)

    def step(self, controller, judge, reward):
        if self.is_training():
            self._step_train()
        else:
            self._step_exploit(controller, judge, reward)

    def is_training(self):
        return (self.cur_steps % (self.train_iters + self.exploit_iters)) < self.train_iters

    def _step_train(self):
        self.cur_steps += 1

    def _step_exploit(self, controller, judge, reward):
        self.cur_exploit_reward += reward
        self.cur_steps += 1
        if self.is_training(): # end of exploit
            if self.cur_exploit_reward > self.best_exploit_reward:
                self.best_exploit_reward = self.cur_exploit_reward
                controller.save_controller(log().get_log_path())
                judge.save_judge(log().get_log_path())
            self.cur_exploit_reward = 0

