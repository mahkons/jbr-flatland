import torch
#  import torch.multiprocessing as mp # Aaargh
import ray
from collections import defaultdict
from itertools import count
import time
import os

from agent.Agent import Agent
from logger import log

from flatland.envs.agent_utils import RailAgentStatus


@ray.remote
class A3CRunner():
    def __init__(self, handle, env_config, controller_config, device):
        self.handle = handle
        self.env = env_config.create_env()
        self.device = device
        self.controller = controller_config.create_controller(self.env, self.device)

        self.done = defaultdict(lambda: False)
        self.state = None
        self.all_time = 0
        self.env_step_time = 0

    def run(self, actor_params, critic_params, max_steps):
        time_start = time.time()
        self.controller.actor_net.load_state_dict(actor_params)
        self.controller.critic_net.load_state_dict(critic_params)

        if self.state == None:
            self.state = self.env.reset()
            self.done = defaultdict(lambda: False)
            self.prev_valid_state = self.state
            self.prev_valid_action = dict()
            self.prev_valid_action_info = dict()

        states, next_states, actions, rewards, dones = [], [], [], [], []
        for step in count():
            action_dict, action_info = dict(), dict()
            for i in self.state.keys():
                if self.env.agents[i].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                        and not self.env.obs_builder.deadlock_checker.is_deadlocked(i):
                    action_dict[i], action_info[i] = self.controller.select_action(handle=i,
                            state=torch.tensor(self.state[i], dtype=torch.float, device=self.device), train=True)

            self.prev_valid_action.update(action_dict)
            self.prev_valid_action_info.update(action_info)
            time_env_start = time.time()
            next_state, reward, self.done, _, _ = self.env.step(action_dict)
            self.env_step_time += time.time() - time_env_start

            for i in next_state.keys(): # if valid observation
                if not i in self.prev_valid_state: # just departed
                    continue 
                states.append(torch.tensor(self.prev_valid_state[i], dtype=torch.float, device=self.device))
                next_states.append(torch.tensor(next_state[i], dtype=torch.float, device=self.device))
                actions.append(torch.tensor(self.prev_valid_action[i], dtype=torch.long, device=self.device))
                rewards.append(torch.tensor(reward[i], dtype=torch.float, device=self.device))
                dones.append(torch.tensor(self.done[i], dtype=torch.float, device=self.device))
            self.state = next_state
            self.prev_valid_state.update(self.state)

            def get_grads(params):
                return [p.grad for p in params]

            if self.done['__all__'] or step == max_steps - 1:
                self.controller.actor_optimizer.zero_grad()
                self.controller.critic_optimizer.zero_grad()
                if states:
                    actor_loss, critic_loss = self.controller.calc_loss_batch(
                        torch.stack(states),
                        torch.stack(actions),
                        torch.stack(next_states),
                        torch.stack(rewards),
                        torch.stack(dones),
                    )
                    actor_loss.backward()
                    critic_loss.backward()

                if self.done['__all__']:
                    self.state = None

                percent_done = sum([1 for agent in self.env.agents if agent.status == RailAgentStatus.DONE_REMOVED]) / self.env.n_agents
                self.all_time += time.time() - time_start
                #  print(self.env_step_time, self.all_time, self.env_step_time/self.all_time)
                return get_grads(self.controller.actor_net.parameters()), \
                       get_grads(self.controller.critic_net.parameters()), \
                       {"handle": self.handle, "steps": step, "finished_episode": self.done['__all__'],
                               "reward": self.env.get_total_reward(),
                               "percent_done": percent_done}
            

# not exactly A3C
# blocks to update on gradients
# seems to work
class MultiAgent():
    def __init__(self, env_config, controller_config, n_workers, batch_size, device):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.env_config = env_config
        self.controller_config = controller_config
        self.device = device

        self.env = env_config.create_env()
        self.controller = controller_config.create_controller(self.env, device)

        num_gpus = 0
        if device == torch.device("cuda"):
            num_gpus = 1

        ray.init(num_gpus=num_gpus)
        self.agents = [None] * n_workers
        for handle in range(n_workers):
            self.agents[handle] = A3CRunner.options(num_gpus=num_gpus/float(n_workers)).remote(
                    handle, env_config, controller_config, device)
            env_config.update_random_seed()


    # Might do a bit more rollouts then specified
    # WARNING actual batch_size is n_agents time bigger
    def rollouts(self, max_opt_steps=10**10, max_episodes=10**10):
        actor_params, critic_params = \
            self.controller.actor_net.state_dict(), self.controller.critic_net.state_dict()

        gradient_list = [agent.run.remote(actor_params, critic_params, self.batch_size) for agent in self.agents]

        cur_steps, cur_episode = 0, 0
        while True:
            done_id, gradient_list = ray.wait(gradient_list)
            actor_gradient, critic_gradient, info = ray.get(done_id)[0]

            cur_steps += info["steps"]
            cur_episode += info["finished_episode"]
            
            if cur_steps >= max_opt_steps or cur_episode >= max_episodes:
                break
                
            if info["finished_episode"]:
                log().add_plot_point("reward", (cur_episode, cur_steps, info["reward"]))
                log().add_plot_point("percent_done", (cur_episode, cur_steps, info["percent_done"]))
                log().add_plot_point("time", (cur_episode, cur_steps, time.time()))
                print(cur_episode + 1, info["reward"])

            for param, grad in zip(self.controller.actor_net.parameters(), actor_gradient):
                param._grad = grad
            for param, grad in zip(self.controller.critic_net.parameters(), critic_gradient):
                param._grad = grad
            self.controller.actor_optimizer.step()
            self.controller.critic_optimizer.step()
            
            actor_params, critic_params = \
                self.controller.actor_net.state_dict(), self.controller.critic_net.state_dict()
            gradient_list.extend([self.agents[info["handle"]].run.remote(actor_params, critic_params, self.batch_size)])

        return
        
