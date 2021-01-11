import numpy as np
import torch
import time
from time import sleep
from collections import defaultdict
import itertools
import random
from logger import log, init_logger

from env.Flatland import Flatland, FlatlandWrapper
from env.Contradictions import Contradictions
from env.rewards.FakeRewardShaper import FakeRewardShaper
from env.GreedyFlatland import GreedyFlatland
from env.DeadlockChecker import DeadlockChecker
from env.GreedyChecker import GreedyChecker
from env.timetables import AllAgentLauncher, ShortestPathAgentLauncher, NetworkLoadAgentLauncher
from env.timetables.ShortestPathAgentLauncher import ConstWindowSizeGenerator, \
    LinearOnAgentNumberSizeGenerator
from env.observations import ShortPathObs, SimpleObservation
from agent.controllers.ActorCritic import ActorCritic
from agent.PPO.PPOController import PPOController
from agent.judge.Judge import Judge

from flatland.evaluators.client import FlatlandRemoteClient
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.evaluators.client import TimeoutException
from flatland.envs.agent_utils import RailAgentStatus

RANDOM_SEED = 23
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
init_logger("logdir", "tmp", use_wandb=False)


def random_action():
    return np.random.randint(0, 5)

def evaluate_remote():
    remote_client = FlatlandRemoteClient()
    my_observation_builder = SimpleObservation(max_depth=3, neighbours_depth=3,
            timetable=Judge(LinearOnAgentNumberSizeGenerator(0.03, 5), device=torch.device("cpu")),
            deadlock_checker=DeadlockChecker(), greedy_checker=GreedyChecker(), parallel=False, eval=True)
    contr = Contradictions()

    params = torch.load("generated/params.torch")
    params.neighbours_depth=my_observation_builder.neighbours_depth
    controller = PPOController(params, torch.device("cpu"))
    controller.load_controller("generated/controller.torch")
    my_observation_builder.timetable.load_judge("generated/judge.torch")

    sum_reward, sum_percent_done = 0., 0.
    for evaluation_number in itertools.count():
        time_start = time.time()
        observation, info = remote_client.env_create(obs_builder_object=my_observation_builder)
        if not observation:
            break

        local_env = FlatlandWrapper(remote_client.env, FakeRewardShaper())
        local_env.n_agents = len(local_env.agents)
        log().check_time()
        contr.reset(local_env)
        log().check_time("contradictions_reset")
        #  env_renderer = RenderTool(
            #  local_env.env,
            #  agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            #  show_debug=True,
            #  screen_height=600,
            #  screen_width=800
        #  )

        env_creation_time = time.time() - time_start

        print("Evaluation Number : {}".format(evaluation_number))

        # max_time_steps = int(4 * 2 * (env.width + env.height + n_agents/n_cities))
        time_taken_by_controller = []
        time_taken_per_step = []
        steps = 0
        done = defaultdict(lambda: False)
        while True:
            try:
                #  env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
                time_start = time.time()
                action_dict = dict()
                handles_to_ask = list()
                observation = {k: torch.tensor(v, dtype=torch.float) for k, v in observation.items() if v is not None}
                for i in range(local_env.n_agents):
                    if not done[i]:
                        if local_env.obs_builder.greedy_checker.greedy_position(i):
                            action_dict[i] = 0
                        elif i in observation:
                            handles_to_ask.append(i)

                for handle in handles_to_ask:
                    for opp_handle in local_env.obs_builder.encountered[handle]:
                        if opp_handle != -1 and opp_handle not in observation:
                            observation[opp_handle] = torch.tensor(local_env.obs_builder._get_internal(opp_handle), dtype=torch.float)

                time_taken_per_step.append(time.time() - time_start)
                time_start = time.time()

                controller_actions = controller.fast_select_actions(handles_to_ask, observation,
                        local_env.obs_builder.encountered, train=True)
                action_dict.update(controller_actions)
                action_dict = {k: local_env.transform_action(k, v) for k, v in action_dict.items()}
                action_dict = {handle: action for handle, action in action_dict.items() if action != -1}

                time_taken = time.time() - time_start
                time_taken_by_controller.append(time_taken)

                log().check_time()
                contr.start_episode()
                for h in range(len(local_env.agents)):
                    if h in action_dict:
                        a = action_dict[h]
                        if contr.is_bad(h, a):
                            action_dict[h] = 4
                        else:
                            contr.add_elem(h, a)
                    else:
                        if local_env.agents[h].status == RailAgentStatus.ACTIVE:
                            contr.add_elem(h, 4)

                contr.start_episode()
                for h in reversed(range(len(local_env.agents))):
                    if h in action_dict:
                        a = action_dict[h]
                        if contr.is_bad(h, a):
                            action_dict[h] = 4
                        else:
                            contr.add_elem(h, a)
                    else:
                        if local_env.agents[h].status == RailAgentStatus.ACTIVE:
                            contr.add_elem(h, 4)
                log().check_time("contradictions_update")

                time_start = time.time()
                observation, all_rewards, done, info = remote_client.env_step(action_dict)
                num_done = sum([1 for agent in local_env.agents if agent.status == RailAgentStatus.DONE_REMOVED])
                num_started = sum([1 for handle in range(len(local_env.agents)) if local_env.obs_builder.timetable.is_ready(handle)])

                finished_handles = [handle for handle in range(len(local_env.agents))
                        if local_env.obs_builder.timetable.ready_to_depart[handle] == 2]
                reward = torch.sum(local_env._max_episode_steps - local_env.obs_builder.timetable.end_time[finished_handles])
                reward /= len(local_env.agents) * local_env._max_episode_steps
                percent_done = float(num_done) / len(local_env.agents)
                deadlocked = int(sum(local_env.obs_builder.deadlock_checker._is_deadlocked) + 0.5)

                steps += 1
                time_taken = time.time() - time_start
                time_taken_per_step.append(time_taken)

                if done['__all__']:
                    print("Done agents {}/{}".format(num_done, len(local_env.agents)))
                    print("Started agents {}/{}".format(num_started, len(local_env.agents)))
                    print("Deadlocked agents {}/{}".format(deadlocked, len(local_env.agents)))
                    print("Reward: {}        Percent done: {}".format(reward, percent_done))
                    sum_reward += reward
                    sum_percent_done += percent_done
                    print("Total reward: {}        Avg percent done: {}".format(sum_reward, sum_percent_done / (evaluation_number + 1)))
                    #  env_renderer.close_window()
                    break
            except TimeoutException as err:
                print("Timeout! Will skip this episode and go to the next.", err)
                break

        
        np_time_taken_by_controller = np.array(time_taken_by_controller)
        np_time_taken_per_step = np.array(time_taken_per_step)
        print("="*100)
        print("="*100)
        print("Evaluation Number : ", evaluation_number)
        print("Current Env Path : ", remote_client.current_env_path)
        print("Env Creation Time : ", env_creation_time)
        print("Number of Steps : {}/{}".format(steps, local_env._max_episode_steps))
        print("Mean/Std/Sum of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std(), np_time_taken_by_controller.sum())
        print("Mean/Std/Sum of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std(), np_time_taken_per_step.sum())
        log().print_time_metrics()
        log().zero_time_metrics()
        print("="*100)
        print("\n\n")

    print("Evaluation of all environments complete...")
    print(remote_client.submit())

if __name__ == "__main__":
    evaluate_remote()

