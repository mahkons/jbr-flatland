import numpy as np
from collections import defaultdict
from flatland.envs.agent_utils import RailAgentStatus

from time import sleep
from copy import deepcopy

from env.Flatland import get_new_position
from env.observations.SimpleObservation import ObservationDecoder

class DeadlockChecker():
    def __init__(self):
        pass

    def reset(self, env):
        self._is_deadlocked = np.zeros(len(env.agents))
        self._old_deadlock = np.zeros(len(env.agents))
        self.env = env

        self.agent_positions = defaultdict(lambda: -1)

    def is_deadlocked(self, handle):
        return self._is_deadlocked[handle]

    def old_deadlock(self, handle):
        return self._old_deadlock[handle]

    def _check_blocked(self, handle):
        agent = self.env.agents[handle]
        transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        self.checked[handle] = 1

        for direction, transition in enumerate(transitions):
            if transition == 0:
                continue # no road
            
            new_position = get_new_position(agent.position, direction)
            handle_opp_agent = self.agent_positions[new_position]
            if handle_opp_agent == -1:
                self.checked[handle] = 2
                return False # road is free

            if self._is_deadlocked[handle_opp_agent]:
                continue # road is blocked

            if self.checked[handle_opp_agent] == 0:
                self._check_blocked(handle_opp_agent)

            if self.checked[handle_opp_agent] == 2 and not self._is_deadlocked[handle_opp_agent]:
                self.checked[handle] = 2
                return False # road may become free

            self.dep[handle].append(handle_opp_agent)

            continue # road is blocked. cycle

        if not self.dep[handle]:
            self.checked[handle] = 2
            if len(list(filter(lambda t: t != 0, transitions))) == 0:
                return False # dead-end is not deadlock
            self._is_deadlocked[handle] = 1
            return True
        return None # shrug


    def _fix_deps(self):
        any_changes = True
        # might be slow, but in practice won't # TODO can be optimized
        while any_changes:
            any_changes = False
            for handle, agent in enumerate(self.env.agents):
                if self.checked[handle] == 1:
                    cnt = 0
                    for opp_handle in self.dep[handle]:
                        if self.checked[opp_handle] == 2:
                            if self._is_deadlocked[opp_handle]:
                                cnt += 1
                            else:
                                self.checked[handle] = 2
                                any_changes = True
                    if cnt == len(self.dep[handle]):
                        self.checked[handle] = 2
                        self._is_deadlocked[handle] = True
                        any_changes = True
        for handle, agent in enumerate(self.env.agents):
            if self.checked[handle] == 1:
                self._is_deadlocked[handle] = True
                self.checked[handle] = 2


    def update_deadlocks(self):
        self.agent_positions.clear()
        self.dep = [list() for _ in range(len(self.env.agents))]
        for handle, agent in enumerate(self.env.agents):
            self._old_deadlock[handle] = self._is_deadlocked[handle]
            if agent.status == RailAgentStatus.ACTIVE:
                self.agent_positions[agent.position] = handle
        self.checked = [0] * len(self.env.agents)
        for handle, agent in enumerate(self.env.agents):
            if agent.status == RailAgentStatus.ACTIVE and not self._is_deadlocked[handle] \
                    and not self.checked[handle]:
                self._check_blocked(handle)

        self._fix_deps()
