from configs.Config import Config

from env.observations import SimpleObservation, ShortPathObs
from env.DeadlockChecker import DeadlockChecker
from env.GreedyChecker import GreedyChecker

class ObsBuilderConfig(Config):
    def create_builder(self):
        pass


class SimpleObservationConfig(ObsBuilderConfig):
    def __init__(self, max_depth, neighbours_depth, timetable_config):
        self.max_depth = max_depth
        self.neighbours_depth = neighbours_depth
        self.timetable_config = timetable_config

    def create_builder(self):
        return SimpleObservation(
                    max_depth=self.max_depth,
                    neighbours_depth=self.neighbours_depth,
                    timetable=self.timetable_config.create_timetable(),
                    deadlock_checker=DeadlockChecker(),
                    greedy_checker=GreedyChecker(),
                )


class ShortPathObsConfig(ObsBuilderConfig):
    def __init__(self):
        pass

    def create_builder(self):
        return ShortPathObs()
