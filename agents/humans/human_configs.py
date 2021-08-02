from typing import Any, Dict

from trajectory.trajectory import SystemConfig
from utils.utils import *


class HumanConfigs:
    # NOTE: these are primarily used for the "initial" configs of the Human/Agent
    # and the generation of the configs from the environment
    def __init__(self, start_config: SystemConfig, goal_config: SystemConfig):
        self.start_config: SystemConfig = start_config
        self.goal_config: SystemConfig = goal_config

    # Getters for the HumanConfigs class
    def get_start_config(self) -> SystemConfig:
        return self.start_config

    def get_goal_config(self) -> SystemConfig:
        return self.goal_config

    @staticmethod
    def generate_human_config(start_config: SystemConfig, goal_config: SystemConfig):
        """
        Sample a new random human from all required features
        return HumanConfigs(start_config, goal_config)
        """
        return HumanConfigs(start_config, goal_config)

    @staticmethod
    def generate_random_human_config_from_start(
        start_config: SystemConfig, environment: Dict[str, Any]
    ):
        """
        Generate a human with a random goal config given a known start
        config. The generated start config will be near center by a threshold
        """
        goal_config: SystemConfig = generate_random_config(environment)
        return HumanConfigs.generate_human_config(start_config, goal_config)

    @staticmethod
    def generate_random_human_config_with_goal(
        goal_config: SystemConfig, environment: Dict[str, Any]
    ):
        """
        Generate a human with a random start config given a known goal
        config. The generated start config will be near center by a threshold
        """
        start_config: SystemConfig = generate_random_config(environment)
        return HumanConfigs.generate_human_config(start_config, goal_config)

    @staticmethod
    def generate_random_human_config(environment: Dict[str, Any]):
        """
        Generate a random human config (both start and goal configs) from
        the given environment
        """
        start_config: SystemConfig = generate_random_config(environment)
        goal_config: SystemConfig = generate_random_config(environment)
        return HumanConfigs.generate_human_config(start_config, goal_config)
