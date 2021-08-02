from dotmap import DotMap
from trajectory.trajectory import SystemConfig
from agents.humans.human_appearance import HumanAppearance
from agents.humans.human_configs import HumanConfigs
from utils.utils import generate_name, generate_config_from_pos_3
from agents.agent import Agent
import numpy as np
from typing import Any, Dict, List, Optional

# TODO: combine appearance & configs class to a single file for clarity


class Human(Agent):
    def __init__(self, name: str, appearance: HumanAppearance, start_configs: SystemConfig):
        self.name: str = name
        self.appearance: HumanAppearance = appearance
        super().__init__(start_configs.get_start_config(),
                         start_configs.get_goal_config(), name)

    # Getters for the Human class
    # NOTE: most of the dynamics/configs implementation is in Agent.py

    def get_appearance(self) -> HumanAppearance:
        return self.appearance

    @staticmethod
    def generate_human(appearance: HumanAppearance, configs: SystemConfig, name: Optional[str] = None, max_chars: Optional[int] = 20, verbose: Optional[bool] = False):
        """
        Sample a new random human from all required features
        """
        human_name: str = name if name is not None else generate_name(
            max_chars)
        if verbose:
            # In order to print more readable arrays
            np.set_printoptions(precision=2)
            pos_2 = (configs.get_start_config().position_nk2())[0][0]
            goal_2 = (configs.get_goal_config().position_nk2())[0][0]
            print(" Human", human_name, "at", pos_2, "with goal", goal_2)
        return Human(human_name, appearance, configs)

    @staticmethod
    def generate_human_with_appearance(appearance: HumanAppearance,
                                       environment: Dict[str, Any]):
        """
        Sample a new human with a known appearance at a random 
        config with a random goal config.
        """
        configs = HumanConfigs.generate_random_human_config(environment)
        return Human.generate_human(appearance, configs)

    @staticmethod
    def generate_human_with_configs(configs: HumanConfigs, generate_appearance: Optional[bool] = False, name: Optional[str] = None, verbose: Optional[bool] = False):
        """
        Sample a new random from known configs and a randomized
        appearance, if any of the configs are None they will be generated
        """
        if generate_appearance:
            appearance = \
                HumanAppearance.generate_rand_human_appearance(HumanAppearance)
        else:
            appearance = None
        return Human.generate_human(appearance, configs, verbose=verbose, name=name)

    @staticmethod
    def generate_random_human_from_environment(environment: Dict[str, Any],
                                               generate_appearance: Optional[bool] = False):
        """
        Sample a new human without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        appearance = None
        if generate_appearance:
            appearance = \
                HumanAppearance.generate_rand_human_appearance(HumanAppearance)
        configs = HumanConfigs.generate_random_human_config(environment)
        return Human.generate_human(appearance, configs)

    @staticmethod
    def generate_humans(p: DotMap, starts: List[List[float]], goals: List[List[float]]) -> List[Agent]:
        """
        Generate and add num_humans number of randomly generated humans to the simulator
        """
        num_gen_humans: int = min(len(starts), len(goals))
        print("Generating auto humans:", num_gen_humans)

        generated_humans: List[Agent] = []
        for i in range(num_gen_humans):
            start_config = generate_config_from_pos_3(starts[i])
            goal_config = generate_config_from_pos_3(goals[i])
            start_goal_configs = HumanConfigs(start_config, goal_config)
            human_i_name = "auto_%04d" % i
            # Generates a random human from the environment
            new_human_i = Human.generate_human_with_configs(
                start_goal_configs,
                generate_appearance=p.render_3D,
                name=human_i_name
            )
            # Input human fields into simulator
            generated_humans.append(new_human_i)
        return generated_humans
