import json
from typing import Dict, List, Optional

import numpy as np
from agents.agent import Agent
from agents.humans.human import Human, HumanAppearance
from agents.robot_agent import RobotAgent
from trajectory.trajectory import SystemConfig, Trajectory
from utils.utils import color_text, euclidean_dist2, to_json_type

""" These are smaller "wrapper" classes that are visible by other
gen_agents/humans and saved during state deepcopies
NOTE: they are all READ-ONLY (only getters)
"""


class AgentState:
    def __init__(
        self,
        name: str,
        goal_config: SystemConfig,
        start_config: SystemConfig,
        current_config: SystemConfig,
        trajectory: Optional[Trajectory] = None,
        appearance: Optional[HumanAppearance] = None,
        collided: bool = False,
        end_acting: bool = False,
        collision_cooldown: int = -1,
        radius: int = 0,
        color: str = None,
    ):
        """Initialize an AgentState with either an Agent instance (a) or all the individual fields"""
        self.name: str = name
        self.goal_config: SystemConfig = goal_config
        self.start_config: SystemConfig = start_config
        self.current_config: SystemConfig = current_config
        self.trajectory: Trajectory = trajectory
        self.appearance: HumanAppearance = appearance
        self.collided: bool = collided
        self.end_acting: bool = end_acting
        self.collision_cooldown: int = collision_cooldown
        self.radius: float = radius
        self.color: str = color

    @classmethod
    def from_agent(cls, a: Agent):
        appearance = None
        if isinstance(a, Human):  # only Humans have appearances
            appearance = a.get_appearance()
        return cls(
            name=a.get_name(),
            goal_config=a.get_goal_config(),
            start_config=a.get_start_config(),
            current_config=a.get_current_config(),
            trajectory=a.get_trajectory(deepcpy=True),
            appearance=appearance,
            collided=a.get_collided(),
            end_acting=a.get_end_acting(),
            collision_cooldown=a.get_collision_cooldown(),
            radius=a.get_radius(),
            color=a.get_color(),
        )

    def get_name(self) -> str:
        return self.name

    def get_current_config(self) -> SystemConfig:
        return self.current_config

    def get_start_config(self) -> SystemConfig:
        return self.start_config

    def get_goal_config(self) -> SystemConfig:
        return self.goal_config

    def get_trajectory(self) -> Optional[Trajectory]:
        return self.trajectory

    def get_appearance(self) -> HumanAppearance:
        return self.appearance

    def get_collided(self) -> bool:
        return self.collided

    def get_radius(self) -> float:
        return self.radius

    def get_color(self) -> str:
        return self.color

    def get_collision_cooldown(self) -> bool:
        return self.collision_cooldown

    def get_pos3(self) -> np.ndarray:
        return self.get_current_config().position_and_heading_nk3(squeeze=True)

    def to_json_type(self) -> Dict[str, str]:
        json_dict: Dict[str, str] = {}
        json_dict["name"] = self.name
        json_dict["start_config"] = to_json_type(
            self.get_start_config().position_and_heading_nk3(squeeze=True)
        )
        json_dict["goal_config"] = to_json_type(
            self.get_goal_config().position_and_heading_nk3(squeeze=True)
        )
        json_dict["current_config"] = to_json_type(
            self.get_current_config().position_and_heading_nk3(squeeze=True)
        )
        json_dict["radius"] = self.radius
        return json_dict

    @classmethod
    def from_json(cls, json_dict: Dict[str, str]):
        assert "name" in json_dict
        name: str = json_dict["name"]
        start_config = None
        goal_config = None
        if "start_config" in json_dict:
            start_config = SystemConfig.from_pos3(json_dict["start_config"])
        if "goal_config" in json_dict:
            goal_config = SystemConfig.from_pos3(json_dict["goal_config"])
        assert "current_config" in json_dict
        current_config = SystemConfig.from_pos3(json_dict["current_config"])
        assert "radius" in json_dict
        radius = json_dict["radius"]
        return cls(
            name=name,
            goal_config=goal_config,
            start_config=start_config,
            current_config=current_config,
            trajectory=None,
            collided=False,
            end_acting=False,
            collision_cooldown=-1,
            radius=radius,
            color=None,
        )


class SimState:
    def __init__(
        self,
        environment: Optional[Dict[str, float or int or np.ndarray]] = None,
        pedestrians: Optional[Dict[str, Agent]] = None,
        robots: Dict[str, RobotAgent] = None,
        sim_t: Optional[float] = None,
        wall_t: Optional[float] = None,
        delta_t: Optional[float] = None,
        robot_on: Optional[bool] = True,
        episode_name: Optional[str] = None,
        max_time: Optional[float] = None,
        ped_collider: Optional[str] = "",
    ):
        self.environment: Dict[str, float or int or np.ndarray] = environment
        # no distinction between prerecorded and auto agents
        # new dict that the joystick will be sent
        self.pedestrians: Dict[str, Agent] = pedestrians
        self.robots: Dict[str, RobotAgent] = robots
        self.sim_t: float = sim_t
        self.wall_t: float = wall_t
        self.delta_t: float = delta_t
        self.robot_on: bool = robot_on
        self.episode_name: str = episode_name
        self.episode_max_time: float = max_time
        self.ped_collider: str = ped_collider

    def get_environment(self) -> Dict[str, float or int or np.ndarray]:
        return self.environment

    def get_map(self) -> np.ndarray:
        return self.environment["map_traversible"]

    def get_pedestrians(self) -> Dict[str, Agent]:
        return self.pedestrians

    def get_robots(self) -> Dict[str, RobotAgent]:
        return self.robots

    def get_robot(
        self, index: Optional[int] = 0, name: Optional[str] = None
    ) -> RobotAgent:
        if name:  # index robot by name
            return self.robots[name]
        return list(self.robots.values())[index]  # index robot by posn

    def get_sim_t(self) -> float:
        return self.sim_t

    def get_wall_t(self) -> float:
        return self.wall_t

    def get_delta_t(self) -> float:
        return self.delta_t

    def get_robot_on(self) -> bool:
        return self.robot_on

    def get_episode_name(self) -> str:
        return self.episode_name

    def get_episode_max_time(self) -> float:
        return self.episode_max_time

    def get_collider(self) -> str:
        return self.ped_collider

    def get_all_agents(
        self, include_robot: Optional[bool] = False
    ) -> Dict[str, Agent or RobotAgent]:
        all_agents = {}
        all_agents.update(self.get_pedestrians())
        if include_robot:
            all_agents.update(self.get_robots())
        return all_agents

    def to_json(
        self,
        robot_on: Optional[bool] = True,
        send_metadata: Optional[bool] = False,
        termination_cause: Optional[str] = None,
    ) -> str:
        json_dict: Dict[str, float or int or np.ndarray] = {}
        json_dict["robot_on"] = to_json_type(robot_on)
        if robot_on:  # only send the world if the robot is ON
            robots_json: dict = to_json_type(self.get_robots())
            if not send_metadata:
                # NOTE: the robot(s) send their start/goal posn iff sending metadata
                for robot_name in robots_json:
                    # don't need to send the robot start & goal since those are constant
                    del robots_json[robot_name]["start_config"]
                    del robots_json[robot_name]["goal_config"]
            else:
                # only include environment and episode name iff sending metadata
                json_dict["environment"] = to_json_type(self.get_environment())
                json_dict["episode_name"] = to_json_type(self.get_episode_name())
            # append other fields to the json dictionary
            json_dict["pedestrians"] = to_json_type(self.get_pedestrians())
            json_dict["robots"] = robots_json
            json_dict["delta_t"] = to_json_type(self.get_delta_t())
            json_dict["episode_max_time"] = to_json_type(self.get_episode_max_time())
        else:
            json_dict["termination_cause"] = to_json_type(termination_cause)
        # sim_state should always have time
        json_dict["sim_t"] = to_json_type(self.get_sim_t())
        return json.dumps(json_dict, indent=1)

    @classmethod
    def from_json(cls, json_str: Dict[str, str or int or float]):
        def try_loading(key: str) -> Optional[str or int or float]:
            if key in json_str:
                return json_str[key]
            return None

        return cls(
            environment=try_loading("environment"),
            pedestrians=SimState.init_agent_dict(json_str["pedestrians"]),
            robots=SimState.init_agent_dict(json_str["robots"]),
            sim_t=json_str["sim_t"],
            wall_t=None,
            delta_t=json_str["delta_t"],
            robot_on=json_str["robot_on"],
            episode_name=try_loading("episode_name"),
            max_time=json_str["episode_max_time"],
            ped_collider="",
        )

    @staticmethod
    def init_agent_dict(
        json_str_dict: Dict[str, Dict[str, str or float or int or dict]]
    ) -> Dict[str, Dict[str, AgentState]]:
        agent_dict: Dict[str, AgentState] = {}
        for agent_name in json_str_dict.keys():
            agent_dict[agent_name] = AgentState.from_json(json_str_dict[agent_name])
        return agent_dict


"""BEGIN SimState utils"""


def get_all_agents(
    sim_state: Dict[float, SimState], include_robot: Optional[bool] = False
) -> Dict[str, Agent or RobotAgent]:
    all_agents = {}
    all_agents.update(get_agents_from_type(sim_state, "pedestrians"))
    if include_robot:
        all_agents.update(get_agents_from_type(sim_state, "robots"))
    return all_agents


def get_agents_from_type(sim_state: SimState, agent_type: str) -> Dict[str, Agent]:
    if callable(getattr(sim_state, "get_" + agent_type, None)):
        getter_agent_type = getattr(sim_state, "get_" + agent_type, None)
        return getter_agent_type()
    return {}  # empty dict


def compute_next_vel(
    sim_state_prev: SimState, sim_state_now: SimState, agent_name: str
) -> float:
    old_agent = sim_state_prev.get_all_agents()[agent_name]
    old_pos = old_agent.get_current_config().position_and_heading_nk3(squeeze=True)
    new_agent = sim_state_now.get_all_agents()[agent_name]
    new_pos = new_agent.get_current_config().position_and_heading_nk3(squeeze=True)
    # calculate distance over time
    delta_t = sim_state_now.get_sim_t() - sim_state_prev.get_sim_t()
    return euclidean_dist2(old_pos, new_pos) / delta_t


def compute_agent_state_velocity(
    sim_states: List[SimState], agent_name: str
) -> List[float]:
    if len(sim_states) > 1:  # need at least two to compute differences in positions
        if agent_name in get_all_agents(sim_states[-1]):
            agent_velocities = []
            for i in range(len(sim_states)):
                if i > 0:
                    prev_sim_s = sim_states[i - 1]
                    now_sim_s = sim_states[i]
                    speed = compute_next_vel(prev_sim_s, now_sim_s, agent_name)
                    agent_velocities.append(speed)
                else:
                    agent_velocities.append(0.0)  # initial velocity is 0
            return agent_velocities
        else:
            print(
                "%sAgent" % color_text["red"],
                agent_name,
                "is not in the SimStates%s" % color_text["reset"],
            )
    else:
        return []


def compute_agent_state_acceleration(
    sim_states: List[SimState],
    agent_name: str,
    velocities: Optional[List[float]] = None,
) -> List[float]:
    if len(sim_states) > 1:  # need at least two to compute differences in velocities
        # optionally compute velocities as well
        if velocities is None:
            velocities = compute_agent_state_velocity(sim_states, agent_name)
        if agent_name in get_all_agents(sim_states[-1]):
            agent_accels = []
            for i, this_vel in enumerate(velocities):
                if i > 0:
                    # compute delta_t between sim states
                    sim_st_now = sim_states[i]
                    sim_st_prev = sim_states[i - 1]
                    delta_t = sim_st_now.get_sim_t() - sim_st_prev.get_sim_t()
                    # compute delta_v between velocities
                    last_vel = velocities[i - 1]
                    # calculate speeds over time
                    accel = (this_vel - last_vel) / delta_t
                    agent_accels.append(accel)
                    if i == len(sim_states) - 1:
                        # last element gets no acceleration
                        break
            return agent_accels
        else:
            print(
                "%sAgent" % color_text["red"],
                agent_name,
                "is not in the SimStates%s" % color_text["reset"],
            )
    else:
        return []


def compute_all_velocities(sim_states: List[SimState]) -> Dict[str, float]:
    all_velocities = {}
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert isinstance(agent_name, str)  # keyed by name
        all_velocities[agent_name] = compute_agent_state_velocity(
            sim_states, agent_name
        )
    return all_velocities


def compute_all_accelerations(sim_states: List[SimState]) -> Dict[str, float]:
    all_accels = {}
    # TODO: add option of providing precomputed velocities list
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert isinstance(agent_name, str)  # keyed by name
        all_accels[agent_name] = compute_agent_state_acceleration(
            sim_states, agent_name
        )
    return all_accels
