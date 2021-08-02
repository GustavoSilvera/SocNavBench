from agents.robot_agent import RobotAgent
from trajectory.trajectory import SystemConfig, Trajectory
import numpy as np
import json
from utils.utils import generate_config_from_pos_3, euclidean_dist2
from utils.utils import color_red, color_reset
from typing import Any, List, Dict, Optional
from agents.agent import Agent
from agents.humans.human import Human
from agents.humans.human_appearance import HumanAppearance

""" These are smaller "wrapper" classes that are visible by other
gen_agents/humans and saved during state deepcopies
NOTE: they are all READ-ONLY (only getters)
"""


class AgentState():
    def __init__(self, a: Agent = None, name: str = None, goal_config: SystemConfig = None, start_config: SystemConfig = None,
                 current_config: SystemConfig = None, trajectory: Trajectory = None, collided: bool = False, end_acting: bool = False,
                 collision_cooldown: int = -1, radius: int = 0, color: str = None):
        """Initialize an AgentState with either an Agent instance (a) or all the individual fields"""
        if a is not None:
            self.name: str = a.get_name()
            self.goal_config: SystemConfig = a.get_goal_config()
            # TODO: get start/current configs from self.trajectory
            self.start_config: SystemConfig = a.get_start_config()
            self.current_config: SystemConfig = a.get_current_config()
            # deepcopying the trajectory to not have memory aliasing
            # for multiple sim-states spanning a wide timerange
            self.trajectory: Trajectory = a.get_trajectory(deepcpy=True)
            self.collided: bool = a.get_collided()
            self.end_acting: bool = a.get_end_acting()
            self.collision_cooldown: int = a.get_collision_cooldown()
            self.radius: float = a.get_radius()
            self.color: str = a.get_color()
        else:
            self.name: str = name
            self.goal_config: SystemConfig = goal_config
            self.start_config: SystemConfig = start_config
            self.current_config: SystemConfig = current_config
            self.trajectory: Trajectory = trajectory
            self.collided: bool = collided
            self.end_acting: bool = end_acting
            self.collision_cooldown: int = collision_cooldown
            self.radius: float = radius
            self.color: str = color

    def get_name(self) -> str:
        return self.name

    def get_current_config(self) -> SystemConfig:
        return self.current_config

    def get_start_config(self) -> SystemConfig:
        return self.start_config

    def get_goal_config(self) -> SystemConfig:
        return self.goal_config

    def get_trajectory(self) -> Trajectory:
        return self.trajectory

    def get_collided(self) -> bool:
        return self.collided

    def get_radius(self) -> float:
        return self.radius

    def get_color(self) -> str:
        return self.color

    def get_collision_cooldown(self) -> bool:
        return self.collision_cooldown

    def get_pos3(self) -> np.ndarray:
        return self.get_current_config().to_3D_numpy()

    def to_json(self, include_start_goal: Optional[bool] = False) -> Dict[str, str]:
        name_json = SimState.to_json_type(self.name)
        # NOTE: the configs are just being serialized with their 3D positions
        if include_start_goal:
            start_json = SimState.to_json_type(
                self.get_start_config().to_3D_numpy())
            goal_json = SimState.to_json_type(
                self.get_goal_config().to_3D_numpy())
        current_json = SimState.to_json_type(
            self.get_current_config().to_3D_numpy())
        # trajectory_json = "None"
        radius_json = self.radius
        json_dict: Dict[str, str] = {}
        json_dict['name'] = name_json
        # NOTE: the start and goal (of the robot) are only sent when the environment is sent
        if include_start_goal:
            json_dict['start_config'] = start_json
            json_dict['goal_config'] = goal_json
        json_dict['current_config'] = current_json
        # json_dict['trajectory'] = trajectory_json
        json_dict['radius'] = radius_json
        # returns array (python list) to be json'd in_simstate
        return json_dict

    @ staticmethod
    def from_json(json_str: Dict[str, str]):
        name: str = json_str['name']
        if 'start_config' in json_str:
            start_config = \
                generate_config_from_pos_3(json_str['start_config'])
        else:
            start_config = None
        if 'goal_config' in json_str:
            goal_config = \
                generate_config_from_pos_3(json_str['goal_config'])
        else:
            goal_config = None
        current_config = \
            generate_config_from_pos_3(json_str['current_config'])
        trajectory = None  # unable to recreate trajectory
        radius = json_str['radius']
        collision_cooldown = -1
        collided = False
        end_acting = False
        color = None
        return AgentState(None, name, goal_config, start_config, current_config,
                          trajectory, collided, end_acting, collision_cooldown,
                          radius, color)


class HumanState(AgentState):
    def __init__(self, human: Human):
        self.appearance: HumanAppearance = human.get_appearance()
        # Initialize the agent state class
        super().__init__(a=human)

    def get_appearance(self) -> HumanAppearance:
        return self.appearance


class SimState():
    def __init__(self, environment: Optional[Dict[str, float or int or np.ndarray]] = None, pedestrians: Optional[Dict[str, Agent]] = None,
                 robots: Dict[str, RobotAgent] = None, sim_t: Optional[float] = None, wall_t: Optional[float] = None,
                 delta_t: Optional[float] = None, episode_name: Optional[str] = None, max_time: Optional[float] = None,
                 ped_collider: Optional[str] = ""):
        self.environment: Dict[str, float or int or np.ndarray] = environment
        # no distinction between prerecorded and auto agents
        # new dict that the joystick will be sent
        self.pedestrians: Dict[str, Agent] = pedestrians
        self.robots: Dict[str, RobotAgent] = robots
        self.sim_t: float = sim_t
        self.wall_t: float = wall_t
        self.delta_t: float = delta_t
        self.robot_on: bool = True  # TODO: why keep this if not using explicitly?
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

    def get_robot(self, index: Optional[int] = 0, name: Optional[str] = None) -> RobotAgent:
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

    def get_all_agents(self, include_robot: Optional[bool] = False) -> Dict[str, Agent or RobotAgent]:
        all_agents = {}
        all_agents.update(self.get_pedestrians())
        if include_robot:
            all_agents.update(self.get_robots())
        return all_agents

    def to_json(self, robot_on: Optional[bool] = True, send_metadata: Optional[bool] = False, termination_cause: Optional[str] = None) -> str:
        json_dict: Dict[str, float or int or np.ndarray] = {}
        json_dict['robot_on'] = robot_on  # true or false
        sim_t_json = self.get_sim_t()
        if robot_on:  # only send the world if the robot is ON
            if send_metadata:
                environment_json = \
                    SimState.to_json_dict(self.get_environment())
                episode_json = self.get_episode_name()
                episode_max_time_json = self.get_episode_max_time()
            else:
                environment_json = {}  # empty dictionary
                episode_json = {}
                episode_max_time_json = {}
            # serialize all other fields
            ped_json = \
                SimState.to_json_dict(self.get_pedestrians())
            # NOTE: the robot only includes its start/goal posn if sending metadata
            robots_json = \
                SimState.to_json_dict(self.get_robots(),
                                      include_start_goal=send_metadata)
            delta_t_json = self.get_delta_t()
            # append them to the json dictionary
            json_dict['environment'] = environment_json
            json_dict['pedestrians'] = ped_json
            json_dict['robots'] = robots_json
            json_dict['delta_t'] = delta_t_json
            json_dict['episode_name'] = episode_json
            json_dict['episode_max_time'] = episode_max_time_json
        else:
            json_dict['termination_cause'] = termination_cause
        # sim_state should always have time
        json_dict['sim_t'] = sim_t_json
        return json.dumps(json_dict, indent=1)

    @ staticmethod
    def init_agent_dict(json_str_dict: Dict[str, Dict[str, AgentState]]) -> Dict[str, Dict[str, AgentState]]:
        agent_dict: Dict[str, AgentState] = {}
        for d in json_str_dict.keys():
            agent_dict[d] = AgentState.from_json(json_str_dict[d])
        return agent_dict

    @ staticmethod
    def from_json(json_str: Dict[str, str or int or float]):
        new_state = SimState()
        new_state.environment = json_str['environment']
        new_state.pedestrians = \
            SimState.init_agent_dict(json_str['pedestrians'])
        new_state.robots = SimState.init_agent_dict(json_str['robots'])
        new_state.sim_t = json_str['sim_t']
        new_state.delta_t = json_str['delta_t']
        new_state.robot_on = json_str['robot_on']
        new_state.episode_name = json_str['episode_name']
        new_state.episode_max_time = json_str['episode_max_time']
        new_state.wall_t = None
        new_state.ped_collider = ""
        return new_state

    @ staticmethod
    def to_json_type(elem: Any, include_start_goal=False) -> int or str or Dict:
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            # recursive for dictionaries within dictionaries
            return SimState.to_json_dict(elem, include_start_goal=include_start_goal)
        if isinstance(elem, AgentState):
            return elem.to_json(include_start_goal=include_start_goal)
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return str(elem)

    @ staticmethod
    def to_json_dict(param_dict: Dict[str, Any], include_start_goal: Optional[bool] = False) -> Dict[str, int or str or AgentState]:
        """ Converts params_dict to a json serializable dict."""
        json_dict: Dict[str, int or str or AgentState] = {}
        for key in param_dict.keys():
            json_dict[key] = SimState.to_json_type(param_dict[key],
                                                   include_start_goal=include_start_goal)
        return json_dict


"""BEGIN SimState utils"""


def get_all_agents(sim_state: Dict[float, SimState], include_robot: Optional[bool] = False) -> Dict[str, Agent or RobotAgent]:
    all_agents = {}
    all_agents.update(get_agents_from_type(sim_state, "pedestrians"))
    if include_robot:
        all_agents.update(get_agents_from_type(sim_state, "robots"))
    return all_agents


def get_agents_from_type(sim_state: SimState, agent_type: str) -> Dict[str, Agent]:
    if callable(getattr(sim_state, 'get_' + agent_type, None)):
        getter_agent_type = getattr(sim_state, 'get_' + agent_type, None)
        return getter_agent_type()
    return {}  # empty dict


def compute_next_vel(sim_state_prev: SimState, sim_state_now: SimState, agent_name: str) -> float:
    old_agent = sim_state_prev.get_all_agents()[agent_name]
    old_pos = old_agent.get_current_config().to_3D_numpy()
    new_agent = sim_state_now.get_all_agents()[agent_name]
    new_pos = new_agent.get_current_config().to_3D_numpy()
    # calculate distance over time
    delta_t = sim_state_now.get_sim_t() - sim_state_prev.get_sim_t()
    return euclidean_dist2(old_pos, new_pos) / delta_t


def compute_agent_state_velocity(sim_states: List[SimState], agent_name: str) -> List[float]:
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
            print("%sAgent" % color_red, agent_name,
                  "is not in the SimStates%s" % color_reset)
    else:
        return []


def compute_agent_state_acceleration(sim_states: List[SimState], agent_name: str, velocities: Optional[List[float]] = None) -> List[float]:
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
            print("%sAgent" % color_red, agent_name,
                  "is not in the SimStates%s" % color_reset)
    else:
        return []


def compute_all_velocities(sim_states: List[SimState]) -> Dict[str, float]:
    all_velocities = {}
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert(isinstance(agent_name, str))  # keyed by name
        all_velocities[agent_name] = \
            compute_agent_state_velocity(sim_states, agent_name)
    return all_velocities


def compute_all_accelerations(sim_states: List[SimState]) -> Dict[str, float]:
    all_accels = {}
    # TODO: add option of providing precomputed velocities list
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert(isinstance(agent_name, str))  # keyed by name
        all_accels[agent_name] = compute_agent_state_acceleration(
            sim_states, agent_name)
    return all_accels
