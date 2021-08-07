import json

from typing import Dict, List, Optional
from dotmap import DotMap

import numpy as np
from agents.agent import Agent
from agents.humans.human import Human, HumanAppearance
from agents.robot_agent import RobotAgent
from trajectory.trajectory import SystemConfig, Trajectory
from utils.utils import color_text, euclidean_dist2, to_json_type
from matplotlib import pyplot

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

    def render(self, ax: pyplot.Axes, p: DotMap) -> None:
        # get number of pixels-per-meter based off the ax plot space
        img_scale = ax.transData.transform((0, 1)) - ax.transData.transform((0, 0))
        ppm: int = int(img_scale[1])  # pixels per meter off image scale
        ms: float = self.radius * ppm  # markersize
        x, y, th = self.current_config.position_and_heading_nk3(squeeze=True)
        traj_col = p.traj_col if p.traj_col else self.color  # for overwriting
        start_pos3 = self.start_config.position_and_heading_nk3(squeeze=True)
        goal_pos3 = self.goal_config.position_and_heading_nk3(squeeze=True)
        start_x, start_y, start_th = start_pos3
        goal_x, goal_y, goal_th = goal_pos3

        # draw trajectory
        if p.plot_trajectory and self.trajectory is not None:
            self.trajectory.render(
                ax,
                freq=p.traj_freq,
                color=traj_col,
                alpha=p.traj_alpha,
                plot_quiver=False,
                clip=p.max_traj_length,
                marker=p.traj_marker,
                linewidth=p.traj_width,
                zorder=1,
            )

        # draw agent body
        ax.plot(
            x, y, p.normal_color, markersize=ms, label=p.label, alpha=p.alpha, zorder=2
        )

        # make the agent change colour when collided
        if self.collided or self.collision_cooldown > 0:
            ax.plot(x, y, p.collision_color, markersize=ms, alpha=p.alpha, zorder=2)
            ax.plot(x, y, p.normal_color, markersize=ms * 0.4, label=None, zorder=3)

        # draw start config
        if p.plot_start:
            ax.plot(
                start_x,
                start_y,
                p.start_col,
                markersize=ms,
                label=p.start_label,
                alpha=1,
                zorder=2,
            )

        # draw goal config
        if p.plot_goal:
            ax.plot(
                goal_x,
                goal_y,
                p.goal_col,
                markersize=2 * ms,
                marker="*",
                label=p.goal_label,
                alpha=1,
                zorder=1,
            )

        # draw quiver (heading arrow)
        if p.plot_quiver:

            s = 0.5  # scale

            def plot_quiver(xpos: float, ypos: float, theta: float) -> None:
                ax.quiver(
                    xpos,
                    ypos,
                    s * np.cos(theta),
                    s * np.sin(theta),
                    scale=1,
                    scale_units="xy",
                    zorder=1,  # behind the agent body
                )

            plot_quiver(x, y, th)  # plot agent body quiver

            # plot start quiver
            if p.plot_start and start_pos3 is not None:
                plot_quiver(start_x, start_y, start_th)

            # plot goal quiver
            if p.plot_goal and goal_pos3 is not None:
                plot_quiver(goal_x, goal_y, goal_th)

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
        robots: Optional[Dict[str, RobotAgent]] = None,
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
        self.pedestrians: Dict[str, AgentState] = pedestrians
        self.robots: Dict[str, AgentState] = robots
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

    def render(self, ax: pyplot.Axes, p: DotMap) -> None:
        """NOTE: this only renders the topview (schematic mode)"""
        """for the rgb & depth views we use the SocNavRenderer"""
        # Compute the real_world extent (in meters) of the traversible
        map_scale = self.environment["map_scale"]
        traversible = self.environment["map_traversible"]
        ax.set_xlim(0.0, traversible.shape[1] * map_scale)
        ax.set_ylim(0.0, traversible.shape[0] * map_scale)
        human_traversible = None
        if "human_traversible" in self.environment and p.draw_human_traversible:
            assert p.render_3D
            human_traversible = self.environment["human_traversible"]
        extent = (
            np.array([0.0, traversible.shape[1], 0.0, traversible.shape[0]]) * map_scale
        )
        # plot the map traversible
        ax.imshow(
            traversible, extent=extent, cmap="gray", vmin=-0.5, vmax=1.5, origin="lower"
        )

        if human_traversible is not None:  # plot human traversible
            # NOTE: the human radius is only available given the openGL human modeling
            # Plot the 5x5 meter human radius grid atop the environment traversible
            alphas = np.empty(np.shape(human_traversible))
            for y in range(human_traversible.shape[1]):
                for x in range(human_traversible.shape[0]):
                    alphas[x][y] = not (human_traversible[x][y])
            ax.imshow(
                human_traversible,
                extent=extent,
                cmap="autumn_r",
                vmin=-0.5,
                vmax=1.5,
                origin="lower",
                alpha=alphas,
            )
            # alphas = np.all(np.logical_not(human_traversible))

        for human in self.pedestrians.values():
            human.render(ax, p.human_render_params)

        for robot in self.robots.values():
            robot.render(ax, p.robot_render_params)

        # plot a small tick in the bottom left corner of schematic showing
        # how long a real world meter would be in the simulator world
        if p.plot_meter_tick:
            # plot other useful informational visuals in the topview
            # such as the key to the length of a "meter" unit
            plot_line_loc = self.environment["room_center"][:2] * 0.65
            start = [0, 0] + plot_line_loc
            end = [1, 0] + plot_line_loc
            gather_xs = [start[0], end[0]]
            gather_ys = [start[1], end[1]]
            col = "k-"
            h = 0.1  # height of the "ticks" of the key
            ax.plot(gather_xs, gather_ys, col)  # main line
            ax.plot(
                [start[0], start[0]], [start[1] + h, start[1] - h], col
            )  # tick left
            ax.plot([end[0], end[0]], [end[1] + h, end[1] - h], col)  # tick right
            if p.plot_quiver:
                ax.text(
                    0.5 * (start[0] + end[0]) - 0.2,
                    start[1] + 0.5,
                    "1m",
                    fontsize=14,
                    verticalalignment="top",
                )
        if len(self.robots) > 0 or len(self.pedestrians) > 0:
            # ensure no duplicate labels occur
            handles, labels = ax.get_legend_handles_labels()
            unique = [
                (h, l)
                for i, (h, l) in enumerate(zip(handles, labels))
                if l not in labels[:i]
            ]
            ax.legend(*zip(*unique))


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
