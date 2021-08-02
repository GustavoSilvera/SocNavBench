import os
import json
import copy
import numpy as np
import shutil
from dotmap import DotMap
from random import random
import string
import random
import time
from trajectory.trajectory import SystemConfig
from typing import List, Dict, Tuple, Optional
from matplotlib import pyplot, figure

color_orange = '\033[33m'
color_green = '\033[32m'
color_red = '\033[31m'
color_blue = '\033[36m'
color_yellow = '\033[35m'
color_reset = '\033[00m'


def ensure_odd(integer: int) -> bool:
    if integer % 2 == 0:
        integer += 1
    return integer


def render_angle_frequency(p: DotMap) -> int:
    """Returns a render angle frequency
    that looks heuristically nice on plots."""
    return int(p.episode_horizon / 25)


def log_dict_as_json(params: DotMap, filename: str) -> None:
    """Save params (either a DotMap object or a python dictionary) to a file in json format"""
    with open(filename, 'w') as f:
        if isinstance(params, DotMap):
            params = params.toDict()
        param_dict_serializable = _to_json_serializable_dict(
            copy.deepcopy(params))
        json.dump(param_dict_serializable, f, indent=4, sort_keys=True)


def get_time_str() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def _to_json_serializable_dict(param_dict: Dict[str, str]) -> Dict[str, int or str]:
    """Converts params_dict to a json serializable dict.

    Args:
        param_dict (dict): the dictionary to be serialized
    """
    def _to_serializable_type(elem):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            return _to_json_serializable_dict(elem)
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return str(elem)
    for key in param_dict.keys():
        param_dict[key] = _to_serializable_type(param_dict[key])
    return param_dict


def euclidean_dist2(p1: List[float], p2: List[float]) -> float:
    """Compute the 2D euclidean distance from p1 to p2.

    Args:
        p1 (list): A point in a 2D space (with at least 2 dimens).
        p2 (list): Another point in a 2D space (with at least 2 dimens).

    Returns:
        dist (float): the euclidean (straight-line) distance between the points.
    """
    diff_x: float = p1[0] - p2[0]
    diff_y: float = p1[1] - p2[1]
    return np.sqrt(diff_x**2 + diff_y**2)


def absmax(x: np.ndarray) -> float or int:
    # returns maximum based off magnitude, not sign
    return max(x.min(), x.max(), key=abs)


def touch(path: str) -> None:
    """Creates an empty file at a specific file location

    Args:
        path (str): The absolute path for the location of the new file
    """
    basedir: str = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(path, 'a'):
        os.utime(path, None)


def natural_sort(l: List[float or int]) -> List[str or int]:
    """Sorts a list of items naturally.

    Args:
        l (list): the list of elements to sort. 

    Returns:
        A naturally sorted list with the same elements as l
    """
    import re

    def convert(text: str) -> int or str:
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: str) -> List[int or str]:
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def generate_name(max_chars: int) -> str:
    """Creates a string of max_chars random characters.

    Args:
        max_chars (int): number of characters in random string (name).

    Returns:
        A string of length max_chars with random ascii characters
    """
    return "".join([
        random.choice(string.ascii_letters + string.digits)
        for _ in range(max_chars)
    ])


def conn_recv(connection, buffr_amnt: int = 1024) -> Tuple[bytes, int]:
    """Makes sure all the data from a socket connection is correctly received

    Args:
        connection: The socket connection used as a communication channel.
        buffr_amnt (int, optional): Amount of bytes to transfer at a time. Defaults to 1024.

    Returns:
        data (bytes): The data received from the socket.
        response_len (int): The number of bytes that were transferred
    """
    chunks: List[bytes] = []
    response_len: int = 0
    while True:
        chunk = connection.recv(buffr_amnt)
        if chunk == b'':
            break
        chunks.append(chunk)
        response_len += len(chunk)
    data: bytes = b''.join(chunks)
    return data, response_len


def mkdir_if_missing(dirname: str) -> None:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def delete_if_exists(dirname: str) -> None:
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def check_dotmap_equality(d1: DotMap, d2: DotMap) -> bool:
    """Check equality on nested map objects that all keys and values match."""
    assert(len(set(d1.keys()).difference(set(d2.keys()))) == 0)
    equality: List[bool] = [True] * len(d1.keys())
    for i, key in enumerate(d1.keys()):
        d1_attr = getattr(d1, key)
        d2_attr = getattr(d2, key)
        if type(d1_attr) is DotMap:
            equality[i] = check_dotmap_equality(d1_attr, d2_attr)
    return np.array(equality).all()


def configure_plotting() -> None:
    pyplot.plot.style.use('ggplot')


def subplot2(plt: pyplot.plot, Y_X: Tuple[int, int], sz_y_sz_x: Optional[Tuple[int, int]] = (10, 10), space_y_x: Optional[Tuple[int, int]] = (0.1, 0.1), T: Optional[bool] = False) -> Tuple[figure.Figure, pyplot.axes, List[pyplot.axes]]:
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams['figure.figsize'] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes, axes_list


def termination_cause_to_color(cause: str) -> Optional[str]:
    cause_colour_mappings: Dict[str, str] = {
        "Success": "green",
        "Pedestrian Collision": "red",
        "Obstacle Collision": "orange",
        "Timeout": "blue",
    }
    if cause in cause_colour_mappings:
        return cause_colour_mappings[cause]
    return None


def color_print(color: str) -> str:
    colour_map: Dict[str, str] = {
        "green": color_green,
        "red": color_red,
        "blue": color_blue,
        "yellow": color_yellow,
        "orange": color_orange,
    }
    if color in colour_map:
        return colour_map[color]
    return color_reset  # default is no colour


def iter_print(l: List or Dict) -> str:
    if isinstance(l[0], float):
        return ','.join(["{0: 0.2f}".format(i) for i in l])
    # return string
    return ','.join([str(i) for i in l])


""" BEGIN configs functions """


def generate_config_from_pos_3(pos_3: np.ndarray, dt: Optional[float] = 0.1, v: Optional[float] = 0, w: Optional[float] = 0) -> SystemConfig:
    pos_n11 = np.array([[[pos_3[0], pos_3[1]]]], dtype=np.float32)
    heading_n11 = np.array([[[pos_3[2]]]], dtype=np.float32)
    speed_nk1 = np.ones((1, 1, 1), dtype=np.float32) * v
    angular_speed_nk1 = np.ones((1, 1, 1), dtype=np.float32) * w
    return SystemConfig(dt, 1, 1,
                        position_nk2=pos_n11,
                        heading_nk1=heading_n11,
                        speed_nk1=speed_nk1,
                        angular_speed_nk1=angular_speed_nk1,
                        variable=False)


def generate_random_config(environment: Dict[str, int or float or np.ndarray], dt: Optional[float] = 0.1, max_vel: Optional[float] = 0.6) -> SystemConfig:
    pos_3: np.ndarray = generate_random_pos_in_environment(environment)
    return generate_config_from_pos_3(pos_3, dt=dt, v=max_vel)


def generate_random_pos_3(center: np.ndarray, xdiff: Optional[float] = 3.0, ydiff: Optional[float] = 3.0) -> np.ndarray:
    """
    Generates a random position near the center within an elliptical radius of xdiff and ydiff
    """
    offset_x = 2 * xdiff * random.random() - xdiff  # bound by (-xdiff, xdiff)
    offset_y = 2 * ydiff * random.random() - ydiff  # bound by (-ydiff, ydiff)
    offset_theta = 2 * np.pi * random.random()  # bound by (0, 2*pi)
    return np.add(center, np.array([offset_x, offset_y, offset_theta]))


def within_traversible(new_pos: np.ndarray, traversible: np.ndarray, map_scale: float,
                       stroked_radius: Optional[bool] = False) -> bool:
    """
    Returns whether or not the position is in a valid spot in the
    traversible
    """
    pos_x = int(new_pos[0] / map_scale)
    pos_y = int(new_pos[1] / map_scale)
    # Note: the traversible is mapped unintuitively, goes [y, x]
    try:
        if (not traversible[pos_y][pos_x]):  # Looking for invalid spots
            return False
        return True
    except:
        return False


def within_traversible_with_radius(new_pos: np.ndarray, traversible: np.ndarray, map_scale: float, radius: Optional[int] = 1,
                                   stroked_radius: Optional[bool] = False) -> bool:
    """
    Returns whether or not the position is in a valid spot in the
    traversible the Radius input can determine how many surrounding
    spots must also be valid
    """
    # TODO: use np vectorizing instead of double for loops
    for i in range(2 * radius):
        for j in range(2 * radius):
            if stroked_radius:
                if not((i == 0 or i == radius - 1 or j == 0 or j == radius - 1)):
                    continue
            pos_x = int(new_pos[0] / map_scale) - radius + i
            pos_y = int(new_pos[1] / map_scale) - radius + j
            # Note: the traversible is mapped unintuitively, goes [y, x]
            if (not traversible[pos_y][pos_x]):  # Looking for invalid spots
                return False
    return True


def generate_random_pos_in_environment(environment: Dict[str, int or float or np.ndarray]) -> np.ndarray:
    """
    Generate a random position (x : meters, y : meters, theta : radians)
    and near the 'center' with a nearby valid goal position.
    - Note that the obstacle_traversible and human_traversible are both
    checked to generate a valid pos_3.
    - Note that the "environment" holds the map scale and all the
    individual traversibles if they exists
    - Note that the map_scale primarily refers to the traversible's level
    of precision, it is best to use the dx_m provided in examples.py
    """
    map_scale = float(environment["map_scale"])
    # Combine the occupancy information from the static map and the human
    if "human_traversible" in environment:
        # in this case there exists a "human" traversible as well, and we
        # don't want to generate one human in the traversible of another
        global_traversible = np.empty(environment["map_traversible"].shape)
        global_traversible.fill(True)
        map_t = environment["map_traversible"]
        human_t = environment["human_traversible"]
        # append the map traversible
        global_traversible = np.stack([global_traversible, map_t], axis=2)
        global_traversible = np.all(global_traversible, axis=2)
        # stack the human traversible on top of the map one
        global_traversible = np.stack([global_traversible, human_t], axis=2)
        global_traversible = np.all(global_traversible, axis=2)
    else:
        global_traversible = environment["map_traversible"]

    # Generating new position as human's position
    pos_3 = np.array([0, 0, 0])  # start far out of the traversible

    # continuously generate random positions near the center until one is valid
    while not within_traversible(pos_3, global_traversible, map_scale):
        new_x = random.randint(0, global_traversible.shape[0])
        new_y = random.randint(0, global_traversible.shape[1])
        new_theta = 2 * np.pi * random.random()  # bound by (0, 2*pi)
        pos_3 = np.array([new_x, new_y, new_theta])

    return pos_3


""" END configs functions """
