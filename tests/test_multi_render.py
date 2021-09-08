import random
from dotmap import DotMap
from params.central_params import (
    create_episodes_params,
    create_socnav_params,
    get_seed,
)
from simulators.simulator import Simulator

# seed the random number generator
random.seed(get_seed())


def render_multi_robot() -> None:
    """NOTE: this does NOT actually run any simulations, instead it takes historical data"""
    """that was previously run and is saved in tests/socnav/ and renders all the corresponding"""
    """algorithms overlayed on top of each other for easier comparison"""
    p: DotMap = create_socnav_params()  # used to instantiate the camera and its parameters
    p.episode_params = create_episodes_params()
    for test_name in list(p.episode_params.tests.keys()):
        episode: DotMap = p.episode_params.tests[test_name]
        simulator = Simulator(environment=None, renderer=None, episode_params=episode)
        simulator.params.render_params.draw_parallel_robots = True  # force true
        simulator.render(renderer=None, filename="multi_robot_{}_obs".format(test_name))


if __name__ == "__main__":
    render_multi_robot()
