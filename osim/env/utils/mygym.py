from gymnasium import spaces
import numpy as np


class Specification:
    timestep_limit = None

    def __init__(self, timestep_limit):
        self.timestep_limit = timestep_limit


def convert_to_gym(space):
    return spaces.Box(
        low=np.array(space[0], dtype=np.float32),
        high=np.array(space[1], dtype=np.float32),
        dtype=np.float32,
    )


def gymify_env(env):
    env.action_space = convert_to_gym(env.action_space)
    env.observation_space = convert_to_gym(env.observation_space)

    env.spec = Specification(env.timestep_limit)
    env.spec.action_space = env.action_space
    env.spec.observation_space = env.observation_space

    return env
