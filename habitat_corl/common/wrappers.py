from copy import deepcopy

import gym
from gym import ObservationWrapper, ActionWrapper

import numpy as np
from typing import Union


class HabitatWrapper(ObservationWrapper):
    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation = self.env.step(action)
        return self.observation(observation)

class HabitatAcionWrapper(gym.ActionWrapper):
    def __init__(self, env, ignore_stop=False):
        super().__init__(env)
        self.ignore_stop = ignore_stop
        if ignore_stop:
            self.action_space = gym.spaces.Discrete(
                self.action_space.n - 1
            )

    def action(self, action):
        if self.ignore_stop:
            action = action + 1
        return action
class ContinuousActionWrapper(ActionWrapper):
    def __init__(self, env, turn_angle=30):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.turn_angle = turn_angle
    def _quat_to_xy_heading(self, quat):
        from habitat.utils.geometry_utils import quaternion_rotate_vector
        from habitat.tasks.utils import cartesian_to_polar
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def _get_heading(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        agent_state = env.sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        return self._quat_to_xy_heading(rotation_world_agent.inverse())

    def action(self, action):
        # action = action[0]
        # # clip action to [0, 1]
        # action = np.clip(action, 0, 1)
        # target_angle = action * (2 * np.pi)

        # cartesian to radian angle
        target_angle = np.arctan2(action[1], action[0])
        target_angle += np.pi

        current_heading = self._get_heading() + np.pi

        # tolerance = 7.5 degrees, in radians
        tolerance = (self.turn_angle/2.0) * np.pi / 180

        if target_angle > current_heading:
            # if target angel is greater than current heading,
            # angle while turning left does not wrap around 2pi
            left_angle = target_angle - current_heading
            # angle while turning right does wrap around 2pi
            right_angle = current_heading + (2 * np.pi - target_angle)
        else:
            # if target angle is less than current heading,
            # angle while turning right does not wrap around 2pi
            right_angle = current_heading - target_angle
            # angle while turning left does wrap around 2pi
            left_angle = target_angle + (2 * np.pi - current_heading)

        # if the current heading is within tolerance of the target angle,
        # move forward = 1
        if left_angle < tolerance or right_angle < tolerance:
            return 1

        # if angle to turn left is less than angle to turn right,
        # turn left = 2
        if left_angle < right_angle:
            return 2

        # if angle to turn right is less than angle to turn left,
        # turn right = 3
        if left_angle > right_angle:
            return 3

    def step(self, action):
        if isinstance(action, int) and action <= 0:
            obs = self.env.step(0)
        else:
            obs = self.env.step(self.action(action))

        return obs



def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
    used_inputs=None,
    ignore_stop=False,
    continuous=False,
    turn_angle=30,
) -> gym.Env:
    if used_inputs is None:
        used_inputs = ["postion", "heading", "pointgoal"]

    def state_to_vector(state):
        return np.concatenate([state[key] for key in used_inputs], axis=-1)

    def normalize_state(state):
        return (state - state_mean) / (state_std + 1e-8)

    def transform_state(state):
        raw_state = deepcopy(state)
        state = state_to_vector(state)
        return normalize_state(state), raw_state

    def scale_reward(reward):
        return reward_scale * reward

    env = HabitatWrapper(env)
    env.observation = transform_state
    if continuous:
        env = ContinuousActionWrapper(env, turn_angle=turn_angle)
    else:
        env = HabitatAcionWrapper(env, ignore_stop=ignore_stop)
    return env

