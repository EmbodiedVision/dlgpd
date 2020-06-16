import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import PendulumEnv
from gym.utils import seeding
from collections import namedtuple


EnvironmentInfo = namedtuple(
    "EnvironmentInfo",
    [
        "env_name",
        "obs_to_state_fcn",
        "state_names",
        "state_types",
        "data_collection_kwargs",
        "ctrl_env_kwargs",
        "variation_kwargs",
        "action_size",
    ],
)


def cos_sin_to_angle(cos, sin):
    cos, sin = cos.clip(-1, 1), sin.clip(-1, 1)
    return np.where(sin > 0, np.arccos(cos), 2 * np.pi - np.arccos(cos))


def pendulum_obs2state_fcn(observations):
    angle = cos_sin_to_angle(observations[..., 0], observations[..., 1])
    angle_vel = observations[..., 2]
    return np.stack((angle, angle_vel), axis=-1)


PENDULUM_STATE_NAMES = ["Angle", "Angular velocity"]
PENDULUM_STATE_TYPES = ["angle", "velocity"]
PENDULUM_DATA_COLLECTION_ENV_KWARGS = {"init_type": "random"}
PENDULUM_CTRL_ENV_KWARGS = {"init_type": "bottom"}
PENDULUM_VARIATION_KWARGS = {
    "lighterpole": {"m": 0.2},
    "heavierpole": {"m": 1.5},
    "invactions": {"action_factor": -1},
    "standard": {},
}
ENV_INFO_PENDULUM = EnvironmentInfo(
    env_name="ModifiedPendulumEnv-v0",
    obs_to_state_fcn=pendulum_obs2state_fcn,
    state_names=PENDULUM_STATE_NAMES,
    state_types=PENDULUM_STATE_TYPES,
    data_collection_kwargs=PENDULUM_DATA_COLLECTION_ENV_KWARGS,
    ctrl_env_kwargs=PENDULUM_CTRL_ENV_KWARGS,
    variation_kwargs=PENDULUM_VARIATION_KWARGS,
    action_size=1,
)


def register_envs():
    if "ModifiedPendulumEnv-v0" not in gym.envs.registration.registry.env_specs:
        gym.envs.registration.register(
            id="ModifiedPendulumEnv-v0",
            entry_point="dlgpd.data.envs:ModifiedPendulumEnv",
            kwargs={"render_action": False},
        )


"""
=======================================================================
The code below this block comment is derived from code
of the OpenAI Gym, v0.17.0.
For license information on OpenAI Gym, see 3RD_PARTY_LICENSES.md in the
root directory of this repository.
=======================================================================
"""


class ModifiedPendulumEnv(PendulumEnv):
    """
    Modified PendulumEnv
    """

    def __init__(
        self,
        render_action=False,
        init_type="standard",
        g=10.0,
        m=1.0,
        friction=0,
        action_factor=1.0,
    ):
        super(ModifiedPendulumEnv, self).__init__(g)
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.viewer = None
        self.render_action = render_action

        self.m = m
        self.friction = friction
        self.action_factor = action_factor

        high = np.array([1.0, 1.0, self.max_speed])
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.init_type = init_type

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = u * self.action_factor
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = 1.0
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        friction = -self.friction * thdot
        u += friction

        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = (
            thdot
            + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(
            newthdot, -self.max_speed, self.max_speed
        )  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def _render_no_action(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def render(self, mode="human"):
        if self.render_action:
            return super(ModifiedPendulumEnv, self).render(mode)
        else:
            return self._render_no_action(mode)

    def reset(self):
        if self.init_type == "standard":
            return super(ModifiedPendulumEnv, self).reset()
        elif self.init_type == "random":
            high = np.array([np.pi, self.max_speed])
            self.state = self.np_random.uniform(low=-high, high=high)
            self.last_u = None
            return self._get_obs()
        elif self.init_type == "bottom":
            high = np.array([0.05, 0.05])
            self.state = self.np_random.uniform(low=-high, high=high)
            self.state[0] += np.pi
            self.last_u = None
            return self._get_obs()
        else:
            raise ValueError


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
