import json
import os
import pickle as pkl
import uuid
from collections import namedtuple
from io import BytesIO
from os.path import join

import gym
import numpy as np
import torch
from torchvision import transforms

RolloutFilenames = namedtuple(
    "RolloutFilenames", ["rollout_filename", "rendering_filename", "metadata_filename"]
)


def process_rendering(env, rendering_size, as_png):
    transform_chain = []
    screen = env.render(mode="rgb_array")
    screen = screen.transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    transform_chain.append(transforms.ToPILImage())
    transform_chain += [transforms.Resize(rendering_size)]
    transform = transforms.Compose(transform_chain)
    screen = transform(screen)
    if as_png:
        with BytesIO() as byte_io:
            screen.save(byte_io, "PNG")
            return byte_io.getvalue()
    else:
        return np.array(screen)


class StorageWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        data_dir,
        save_renderings=False,
        rendering_size=(64, 64),
        rendering_as_png=True,
        rollout_seed=None,
        rollout_uuid=None,
    ):
        super(StorageWrapper, self).__init__(env)
        self.data_dir = data_dir
        self.save_renderings = save_renderings
        self.rendering_size = rendering_size
        self.rendering_as_png = rendering_as_png
        self._process_rendering = lambda: process_rendering(
            env, rendering_size, rendering_as_png
        )
        self.rollout_seed = rollout_seed
        self.rollout_uuid = rollout_uuid
        self._in_capture_mode = False

    def _get_rollout_filename(self, rollout_seed=None, rollout_uuid=None):
        if rollout_seed is None:
            if rollout_uuid is None:
                rollout_uuid = uuid.uuid4().hex
        else:
            rollout_uuid = "seed={}".format(self.rollout_seed)
        rollout_filename = join(self.data_dir, "rollout_{}.pkl".format(rollout_uuid))
        metadata_filename = join(self.data_dir, "meta_{}.json".format(rollout_uuid))
        rendering_filename = join(
            self.data_dir, "rendering_{}.pkl".format(rollout_uuid)
        )
        return RolloutFilenames(
            rollout_filename=rollout_filename,
            metadata_filename=metadata_filename,
            rendering_filename=rendering_filename,
        )

    def reset(self, **kwargs):
        if self._in_capture_mode:
            raise RuntimeError("env.reset() must only be called once.")
        self.filenames = self._get_rollout_filename(
            self.rollout_seed, self.rollout_uuid
        )
        if any(os.path.isfile(p) for p in self.filenames):
            raise RuntimeError("Rollout data already exists for this filename.")
        if self.rollout_seed is not None:
            self.seed(self.rollout_seed)
            self.action_space.seed(self.rollout_seed)
            self.observation_space.seed(self.rollout_seed)
        obs = self.env.reset()
        self._action = []
        self._reward = [np.nan]
        self._done = [False]
        self._info = [{}]
        self._observation = [obs]
        if self.save_renderings:
            self._rendering = [self._process_rendering()]
        self._in_capture_mode = True
        return obs

    def step(self, action):
        if not self._in_capture_mode:
            raise RuntimeError("Must reset environment.")
        if self.rollout_seed is not None:
            if action is not None:
                raise ValueError("action must be None if rollout_seed is set")
            action = self.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        self._action.append(action)
        self._reward.append(reward)
        self._done.append(done)
        self._info.append(info)
        self._observation.append(obs)
        if self.save_renderings:
            self._rendering.append(self._process_rendering())
        return obs, reward, done, info

    def close(self):
        if not self._in_capture_mode:
            raise RuntimeError("Cannot close environment; env.reset() was not called.")
        # append NaN action to last step
        self._action.append(self._action[-1] * np.nan)
        rollout = {
            "action": np.stack(self._action),
            "reward": np.stack(self._reward),
            "info": self._info,
            "done": self._done,
            "observation": self._observation,
        }
        if self.rollout_seed is not None:
            rollout["rollout_seed"] = self.rollout_seed
        # save rollout data
        with open(self.filenames.rollout_filename, "wb") as file:
            pkl.dump(rollout, file)
        # save rollout metadata
        with open(self.filenames.metadata_filename, "w") as file:
            json.dump({"rollout_length": len(self._done)}, file)
        # save renderings
        if self.save_renderings:
            rendering = {
                "rendering": self._rendering,
                "rendering_as_png": self.rendering_as_png,
            }
            with open(self.filenames.rendering_filename, "wb") as file:
                pkl.dump(rendering, file)
        return self.env.close()
