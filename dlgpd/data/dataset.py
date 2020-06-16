import json
import os
import pickle as pkl
import re
from io import BytesIO
from os.path import join

import gym
import numpy as np
import torch
from PIL import Image
from joblib import Parallel, delayed
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from xvfbwrapper import Xvfb

from .wrappers import StorageWrapper


def kwargs_to_string(env_kwargs):
    for k, v in env_kwargs.items():
        if "&" in k or "=" in k:
            raise ValueError
        if isinstance(v, str) and ("&" in v or "=" in v):
            raise ValueError
    sorted_keys = sorted(list(env_kwargs.keys()))
    env_kwargs_str = "&".join(["{}={}".format(k, env_kwargs[k]) for k in sorted_keys])
    if env_kwargs_str == "":
        env_kwargs_str = "base"
    return env_kwargs_str


class RolloutSubsequences(object):
    def __init__(self, subsequence_length, rollout_seeds, rollout_lengths):
        self.subsequence_length = subsequence_length
        self._rollout_seeds = rollout_seeds
        self._rollout_lengths = np.array(rollout_lengths)
        self._subsequences_per_rollout = self._rollout_lengths - subsequence_length + 1
        # set subsequences to 0 if _subsequences_per_rollout < 0
        self._subsequences_per_rollout[self._subsequences_per_rollout < 0] = 0
        self._cumulative_subsequences = np.cumsum(self._subsequences_per_rollout)

    @property
    def rollout_seeds(self):
        return self._rollout_seeds

    @property
    def rollout_lengths(self):
        return self._rollout_lengths

    @property
    def subsequences_per_rollout(self):
        return self._subsequences_per_rollout

    @property
    def n_rollouts(self):
        return len(self._rollout_seeds)

    def __getitem__(self, item):
        if item >= self.__len__():
            raise ValueError
        rollout_idx = np.searchsorted(self._cumulative_subsequences, item, side="right")
        cum_sub_concat = np.concatenate((np.zeros(1), self._cumulative_subsequences))
        subsequence_start = int(item - cum_sub_concat[rollout_idx])
        subsequence_end = int(subsequence_start + self.subsequence_length)
        return rollout_idx, subsequence_start, subsequence_end

    def __len__(self):
        return self._cumulative_subsequences[-1]


def save_single_rollout(registry, env_name, env_kwargs, rollout_length, seed, data_dir):
    env = registry.make(env_name, **env_kwargs)
    env = StorageWrapper(
        env,
        data_dir=data_dir,
        save_renderings=True,
        rendering_as_png=False,
        rollout_seed=seed,
    )
    env.reset()
    for step in range(rollout_length - 1):
        # action is randomly sampled from action space for StorageWrapper
        _, _, done, _ = env.step(action=None)
        if done:
            break
    env.close()


def save_rollouts(
    env_name, env_kwargs, rollout_length, seeds, data_directory, n_jobs=4
):
    os.makedirs(data_directory, exist_ok=True)
    registry = gym.envs.registry
    with Xvfb(width=1400, height=900, colordepth=24) as xvfb:
        Parallel(n_jobs=n_jobs)(
            delayed(save_single_rollout)(
                registry, env_name, env_kwargs, rollout_length, seed, data_directory
            )
            for seed in tqdm(seeds)
        )


class RolloutDataset(data.Dataset):
    """ Rollout dataset """

    def __init__(
        self,
        data_base_dir,
        split_name,
        env_name,
        env_kwargs,
        n_rollouts_total,
        n_rollouts_subset,
        rollout_length,
        load_renderings=True,
        auto_generate=True,
        sequence_transform=None,
        subsequence_transform=None,
        image_transform=None,
        subsequence_length=None,
        rollout_subset_sampling_seed=None,
    ):
        """
        Initialize RolloutDataset

        Parameters
        ----------
        data_base_dir: str
            Base data directory
        split_name: str
            Split name
        env_name: Env
            Environment name
        env_kwargs: dict
            Environment kwargs
        n_rollouts_total: int
            Number of rollouts to generate in advance
        n_rollouts_subset: int
            Limit dataset to this number of rollouts
        rollout_length: int
            Length of rollouts
        load_renderings: bool
            Load renderings
        auto_generate: bool
            Automatically generate missing rollouts
        sequence_transform: fcn, optional
            Function to transform sequences
        subsequence_transform: fcn, optional
            Function to transform subsequences
        image_transform: torchvision.transform.Transform, optional
            Image transformation
        subsequence_length: int, optional
            Subsequence length (default: limit_horizon)
        rollout_subset_sampling_seed: int, optional
            Subsampling seed
        """
        self.data_base_dir = data_base_dir
        self.split_name = split_name
        if split_name == "train":
            self.seed_range = (0, int(1e6) - 1)
        elif split_name == "val":
            self.seed_range = (int(1e6), int(2e6) - 1)
        elif split_name == "evidence_0":
            self.seed_range = (int(2e6), int(2e6) + int(1e5) - 1)
        elif split_name == "evidence_1":
            self.seed_range = (int(2e6) + int(1e5), int(2e6) + int(2e5) - 1)
        elif split_name == "evidence_2":
            self.seed_range = (int(2e6) + int(2e5), int(2e6) + int(3e5) - 1)
        elif split_name == "evidence_3":
            self.seed_range = (int(2e6) + int(3e5), int(2e6) + int(4e5) - 1)
        elif split_name == "evidence_4":
            self.seed_range = (int(2e6) + int(4e5), int(2e6) + int(5e5) - 1)
        else:
            raise ValueError("Invalid split_name")
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.n_rollouts_total = n_rollouts_total
        self.n_rollouts_subset = n_rollouts_subset
        self.rollout_length = rollout_length
        if n_rollouts_subset > n_rollouts_total:
            raise ValueError("n_rollouts_subset must be less or equal n_rollouts_total")
        self.load_renderings = load_renderings
        self.auto_generate = auto_generate
        self.sequence_transform = sequence_transform
        self.subsequence_transform = subsequence_transform
        self.subsequence_length = subsequence_length
        if image_transform is None:
            image_transform = transforms.Compose([transforms.ToTensor()])
        self.image_transform = image_transform
        self.rollout_subset_sampling_seed = rollout_subset_sampling_seed
        self._subsequences = self._find_subsequences()
        self._rollout_cache = self._load_all_rollouts()

    def _get_data_directory(self):
        data_directory = os.path.join(
            self.data_base_dir,
            "{}_l{}_{}_{}".format(
                self.env_name.lower(),
                self.rollout_length,
                kwargs_to_string(self.env_kwargs),
                self.split_name,
            ),
        )
        return data_directory

    def _find_subsequences(self):
        available_rollouts = self._scan_data_directory()
        print(
            "Found {} rollouts in data directory".format(available_rollouts.n_rollouts)
        )
        if available_rollouts.n_rollouts < self.n_rollouts_total:
            if self.auto_generate:
                print("Generate missing rollouts...")
                self._generate_missing_rollouts(available_rollouts)
                available_rollouts = self._scan_data_directory()
            else:
                raise ValueError("Not enough rollouts available")
        rollout_subset_sampling_seed = self.rollout_subset_sampling_seed
        if rollout_subset_sampling_seed is None:
            selected_indices = np.arange(0, self.n_rollouts_subset)
        else:
            shuffle_rng = np.random.RandomState(rollout_subset_sampling_seed)
            indices = np.copy(np.arange(0, available_rollouts.n_rollouts))
            shuffled_indices = shuffle_rng.permutation(indices)
            selected_indices = shuffled_indices[: self.n_rollouts_subset]
        subsequences = RolloutSubsequences(
            self.subsequence_length,
            available_rollouts.rollout_seeds[selected_indices],
            available_rollouts.rollout_lengths[selected_indices],
        )
        print("Loaded {} rollouts".format(subsequences.n_rollouts))
        return subsequences

    def _scan_data_directory(self):
        data_dir = self._get_data_directory()
        files_list = os.listdir(data_dir) if os.path.isdir(data_dir) else []
        rollout_seeds = []
        rollout_lengths = []
        for filename in files_list:
            re_match = re.match("meta_seed=(.+)\.json", filename)
            if not re_match:
                continue
            seed = int(re_match.group(1))
            if not self.seed_range[0] <= seed <= self.seed_range[1]:
                continue
            with open(join(data_dir, filename), "r") as file:
                metadata = json.load(file)
            rollout_seeds.append(seed)
            rollout_lengths.append(metadata["rollout_length"])
        rollout_seeds = np.array(rollout_seeds)
        rollout_lengths = np.array(rollout_lengths)
        # sort rollout seeds and rollout lengths
        argsort_idxs = np.argsort(rollout_seeds)
        rollout_seeds = rollout_seeds[argsort_idxs]
        rollout_lengths = rollout_lengths[argsort_idxs]
        if len(rollout_seeds) > 0:
            if rollout_seeds[-1] - self.seed_range[0] != len(rollout_seeds) - 1:
                raise ValueError(
                    f"Inconsistent rollouts in data directory {data_dir}"
                    f"(max: {rollout_seeds[-1]}, "
                    f"seed_base: {self.seed_range[0]}, "
                    f"n_rollouts: {len(rollout_seeds)})"
                )
        return RolloutSubsequences(
            self.subsequence_length, rollout_seeds, rollout_lengths
        )

    def _generate_missing_rollouts(self, available_rollouts):
        available_seeds = set(available_rollouts.rollout_seeds)
        required_seeds = set(self.seed_range[0] + np.arange(self.n_rollouts_total))
        missing_seeds = list(required_seeds.difference(available_seeds))
        missing_seeds = [int(s) for s in missing_seeds]
        save_rollouts(
            self.env_name,
            self.env_kwargs,
            self.rollout_length,
            missing_seeds,
            self._get_data_directory(),
        )

    def _process_raw_rollout(self, raw_rollout_data):
        image_tensors = []
        rollout_data = {
            "action": np.stack(raw_rollout_data["action"]),
            "reward": np.stack(raw_rollout_data["reward"]),
            "done": raw_rollout_data["done"],
            "observation": np.stack(raw_rollout_data["observation"]),
            "rollout_seed": raw_rollout_data["rollout_seed"],
        }
        if self.load_renderings:
            for rendering in raw_rollout_data["rendering"]:
                if raw_rollout_data["rendering_as_png"]:
                    pil_image = Image.open(BytesIO(rendering))
                    image_tensor = self.image_transform(pil_image)
                else:
                    pil_image = Image.fromarray(rendering)
                    image_tensor = self.image_transform(pil_image)
                image_tensors.append(image_tensor)
            image_tensors = torch.stack(image_tensors)
            rollout_data["rendering"] = image_tensors

        if self.sequence_transform is not None:
            rollout_data = self.sequence_transform(rollout_data)

        return rollout_data

    def _load_all_rollouts(self):
        seeds = self._subsequences.rollout_seeds
        rollout_pool = [self._load_rollout(seed) for seed in tqdm(seeds)]
        return rollout_pool

    def _load_rollout(self, rollout_seed):
        rollout_filepath = join(
            self._get_data_directory(), "rollout_seed={}.pkl".format(rollout_seed)
        )
        with open(rollout_filepath, "rb") as file:
            raw_rollout_data = pkl.load(file)
        if self.load_renderings:
            renderings_filepath = join(
                self._get_data_directory(), "rendering_seed={}.pkl".format(rollout_seed)
            )
            with open(renderings_filepath, "rb") as file:
                raw_rendering_data = pkl.load(file)
            raw_rollout_data = dict(**raw_rollout_data, **raw_rendering_data)
        return self._process_raw_rollout(raw_rollout_data)

    def __getitem__(self, item):
        rollout_idx, subsequence_start, subsequence_end = self._subsequences[item]
        sequence_data = self._rollout_cache[rollout_idx]
        _slice = slice(subsequence_start, subsequence_end)
        subsequence_data = {
            "action": sequence_data["action"][_slice],
            "reward": sequence_data["reward"][_slice],
            "done": sequence_data["done"][_slice],
            "observation": sequence_data["observation"][_slice],
            "sequence_start": subsequence_start,
        }
        if "rendering" in sequence_data.keys():
            subsequence_data["rendering"] = sequence_data["rendering"][_slice]

        if self.subsequence_transform is not None:
            subsequence_data = self.subsequence_transform(subsequence_data)
        return subsequence_data

    def __len__(self):
        return len(self._subsequences)
