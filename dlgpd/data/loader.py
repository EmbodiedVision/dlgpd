from collections import defaultdict, namedtuple
from itertools import islice
from pathlib import Path

import gym
import torch
from sacred import Ingredient
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import RolloutDataset
from .envs import register_envs, ENV_INFO_PENDULUM
from ..models import Belief

MODULE_PATH = Path(__file__).parent.parent.parent

data_ingredient = Ingredient("data")


@data_ingredient.config
def data_cfg():
    task_name = "pendulum"
    batch_size_pairs = 1024
    batch_size_chunks = 16
    shuffle_batches_pairs = True
    shuffle_batches_chunks = True
    subset_shuffle_seed = None
    rollout_length = 30
    chunk_length = 30
    data_base_dir = str(MODULE_PATH.joinpath("data"))


@data_ingredient.capture
def get_random_collection_env(task_name, variation_name=None):
    env_info = get_env_info(task_name)
    env_name = env_info.env_name
    env_kwargs = env_info.data_collection_kwargs
    if variation_name is not None:
        env_kwargs = dict(**env_kwargs, **env_info.variation_kwargs[variation_name])
    return env_name, env_kwargs


@data_ingredient.capture
def get_control_env(task_name, variation_name=None):
    env_info = get_env_info(task_name)
    env_name = env_info.env_name
    env_kwargs = env_info.ctrl_env_kwargs
    if variation_name is not None:
        env_kwargs = dict(**env_kwargs, **env_info.variation_kwargs[variation_name])
    return env_name, env_kwargs


def make_env(env_name, env_kwargs):
    register_envs()
    return gym.make(env_name, **env_kwargs)


@data_ingredient.capture
def get_env_info(task_name):
    if task_name == "pendulum":
        return ENV_INFO_PENDULUM
    else:
        raise ValueError(f"No env_info for {task_name}")


def gaussian_noise(tensor):
    tensor += 0.01 * torch.randn_like(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def encode_batch_of_pairs(batch, vae):
    latent_current = vae.encode_sequence(batch["rendering_history"].current)
    latent_next = vae.encode_sequence(batch["rendering_history"].next)
    # remove additional sliding window dimension introduced by "encode_sequence"
    latent_current = Belief(*[k.squeeze(0) for k in latent_current])
    latent_next = Belief(*[k.squeeze(0) for k in latent_next])
    return TransitionTuple(current=latent_current, next=latent_next)


TransitionTuple = namedtuple("TransitionTuple", ["current", "next"])


class DataHandler(object):
    def __init__(
        self,
        chunk_dataset,
        pair_dataset,
        chunk_length,
        batch_size_chunks,
        batch_size_pairs,
        shuffle_batches_chunks,
        shuffle_batches_pairs,
    ):
        self._chunk_dataset = chunk_dataset
        self._pair_dataset = pair_dataset
        self._chunk_loader_sequential, self._chunk_loader_shuffled = [
            DataLoader(
                chunk_dataset,
                batch_size=batch_size_chunks,
                shuffle=shuffle,
                drop_last=False,
                num_workers=1,
            )
            for shuffle in [False, shuffle_batches_chunks]
        ]
        self._pair_loader_sequential, self._pair_loader_shuffled = [
            DataLoader(
                pair_dataset,
                batch_size=batch_size_pairs,
                shuffle=shuffle,
                drop_last=False,
                num_workers=1,
            )
            for shuffle in [False, shuffle_batches_pairs]
        ]
        self._n_chunks = len(chunk_dataset)
        self._chunk_length = chunk_length

    @property
    def chunk_loader(self):
        return self._chunk_loader_shuffled

    @property
    def pair_loader(self):
        return map(self._process_pair_batch, self._pair_loader_shuffled)

    @property
    def n_chunks(self):
        return self._n_chunks

    @property
    def chunk_length(self):
        return self._chunk_length

    def get_chunks_as_batch(self, max_chunks=None):
        keys = ["action", "reward", "observation", "rendering"]
        accumulated_chunk_data = {k: [] for k in keys}
        for chunk_data in self.chunk_iterator(max_chunks):
            for k in keys:
                accumulated_chunk_data[k].append(
                    torch.as_tensor(chunk_data[k]).float().cuda()
                )
        stacked_chunk_data = {
            k: torch.stack(v, dim=1) for k, v in accumulated_chunk_data.items()
        }
        return stacked_chunk_data

    def chunk_iterator(self, max_chunks=None):
        keys = ["action", "reward", "observation", "rendering"]
        _it = range(len(self._chunk_dataset))
        if max_chunks is not None:
            _it = islice(_it, max_chunks)
        for chunk_idx in _it:
            yield {
                k: torch.as_tensor(self._chunk_dataset[chunk_idx][k]).float().cuda()
                for k in keys
            }

    def get_pairs_as_batch(self, max_pairs=None):
        acc_batch = defaultdict(list)
        items_missing = max_pairs
        for batch in self._pair_loader_shuffled:
            bs = len(batch[list(batch.keys())[0]])
            if max_pairs is not None:
                for k, v in batch.items():
                    acc_batch[k].append(v[:items_missing])
                items_missing -= bs
                if items_missing <= 0:
                    break
            else:
                for k, v in batch.items():
                    acc_batch[k].append(v)
        for k, v in acc_batch.items():
            if isinstance(v, (list, tuple)) and isinstance(v[0], torch.Tensor):
                acc_batch[k] = torch.cat(v, dim=0)
        return self._process_pair_batch(acc_batch)

    def _process_pair_batch(self, pair_batch):
        t_b_dim_batch = {
            k: t.transpose(0, 1).float().cuda()
            for k, t in pair_batch.items()
            if k in ["action", "reward", "observation", "rendering"]
        }
        # action applied to second-to-last state
        action = t_b_dim_batch["action"][-2, ...]
        # reward collected when entering last state
        reward = t_b_dim_batch["reward"][-1, ...]
        observation = TransitionTuple(
            current=t_b_dim_batch["observation"][-2, ...],
            next=t_b_dim_batch["observation"][-1, ...],
        )
        rendering = TransitionTuple(
            current=t_b_dim_batch["rendering"][-2, ...],
            next=t_b_dim_batch["rendering"][-1, ...],
        )
        rendering_history = TransitionTuple(
            current=t_b_dim_batch["rendering"][:-1, ...],
            next=t_b_dim_batch["rendering"][1:, ...],
        )
        return dict(
            action=action,
            reward=reward,
            observation=observation,
            rendering=rendering,
            rendering_history=rendering_history,
        )


def load_processed_train_batch(config, vae, max_pairs, with_observation=False):
    # Train data for inferring the scaling
    env_name, env_kwargs = get_random_collection_env(config["data"]["task_name"])
    train_data = load_data(
        env_name,
        env_kwargs,
        split_name="train",
        n_rollouts_total=config["n_train_rollouts_total"],
        n_rollouts_subset=config["n_train_rollouts_subset"],
        rollout_length=config["data"]["rollout_length"],
        chunk_length=config["data"]["chunk_length"],
        batch_size_chunks=1,
        batch_size_pairs=1024,
        shuffle_batches_chunks=True,
        shuffle_batches_pairs=True,
        data_base_dir=config["data"]["data_base_dir"],
        subset_shuffle_seed=config["data"]["subset_shuffle_seed"],
    )
    pair_batch = train_data.get_pairs_as_batch(max_pairs=max_pairs)
    with torch.no_grad():
        latent = encode_batch_of_pairs(pair_batch, vae)
    action = pair_batch["action"]
    reward = pair_batch["reward"]
    if with_observation:
        observation = pair_batch["observation"]
        return latent, action, reward, observation
    else:
        return latent, action, reward


@data_ingredient.capture
def load_data(
    env_name,
    env_kwargs,
    split_name,
    n_rollouts_total,
    n_rollouts_subset,
    rollout_length,
    chunk_length,
    batch_size_chunks,
    batch_size_pairs,
    shuffle_batches_chunks,
    shuffle_batches_pairs,
    data_base_dir,
    subset_shuffle_seed,
):
    """

    Parameters
    ----------
    env_name
    env_kwargs
    split_name
    n_rollouts_total
    n_rollouts_subset
    rollout_length
    batch_size_chunks
    batch_size_pairs
    subset_shuffle_seed

    Returns
    -------

    """
    register_envs()
    image_transform = transforms.Compose([transforms.ToTensor(), gaussian_noise])
    print("Loading chunks...")
    chunk_dataset = RolloutDataset(
        data_base_dir=data_base_dir,
        split_name=split_name,
        env_name=env_name,
        env_kwargs=env_kwargs,
        n_rollouts_total=n_rollouts_total,
        n_rollouts_subset=n_rollouts_subset,
        rollout_length=rollout_length,
        rollout_subset_sampling_seed=subset_shuffle_seed,
        subsequence_length=chunk_length,
        image_transform=image_transform,
    )
    print("Loading transitions...")
    pair_dataset = RolloutDataset(
        data_base_dir=data_base_dir,
        split_name=split_name,
        env_name=env_name,
        env_kwargs=env_kwargs,
        n_rollouts_total=n_rollouts_total,
        n_rollouts_subset=n_rollouts_subset,
        rollout_length=rollout_length,
        rollout_subset_sampling_seed=subset_shuffle_seed,
        subsequence_length=3,
        image_transform=image_transform,
    )
    return DataHandler(
        chunk_dataset,
        pair_dataset,
        chunk_length,
        batch_size_chunks=batch_size_chunks,
        batch_size_pairs=batch_size_pairs,
        shuffle_batches_chunks=shuffle_batches_chunks,
        shuffle_batches_pairs=shuffle_batches_pairs,
    )
