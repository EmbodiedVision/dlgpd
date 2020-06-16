import itertools

import numpy as np
import torch
from torchvision.utils import make_grid
import plac
from pathlib import Path

from .data.loader import encode_batch_of_pairs, load_processed_train_batch
from .utils.plot import plt, cm, sns
from .utils.loading import load_model_skeleton


CMAPS = {
    "velocity": cm.viridis,
    "angle": cm.twilight,
    "position": cm.viridis,
    "default": cm.viridis,
}


@torch.no_grad()
def visualize_embeddings(val_data, vae, env_info, summary_writer, epoch):
    # Visualize latents, colored by states and actions
    sns.axes_style("whitegrid", {"axes.edgecolor": "1."})

    latent_size = vae.latent_size

    data = val_data.get_pairs_as_batch(max_pairs=1000)
    latents = encode_batch_of_pairs(data, vae).current.mean
    observations = data["observation"].current

    latents, observations = map(
        lambda x: x if not isinstance(x, torch.Tensor) else x.cpu().detach().numpy(),
        (latents, observations),
    )
    states = env_info.obs_to_state_fcn(observations)
    state_names = env_info.state_names
    state_types = env_info.state_types

    if latent_size == 2:
        latent_slices = np.array([[0, 1]])
    else:
        latent_slices = np.array(list(itertools.combinations(range(latent_size), 3)))

    ncols = len(state_names)
    nrows = len(latent_slices)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw=None if latent_size <= 2 else dict(projection="3d"),
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    for col_idx, (state_name, state_type) in enumerate(zip(state_names, state_types)):
        for row_idx, slice_idxs in enumerate(latent_slices):
            ax = axes[row_idx, col_idx]
            cmap = CMAPS[state_type]
            plt.set_cmap(cmap)
            if len(slice_idxs) <= 2:
                _plot = ax.scatter(
                    latents[:, slice_idxs[0]],
                    latents[:, slice_idxs[1]],
                    c=states[:, col_idx],
                )
            else:
                _plot = ax.scatter(
                    latents[:, slice_idxs[0]],
                    latents[:, slice_idxs[1]],
                    latents[:, slice_idxs[2]],
                    c=states[:, col_idx],
                )
            cb = fig.colorbar(_plot, ax=ax)
            cb.set_label(state_name)

    fig.suptitle(f"X visualization: Epoch {epoch}")
    fig.tight_layout()

    if summary_writer:
        summary_writer.add_figure(f"embeddings", fig, epoch)
    if summary_writer:
        plt.close("all")
    else:
        return fig


@torch.no_grad()
def visualize_reconstructions(
    val_data, vae, val_reconstruction_n_rollouts, summary_writer, epoch
):
    data_batch = val_data.get_chunks_as_batch(max_chunks=val_reconstruction_n_rollouts)
    rendering = data_batch["rendering"]
    latents = vae.encode_sequence(rendering)
    latent_samples = latents.sample
    reconstructions = vae.decode_sequence(latent_samples)
    # number of reconstructions may be smaller than number of observations,
    # if we have a warmup period
    n_reconstructions = reconstructions.shape[0]
    # compare reconstructions to last 'n_reconstructions' observations
    differences = rendering[-n_reconstructions:] - reconstructions
    # pad reconstructions and differences with white frames at the beginning
    padding = torch.ones(
        rendering.shape[0] - n_reconstructions,
        *reconstructions.shape[1:],
        device=reconstructions.device,
    )
    reconstructions = torch.cat([padding, reconstructions])
    differences = torch.cat([padding, differences])
    tb_grid = grid_from_batched_timeseries(
        [rendering, reconstructions, differences], normalize=False
    )
    summary_writer.add_image("reconstructions/reconstructions", tb_grid, epoch)


def grid_from_batched_timeseries(tensor_or_list, **kwargs):
    # make grid from batched timeseries tensor (T x B x <image_dims>)
    if type(tensor_or_list) in [list, tuple]:
        n_cols = tensor_or_list[0].shape[0]
        strip_n_rows = tensor_or_list[0].shape[1]
        assert all([t.shape[1] == strip_n_rows for t in tensor_or_list])
        assert all([t.shape[0] == n_cols for t in tensor_or_list])
        padding_row = torch.ones_like(tensor_or_list[0])
        tensor_or_list.append(padding_row)
        interleaved_tensor = torch.cat(
            [
                torch.stack(
                    [tensor_or_list[k][:, n] for k in range(len(tensor_or_list))]
                )
                for n in range(strip_n_rows)
            ]
        )
        images = interleaved_tensor.reshape(-1, 3, 64, 64)
    else:
        n_cols = tensor_or_list.shape[0]
        images = tensor_or_list.transpose(0, 1).reshape(-1, 3, 64, 64)
    # nrow parameter is number of images per row -> n_cols
    grid = make_grid(images, nrow=n_cols, **kwargs)
    return grid


@torch.no_grad()
def visualize_reward(val_data, vae, reward_model, env_info, summary_writer, epoch):
    pair_data = val_data.get_pairs_as_batch(max_pairs=1000)
    latents = encode_batch_of_pairs(pair_data, vae).current.mean.detach()
    next_obs = pair_data["observation"].next.cpu()
    actions = pair_data["action"]
    real_rewards = pair_data["reward"]
    real_rewards = real_rewards.flatten().cpu()
    predicted_rewards = reward_model(latents, actions).detach().cpu()
    predicted_rewards = predicted_rewards.flatten()
    fig, axes = _plot_states_rewards(
        next_obs, real_rewards, predicted_rewards, env_info
    )
    if summary_writer:
        summary_writer.add_figure(f"rewards", fig, epoch)
    if summary_writer:
        plt.close("all")
    else:
        return fig


def _set_labels(ax, *state_names):
    ax.set_xlabel(state_names[0])
    ax.set_ylabel(state_names[1])
    if hasattr(ax, "zaxis"):
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(state_names[2])


def _plot_states_rewards(observations, real_rewards, predicted_rewards, env_info):
    observations = np.array(observations)
    real_rewards = np.array(real_rewards)
    predicted_rewards = np.array(predicted_rewards)

    reward_diff = np.log(np.abs(real_rewards - predicted_rewards)).clip(-4, 100)
    states = env_info.obs_to_state_fcn(observations)
    state_names = env_info.state_names
    state_size = states.shape[-1]

    sns.axes_style("whitegrid")

    if state_size == 2:
        slices = np.array([[0, 1]])
    else:
        slices = np.array(list(itertools.combinations(range(state_size), 3)))

    ncols = 3
    nrows = len(slices)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw=None if state_size <= 2 else dict(projection="3d"),
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    for row_idx, slice_idxs in enumerate(slices):
        if state_size == 2:
            _states_args = states[:, 0], states[:, 1]
            _slice_state_names = state_names[0], state_names[1]
        else:
            _states_args = (
                states[:, slice_idxs[0]],
                states[:, slice_idxs[1]],
                states[:, slice_idxs[2]],
            )
            _slice_state_names = [state_names[k] for k in slice_idxs]

        ax = axes[row_idx, 0]
        _plot = ax.scatter(*_states_args, c=real_rewards.flatten(), s=5)
        cb = fig.colorbar(_plot, ax=ax, pad=0.1, shrink=0.8)
        ax.set_title("True Rewards")
        cb.set_label("True Rewards")
        _set_labels(ax, *_slice_state_names)
        ax.dist = 13

        ax = axes[row_idx, 1]
        _plot = ax.scatter(*_states_args, c=predicted_rewards.flatten(), s=5)
        cb = fig.colorbar(_plot, ax=ax, pad=0.1, shrink=0.8)
        ax.set_title("Learned Reward Model")
        cb.set_label("Learned Rewards")
        _set_labels(ax, *_slice_state_names)
        ax.dist = 13

        ax = axes[row_idx, 2]
        _plot = ax.scatter(*_states_args, c=reward_diff.flatten(), s=5)
        cb = fig.colorbar(_plot, ax=ax, pad=0.1, shrink=0.8)
        ax.set_title("Difference")
        cb.set_label("Reward Differences (log-scale)")
        ticks = cb.get_ticks()
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"1e({int(t)})" for t in ticks])
        _set_labels(ax, *_slice_state_names)
        ax.dist = 13

    plt.tight_layout()
    plt.set_cmap(cm.viridis)
    return fig, axes


@plac.annotations(model_path=plac.Annotation(kind="positional", type=str))
def main(model_path,):
    """ Write latent states to *.npz file """

    config, model = load_model_skeleton(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    latent, action, reward, observation = load_processed_train_batch(
        config, model.vae, max_pairs=1000, with_observation=True
    )
    # save latents to run directory
    latent = latent.current.mean.cpu().numpy()
    observation = observation.current.cpu().numpy()
    model_path_obj = Path(model_path)
    npz_file = model_path_obj.parent.parent.joinpath(
        f"{model_path_obj.stem}_latents.npz"
    )
    np.savez(npz_file, {"latent": latent, "observation": observation})
    print(f"Latent states saved at {npz_file}")


if __name__ == "__main__":
    with torch.no_grad():
        plac.call(main)
