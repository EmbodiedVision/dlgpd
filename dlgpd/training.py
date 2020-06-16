"""
Train joint model with Likelihood-loss
"""

from collections import defaultdict
from pathlib import Path
from warnings import warn

import gpytorch
import sacred
import seaborn as sns
import torch
from torch import optim

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm as tqdm_base

sns.set()

from .models.dlgpd import DLGPDModel, DLGPDOptimizer
from .data.loader import (
    data_ingredient,
    load_data,
    encode_batch_of_pairs,
    get_random_collection_env,
    get_env_info,
)
from .models.gp import gp_ingredient, build_gp
from .models.vae import build_vae, vae_ingredient
from .models.reward import build_reward_model, reward_ingredient
from .utils import logging, experiments
from .visualization import (
    visualize_embeddings,
    visualize_reconstructions,
    visualize_reward,
)

from torchvision.utils import make_grid

MODULE_PATH = Path(__file__).parent.parent

experiment_name = "dlgpd"
ex = sacred.Experiment(
    experiment_name,
    ingredients=[data_ingredient, gp_ingredient, vae_ingredient, reward_ingredient],
)

RUN_BASEDIR = MODULE_PATH.joinpath(f"experiments/{experiment_name}/")
ex.observers.append(sacred.observers.FileStorageObserver(str(RUN_BASEDIR)))


@ex.capture
def get_experiment_directory(_run):
    if _run.unobserved:
        date_str = str(_run.start_time).replace(" ", "_")
        return MODULE_PATH.joinpath(
            f"experiments/{experiment_name}/unobs/unobs_{date_str}"
        )
    else:
        run_id = _run._id
        return RUN_BASEDIR.joinpath(f"{run_id}")


@ex.config
def cfg(_log):
    """Main configuration"""
    # Global seed
    seed = 1

    # General Training
    epochs = 2_000  # Number of epochs for the regular training
    loss_weights = {"reconstruction": 1, "neg_mll": 1, "neg_entropy": 1, "reward": 1}
    reconstruction_loss = "bce"  # 'bce' or 'mse'
    latent_size = 3
    gp_evidence_size = 28 * 30  # 30 rollouts with 28 transitions

    # Number of rollouts
    n_train_rollouts_total = 500
    n_train_rollouts_subset = 500
    n_val_rollouts_total = 50
    n_val_rollouts_subset = 50

    # Number of rollouts in the validation data
    validate_every = 10
    visualize_every = 50

    # For validating predictions
    val_filter_length = 10
    val_prediction_length = 20

    # For validating reconstructions
    val_reconstruction_n_rollouts = 20

    # Utilities
    checkpoint_every = 50  # Save model every N steps
    patience = 0  # If validation error does not decrease for `patience` steps STOP

    # Learning rates
    vae_lr = 1e-3
    gp_lr = 1e-3
    reward_model_lr = 1e-3

    reward_zero_grad = True
    detach_latent_for_reward = True

    reward_model_momentum = None
    reward_model_adam_beta1 = 0.9
    reward_model_adam_beta2 = 0.999


@reward_ingredient.config
def reward_cfg():
    mean_fn_name = "empirical_min"
    rescale = "none"
    lengthscale_gamma_prior = None
    outputscale_gamma_prior = (1, 5)
    snr_type = "pilco"
    snr_penalty_tolerance = 10
    snr_penalty_p = 8
    init_noise_factor = 0.2
    noise_bound = None
    noise_bound_factor = 0.001


@gp_ingredient.config
def gp_cfg():
    gp_kernel = "rbf"  # Kernel: 'rbf' or 'matern'
    rescale = "standard"
    lengthscale_gamma_prior = None
    outputscale_gamma_prior = (1, 5)
    snr_type = "pilco"
    snr_penalty_tolerance = 10
    snr_penalty_p = 8
    init_noise_factor = 0.2
    noise_bound = None
    noise_bound_factor = 0.001


@vae_ingredient.config
def vae_cfg():
    class_name = "StackedImageFilter"
    vae_kwargs = {"n_images": 2, "deterministic": False}


@ex.capture
def train_loop(
    train_data,
    val_data,
    env_info,
    dlgpd_model,
    dlgpd_optimizer,
    epochs,
    checkpoint_every,
    checkpoint_fn,
    visualize_every,
    validate_every,
    val_filter_length,
    val_prediction_length,
    val_reconstruction_n_rollouts,
    patience,
    summary_writer,
    _run,
    _log,
):
    """Train the whole model for N epochs

    The loop consists of:
    1. Training step
    2. Checkpoint (every C steps)
    3. Visualization (every V steps)
    4. Validation (every V steps)
    """
    if epochs == 0:
        return

    best_validation_loss = 999999999
    no_improvement_counter = 0

    bar = tqdm(range(1, epochs + 1), leave=False)
    for epoch in bar:
        checkpointed = False

        desc = f"[Train Epoch: {epoch}/{epochs}]"
        _log.debug(f"[Train Epoch: {epoch}/{epochs}]")
        bar.set_description(desc)

        # 1. Train
        dlgpd_model.train()
        train_epoch(
            epoch=epoch,
            epochs=epochs,
            train_data=train_data,
            dlgpd_model=dlgpd_model,
            dlgpd_optimizer=dlgpd_optimizer,
            summary_writer=summary_writer,
        )

        # Prepare for evaluation
        dlgpd_model.eval()

        # 2. Checkpoint
        if epoch % checkpoint_every == 0:
            _log.debug(f"[Train Epoch: {epoch}/{epochs}] Checkpoint")
            checkpoint_fn(epoch)
            checkpointed = True

        # Log parameters
        dlgpd_model.log_hyperparameters(summary_writer, epoch)

        # 3. Visualize states and next states
        if epoch % visualize_every == 0:
            with sns.axes_style("whitegrid", {"axes.edgecolor": "1."}):
                visualize_embeddings(
                    val_data=val_data,
                    vae=dlgpd_model.vae,
                    env_info=env_info,
                    summary_writer=summary_writer,
                    epoch=epoch,
                )

                # visualize_reconstructions(
                #    val_data=val_data,
                #    vae=dlgpd_model.vae,
                #    val_reconstruction_n_rollouts=val_reconstruction_n_rollouts,
                #    summary_writer=summary_writer,
                #    epoch=epoch,
                # )

            visualize_predictions(
                val_data=val_data,
                vae=dlgpd_model.vae,
                gp_model=dlgpd_model.gp,
                reward_model=dlgpd_model.reward_model,
                n_rollouts=10,
                filter_length=val_filter_length,
                prediction_length=val_prediction_length,
                summary_writer=summary_writer,
                epoch=epoch,
            )

            visualize_reward(
                val_data=val_data,
                vae=dlgpd_model.vae,
                reward_model=dlgpd_model.reward_model,
                env_info=env_info,
                summary_writer=summary_writer,
                epoch=epoch,
            )

        # 4. Validate
        if epoch % validate_every == 0:
            val_loss = validate_all(epoch, val_data, dlgpd_model, summary_writer)
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                _log.info(
                    f"[Train Epoch: {epoch}/{epochs}] "
                    "Lowest validation error - Save models"
                )
                # Save as current 'best' model (epoch = None)
                checkpoint_fn(epoch=None)
                # Also save epoch-indexed checkpoint
                if not checkpointed:
                    checkpoint_fn(epoch)
                no_improvement_counter = 0
                _run.result = best_validation_loss
                log_best_validation_loss(epoch, val_loss)
            else:
                no_improvement_counter += validate_every

        if 0 < patience <= no_improvement_counter:
            _log.info(
                f"[Train Epoch: {epoch}/{epochs}]"
                + "No improvement for {} epochs - break training:".format(
                    no_improvement_counter
                )
            )
            break
    _log.info("Final results:")
    _log.info(desc)
    return best_validation_loss


def log_best_validation_loss(epoch, val_loss):
    experiment_directory = get_experiment_directory()
    filename = experiment_directory.joinpath("best_validation_log.txt")
    with open(str(filename), "a") as filehandle:
        filehandle.write(f"{str(epoch)};{str(val_loss)}\n")


def apply_momentum_to_parameters(parameters, momentum):
    # grad_updated = momentum*grad_old + (1-momentum)*grad_new
    if momentum is None:
        return
    for param in parameters:
        if hasattr(param, "grad_cache"):
            if isinstance(momentum, (tuple, list)):
                param.grad = momentum[0] * param.grad_cache + momentum[1] * param.grad
            else:
                param.grad = momentum * param.grad_cache + (1 - momentum) * param.grad
        param.grad_cache = param.grad.clone()


@ex.capture
def train_epoch(
    epoch,
    epochs,
    train_data,
    dlgpd_model,
    dlgpd_optimizer,
    gp_evidence_size,
    loss_weights,
    reconstruction_loss,
    reward_model_momentum,
    reward_zero_grad,
    detach_latent_for_reward,
    summary_writer,
):
    cumulative_metrics = defaultdict(float)
    running_batch_size = 0

    missing_evidence = gp_evidence_size
    evidence_curr_states = []
    evidence_next_states = []
    evidence_actions = []
    evidence_rewards = []

    pair_loader = train_data.pair_loader

    train_bar = tqdm(pair_loader, leave=epoch != epochs)
    for data in train_bar:
        # do not zero grad for reward (momentum on reward model)
        dlgpd_optimizer.zero_grad(except_reward=not reward_zero_grad)

        data["latent"] = encode_batch_of_pairs(data, dlgpd_model.vae)

        batch_size, metrics = compute_pairwise_metrics(
            data, dlgpd_model, in_eval_mode=False
        )

        _, reward_metrics = compute_reward_metrics(
            data, dlgpd_model, detach_latent=detach_latent_for_reward
        )
        metrics = dict(**metrics, **reward_metrics)

        running_batch_size += batch_size
        for k, v in metrics.items():
            cumulative_metrics[k] += float(v.detach() * batch_size)

        # Aggregate losses and perform gradient step
        bce_reconstruction_loss = metrics["bce_reconstruction_loss"]
        mse_reconstruction_loss = metrics["mse_reconstruction_loss"]
        r_loss = dict(bce=bce_reconstruction_loss, mse=mse_reconstruction_loss).get(
            reconstruction_loss.lower()
        )
        neg_mll = metrics["neg_mll"]
        neg_entropy = metrics["neg_entropy"]
        loss = (
            loss_weights["reconstruction"] * r_loss
            + loss_weights["neg_mll"] * neg_mll
            + loss_weights["neg_entropy"] * neg_entropy
            + loss_weights["reward"] * metrics["reward_loss"]
        )

        snr_penalty = dlgpd_model.gp.snr_penalty()
        cumulative_metrics["snr_penalty"] += float(snr_penalty.detach() * batch_size)
        loss += snr_penalty[0]

        cumulative_metrics["loss"] += float(loss.detach() * batch_size)

        loss.backward()

        apply_momentum_to_parameters(
            dlgpd_model.reward_model.parameters(), reward_model_momentum
        )

        dlgpd_optimizer.step()

        penalties = dlgpd_model.gp.hyperparameter_penalties()
        penalty = torch.stack(tuple(penalties.values())).sum()
        cumulative_metrics["hp_penalty"] += float(penalty.detach())

        train_bar.set_description(
            "[Train Epoch: {}/{}] ReconLoss: {:.3f}; -MLL: {:.3f}, NegEnt: {:.3f}, Reward: {:.3f}".format(
                epoch,
                epochs,
                cumulative_metrics["bce_reconstruction_loss"] / running_batch_size,
                cumulative_metrics["neg_mll"] / running_batch_size,
                cumulative_metrics["neg_entropy"] / running_batch_size,
                cumulative_metrics["reward_loss"] / running_batch_size,
            )
        )

        actions = data["action"].cuda().float()
        curr_state_belief = data["latent"].current
        next_state_belief = data["latent"].next
        next_state = next_state_belief.sample.cuda().float()
        curr_state = curr_state_belief.sample.cuda().float()

        if missing_evidence > 0:
            evidence_curr_states.append(curr_state[:missing_evidence].detach())
            evidence_next_states.append(next_state[:missing_evidence].detach())
            evidence_actions.append(actions[:missing_evidence].detach())
            rewards = data["reward"]
            evidence_rewards.append(rewards[:missing_evidence].detach())
            missing_evidence -= batch_size

    evidence_curr_states = torch.cat(evidence_curr_states, dim=0)
    evidence_next_states = torch.cat(evidence_next_states, dim=0)
    evidence_actions = torch.cat(evidence_actions, dim=0)
    evidence_rewards = torch.cat(evidence_rewards, dim=0)

    dlgpd_model.gp.set_data(
        curr_states=evidence_curr_states,
        next_states=evidence_next_states,
        actions=evidence_actions,
    )
    dlgpd_model.reward_model.set_data(
        states=evidence_curr_states,
        actions=evidence_actions,
        true_rewards=evidence_rewards,
    )

    snrs = dlgpd_model.gp.signal_to_noise()
    for i, snr in enumerate(snrs):
        summary_writer.add_scalar(f"zz/snr{i}", snr, epoch)
        summary_writer.add_scalar(f"zz/log_snr{i}", snr.log(), epoch)
    snr_penalty = dlgpd_model.gp.snr_penalty()
    summary_writer.add_scalar(f"zz/snr_penalty", snr_penalty, epoch)

    # Scale cumulative metrics
    for k, v in cumulative_metrics.items():
        cumulative_metrics[k] = v / running_batch_size

    # Log metrics, images, ...
    for k, v in cumulative_metrics.items():
        summary_writer.add_scalar(f"loss/train/{k}", v, epoch)

    return cumulative_metrics


def compute_pairwise_metrics(data, dlgpd_model, in_eval_mode):
    metrics = dict()
    actions = data["action"]
    curr_latent_belief = data["latent"].current
    next_latent_belief = data["latent"].next
    next_latent_std = next_latent_belief.std
    next_latent_logstd = torch.log(next_latent_std)
    next_latent = next_latent_belief.sample
    curr_latent = curr_latent_belief.sample
    next_rendering_history = data["rendering_history"].next

    batch_size = actions.shape[0]

    # 1. Reconstruction loss
    bce_reconstruction_loss = (
        dlgpd_model.vae.reconstruction_loss(
            next_latent, next_rendering_history, "bce"
        ).sum()
        / batch_size
    )
    metrics["bce_reconstruction_loss"] = bce_reconstruction_loss

    mse_reconstruction_loss = (
        dlgpd_model.vae.reconstruction_loss(
            next_latent, next_rendering_history, "mse"
        ).sum()
        / batch_size
    )
    metrics["mse_reconstruction_loss"] = mse_reconstruction_loss

    # 2. Negative differential entropy
    if torch.any(torch.isnan(next_latent_logstd)):
        warn(
            "There are NaNs in the encoder std,"
            "probably you are in deterministic mode."
        )
        neg_entropy = torch.zeros(1).cuda()
    else:
        neg_entropy = -1 / 2 * next_latent_logstd.sum() / batch_size
    metrics["neg_entropy"] = neg_entropy

    # 3. Marginal log-likelihood of samples
    scaled_X = dlgpd_model.gp.set_data(
        curr_states=curr_latent, next_states=next_latent, actions=actions
    )
    with gpytorch.settings.prior_mode(True):
        outputs = dlgpd_model.gp.predict(
            scaled_X, suppress_eval_mode_warning=in_eval_mode
        )
        neg_mll = -torch.sum(torch.stack(dlgpd_model.gp.get_mll(outputs, next_latent)))

    metrics["neg_mll"] = neg_mll

    return batch_size, metrics


def compute_reward_metrics(data, dlgpd_model, detach_latent):
    actions = data["action"]
    curr_latent_belief = data["latent"].current
    curr_latent = curr_latent_belief.sample
    if detach_latent:
        curr_latent = curr_latent.detach()
    batch_size = actions.shape[0]
    rewards = data["reward"]
    reward_metrics = dlgpd_model.reward_model.loss_metrics(
        states=curr_latent, actions=actions, true_rewards=rewards
    )
    metrics = {"reward_" + k: v for k, v in reward_metrics.items()}
    return batch_size, metrics


def compute_prediction_metrics(
    data, vae, gp_model, reward_model, filter_length, prediction_length
):
    """
    GP predictions in image space

    Parameters
    ----------
    data: dict of "rendering", "action"; shape [T x n_batch x <dim>]
    vae: VAE
    gp_model: GP
    filter_length: int, number of observations for initial state estimation
    prediction_length: int, number of prediction steps

    Returns
    -------
    last_filtered_state: mean, std
    predicted_states: mean, std
    predicted_images
    """

    if filter_length <= vae.warmup:
        raise ValueError("We need at least {} images for warmup".format(vae.warmup))

    renderings = data["rendering"]
    if renderings.shape[0] < filter_length + prediction_length:
        raise ValueError("Sequences too short")

    # predict sequences
    # as we may do smoothing to infer the encodings, we
    # cannot get filter_encoding by slicing!
    filter_images = renderings[:filter_length]
    filter_encodings = vae.encode_sequence(filter_images)
    actions = data["action"]
    rewards = data["reward"]
    prediction_actions = actions[filter_length - 1 : -1]
    last_filter_belief = tuple([t[-1] for t in filter_encodings])
    last_filter_mean, last_filter_std = last_filter_belief[:2]

    # predict last_filter_belief forward in time
    with gpytorch.settings.skip_posterior_variances(True):
        mean_prop_predictions, _ = gp_model.n_step_mean_prop_forward(
            last_filter_mean,
            last_filter_std,
            actions=prediction_actions,
            n_steps=prediction_length,
        )
    mean_prop_predictions = torch.stack(mean_prop_predictions, dim=0)

    # concat filter_mean
    # and mean_prop_predictions (1+-step predictions)
    filter_mean = filter_encodings[0]
    latents = torch.cat([filter_mean, mean_prop_predictions], dim=0)
    decoded_images = vae.decode_sequence(latents)

    # compute rewards
    # action[-1] is applied to latent[-1] (aligned on right)
    reward_actions = actions[-len(latents) :]
    # we don't have the groundtruth reward for the last
    # (latent, action) pair
    decoded_rewards = reward_model(
        latents[:-1].view(-1, latents.shape[-1]),
        reward_actions[:-1].view(-1, reward_actions.shape[-1]),
    )
    decoded_rewards = decoded_rewards.view(*latents[:-1].shape[:-1])

    mse_loss = F.mse_loss(
        decoded_images, renderings[-decoded_images.shape[0] :], reduction="none"
    ).sum(dim=(2, 3, 4))
    bce_loss = F.binary_cross_entropy(
        decoded_images, renderings[-decoded_images.shape[0] :], reduction="none"
    ).sum(dim=(2, 3, 4))
    mse_loss_reward = F.mse_loss(
        decoded_rewards, rewards[-decoded_rewards.shape[0] :], reduction="none"
    )

    # TODO: Add latent space prediction metrics
    metrics = {}
    n_filter_decodings = decoded_images.shape[0] - prediction_length
    n_filter_rewards = decoded_rewards.shape[0] - prediction_length

    for step in range(prediction_length + 1):
        if step == 0:
            # average over images in "filter" subsequence
            decoding_slice = slice(0, n_filter_decodings)
            reward_slice = slice(0, n_filter_rewards)
        else:
            decoding_slice = slice(
                n_filter_decodings + step - 1, n_filter_decodings + step
            )
            reward_slice = slice(n_filter_rewards + step - 1, n_filter_rewards + step)

        step_loss_bce = bce_loss[decoding_slice, :].mean(dim=0)
        step_loss_mse = mse_loss[decoding_slice, :].mean(dim=0)
        reward_step_loss_mse = mse_loss_reward[reward_slice, :].mean(dim=0)

        metrics[f"prediction/bce/{step}-step"] = step_loss_bce.mean()
        metrics[f"prediction/mse/{step}-step"] = step_loss_mse.mean()
        metrics[f"prediction/reward/{step}-step"] = reward_step_loss_mse.mean()
        metrics[f"prediction/bce/histogram_{step}-step"] = step_loss_bce
        metrics[f"prediction/mse/histogram_{step}-step"] = step_loss_mse

    first_5_pred_slice = slice(n_filter_decodings, n_filter_decodings + 5)
    metrics["prediction/mse/5-step-mean"] = mse_loss[first_5_pred_slice, :].mean()

    first_20_reward_pred_slice = slice(n_filter_rewards, n_filter_rewards + 20)
    metrics["prediction/reward/20-step-mean"] = mse_loss_reward[
        first_20_reward_pred_slice, :
    ].mean()

    return metrics, decoded_images


@torch.no_grad()
@ex.capture
def validate_pairwise(epoch, val_data, dlgpd_model, summary_writer, _run, _log):
    pairwise_metrics = defaultdict(float)
    # Pairwise losses
    running_batch_size = 0
    for data in val_data.pair_loader:
        data["latent"] = encode_batch_of_pairs(data, dlgpd_model.vae)
        batch_size, metrics = compute_pairwise_metrics(
            data, dlgpd_model, in_eval_mode=True
        )
        running_batch_size += batch_size
        for k, v in metrics.items():
            pairwise_metrics[k] += v.detach() * batch_size
    # Log metrics
    for k, v in pairwise_metrics.items():
        tag_name = f"loss/validation/pairwise/{k}"
        v = v / running_batch_size
        pairwise_metrics[k] = v
        summary_writer.add_scalar(tag_name, v, epoch)
        _run.log_scalar(tag_name, float(v), epoch)
    return pairwise_metrics


@torch.no_grad()
@ex.capture
def validate_sequences(
    epoch,
    val_data,
    dlgpd_model,
    val_filter_length,
    val_prediction_length,
    summary_writer,
    _run,
    _log,
):
    # Sequence prediction losses
    data_batch = val_data.get_chunks_as_batch()
    metrics, _ = compute_prediction_metrics(
        data_batch,
        dlgpd_model.vae,
        dlgpd_model.gp,
        dlgpd_model.reward_model,
        filter_length=val_filter_length,
        prediction_length=val_prediction_length,
    )
    # Log metrics
    for k, v in metrics.items():
        tag_name = f"loss/validation/sequence/{k}"
        if k.split("/")[-1].startswith("histogram_"):
            summary_writer.add_histogram(tag_name, v, epoch)
        else:
            summary_writer.add_scalar(tag_name, v, epoch)
            _run.log_scalar(tag_name, float(v), epoch)
    return metrics


@torch.no_grad()
@ex.capture
def validate_reward(epoch, val_data, dlgpd_model, summary_writer, _run, _log):
    reward_metrics = defaultdict(float)
    # Pairwise losses
    running_batch_size = 0
    for data in val_data.pair_loader:
        data["latent"] = encode_batch_of_pairs(data, dlgpd_model.vae)
        batch_size, metrics = compute_reward_metrics(
            data, dlgpd_model, detach_latent=True
        )
        running_batch_size += batch_size
        for k, v in metrics.items():
            reward_metrics[k] += v.detach() * batch_size
    # Log metrics
    for k, v in reward_metrics.items():
        tag_name = f"reward/validation/{k}"
        v = v / running_batch_size
        reward_metrics[k] = v
        summary_writer.add_scalar(tag_name, v, epoch)
        _run.log_scalar(tag_name, float(v), epoch)
    return reward_metrics


def validate_all(epoch, val_data, dlgpd_model, summary_writer):
    # validate sequences first to use evidence from train_epoch run
    # (evidence is overwritten in validate_pairwise)
    sequence_metrics = validate_sequences(
        epoch=epoch,
        val_data=val_data,
        dlgpd_model=dlgpd_model,
        summary_writer=summary_writer,
    )

    pairwise_metrics = validate_pairwise(
        epoch=epoch,
        val_data=val_data,
        dlgpd_model=dlgpd_model,
        summary_writer=summary_writer,
    )

    reward_metrics = validate_reward(
        epoch=epoch,
        val_data=val_data,
        dlgpd_model=dlgpd_model,
        summary_writer=summary_writer,
    )

    val_loss = sequence_metrics["prediction/mse/5-step-mean"].cpu().detach().numpy()
    val_loss = float(val_loss)
    return val_loss


@torch.no_grad()
def visualize_predictions(
    val_data,
    vae,
    gp_model,
    reward_model,
    n_rollouts,
    filter_length,
    prediction_length,
    summary_writer,
    epoch,
):
    n_rollouts = min(val_data.n_chunks, n_rollouts)
    obs, pred = [], []
    rollout_length = val_data.chunk_length
    for rollout_data in val_data.chunk_iterator(max_chunks=n_rollouts):
        # introduce batch dimension
        rollout_data = {k: t.unsqueeze(1) for k, t in rollout_data.items()}
        _, decodings = compute_prediction_metrics(
            rollout_data,
            vae,
            gp_model,
            reward_model,
            filter_length=filter_length,
            prediction_length=prediction_length,
        )
        obs_frames = rollout_data["rendering"][:, 0]
        pred_frames = decodings[:, 0]
        n_missing = obs_frames.shape[0] - pred_frames.shape[0]
        align_frames = torch.ones(
            n_missing, *obs_frames.shape[-3:], device=obs_frames.device
        )
        pad_pred_frames = torch.cat([align_frames, pred_frames], dim=0)
        obs.append(obs_frames)
        pred.append(pad_pred_frames)
    obs = torch.stack(obs, dim=0)
    pred = torch.stack(pred, dim=0)
    blank = torch.ones_like(obs)
    images = torch.cat((obs, pred, blank), dim=1).reshape(-1, 3, 64, 64)
    grid = make_grid(images, nrow=rollout_length)
    summary_writer.add_image("predictions/images", grid, epoch)
    return grid


@ex.main
def forward_dynamics(
    vae_lr,
    gp_lr,
    reward_model_lr,
    latent_size,
    n_train_rollouts_total,
    n_train_rollouts_subset,
    n_val_rollouts_total,
    n_val_rollouts_subset,
    reward_model_adam_beta1,
    reward_model_adam_beta2,
    max_cholesky_size=128,
    verbose=True,
    _run=None,
    _seed=None,
    _log=None,
):
    experiment_dir = get_experiment_directory()

    # Logging
    ex.logger = logging.get_logger(logdir=Path(experiment_dir))
    ex.logger.setLevel(_run.meta_info["options"]["--loglevel"] or "INFO")
    tqdm = lambda *args, **kwargs: tqdm_base(*args, disable=not verbose, **kwargs)
    globals()["tqdm"] = tqdm

    # Make everything reproducible
    torch.manual_seed(_seed)

    # Model saving
    model_path = experiment_dir / "models"
    _log.info(f"Model path: {model_path}")
    model_path.mkdir(parents=True, exist_ok=True)

    # Tensorboard
    tensorboard_path = experiment_dir
    _log.info(f"Tensorboard path: {tensorboard_path}")
    summary_writer = SummaryWriter(str(tensorboard_path))
    logging.declare_custom_scalars(summary_writer)

    # Env
    env_info = get_env_info()
    action_size = env_info.action_size

    # Models
    # VAE
    vae = build_vae(latent_size=latent_size, image_channels=3)
    vae_optim = optim.Adam(vae.parameters(), lr=vae_lr)
    # GPs
    gp_model = build_gp(input_dim=latent_size + action_size, output_dim=latent_size)
    gp_optim = optim.Adam(gp_model.parameters(), lr=gp_lr)
    # Reward model
    reward_model = build_reward_model(latent_size=latent_size, action_size=action_size)
    reward_model_optim = optim.Adam(
        reward_model.parameters(),
        lr=reward_model_lr,
        betas=(reward_model_adam_beta1, reward_model_adam_beta2),
    )

    dlgpd_model = DLGPDModel(vae=vae, gp=gp_model, reward_model=reward_model)
    dlgpd_model.cuda()
    dlgpd_optim = DLGPDOptimizer(
        vae_optim=vae_optim, gp_optim=gp_optim, reward_model_optim=reward_model_optim
    )

    env_name, env_kwargs = get_random_collection_env()
    train_data = load_data(
        env_name=env_name,
        env_kwargs=env_kwargs,
        split_name="train",
        n_rollouts_total=n_train_rollouts_total,
        n_rollouts_subset=n_train_rollouts_subset,
    )
    val_data = load_data(
        env_name=env_name,
        env_kwargs=env_kwargs,
        split_name="val",
        n_rollouts_total=n_val_rollouts_total,
        n_rollouts_subset=n_val_rollouts_subset,
    )

    ex_objects = {"dlgpd_model": dlgpd_model, "dlgpd_optim": dlgpd_optim}

    def checkpoint_fn(epoch=None):
        experiments.save_experiment(
            ex_objects, model_path=model_path, epoch=epoch, _run=None
        )

    # Initial embedding visualization
    dlgpd_model.eval()
    visualize_embeddings(
        val_data=val_data,
        vae=dlgpd_model.vae,
        env_info=env_info,
        summary_writer=summary_writer,
        epoch=0,
    )

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        # Actual training
        # model is set to train() mode at beginning of train_loop()
        best_loss = train_loop(
            train_data=train_data,
            val_data=val_data,
            env_info=env_info,
            dlgpd_model=dlgpd_model,
            dlgpd_optimizer=dlgpd_optim,
            summary_writer=summary_writer,
            checkpoint_fn=checkpoint_fn,
        )

    summary_writer.export_scalars_to_json(experiment_dir / "tb_scalars.json")
    summary_writer.close()

    _run.result = best_loss
    return best_loss


if __name__ == "__main__":
    import sys

    ex.run_commandline(argv=sys.argv)
