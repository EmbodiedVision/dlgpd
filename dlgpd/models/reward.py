import math

import gpytorch
import torch
from sacred import Ingredient
from torch import nn

from .gp import ExactGPModel, TrainDatasetMean, TrainDatasetMin, ScaleHandler

reward_ingredient = Ingredient("reward")


@reward_ingredient.capture
def build_reward_model(
    latent_size,
    action_size,
    rescale,
    mean_fn_name,
    lengthscale_gamma_prior,
    outputscale_gamma_prior,
    snr_type,
    snr_penalty_tolerance,
    snr_penalty_p,
    init_noise_factor,
    noise_bound,
    noise_bound_factor,
):
    reward_gp = RewardGP(
        latent_size=latent_size,
        action_size=action_size,
        rescale=rescale,
        mean_fn_name=mean_fn_name,
        lengthscale_gamma_prior=lengthscale_gamma_prior,
        outputscale_gamma_prior=outputscale_gamma_prior,
        snr_type=snr_type,
        snr_penalty_tolerance=snr_penalty_tolerance,
        snr_penalty_p=snr_penalty_p,
        init_noise_factor=init_noise_factor,
        noise_bound=noise_bound,
        noise_bound_factor=noise_bound_factor,
    )
    return reward_gp


class RewardGP(nn.Module):
    def __init__(
        self,
        latent_size,
        action_size,
        rescale,
        mean_fn_name,
        lengthscale_gamma_prior,
        outputscale_gamma_prior,
        snr_type,
        snr_penalty_tolerance,
        snr_penalty_p,
        init_noise_factor,
        noise_bound,
        noise_bound_factor,
    ):
        super().__init__()
        input_dim = latent_size + action_size
        _X = torch.zeros(2, input_dim)
        _y_reward = torch.zeros(2)
        self.snr_type = snr_type
        self.snr_penalty_tolerance = snr_penalty_tolerance
        self.snr_penalty_p = snr_penalty_p

        self.gp = ExactGPModel(
            _X,
            _y_reward,
            dim_name="reward",
            mean_module={
                "empirical_mean": TrainDatasetMean(),
                "empirical_min": TrainDatasetMin(),
            }[mean_fn_name],
            outputscale_gamma_prior=outputscale_gamma_prior,
            lengthscale_gamma_prior=lengthscale_gamma_prior,
            kernel="rbf",
            noise_bound=noise_bound,
            outputscale_bound=1e-2,
            init_noise_factor=init_noise_factor,
            noise_bound_factor=noise_bound_factor,
            num_inducing=0,
        )

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.transition_gp = None
        self.scale_handler = ScaleHandler(rescale)

    def set_data(self, states, actions, true_rewards, fixed_scale=False):
        if not fixed_scale:
            self.scale_handler.set_data(states)
        scaled_states = self.scale_handler.scale_states(states)
        X = torch.cat([scaled_states, actions], dim=-1)
        y = true_rewards
        self.gp.set_train_data(inputs=X, targets=y, strict=False)

    def forward(self, states, actions, with_variance=False):
        scaled_states = self.scale_handler.scale_states(states)
        X = torch.cat([scaled_states, actions], dim=-1)
        with gpytorch.settings.fast_pred_var(True):
            # no need for unscaling here
            outputs = self.gp(X)
            prediction = self.gp.likelihood(outputs)
        if with_variance:
            return prediction.mean, prediction.variance
        else:
            return prediction.mean

    def _signal_to_noise(self):
        outputscale = self.gp.covar_module.outputscale
        noise_variance = self.gp.likelihood.noise
        if self.snr_type == "pilco":
            snr = (outputscale / noise_variance).sqrt()
        elif self.snr_type == "squared":
            snr = outputscale / noise_variance
        else:
            raise ValueError("Unknown SNR type")
        return snr

    def _snr_penalty(self):
        snr = self._signal_to_noise()
        penalty = (snr.log() / math.log(self.snr_penalty_tolerance)).pow(
            self.snr_penalty_p
        )
        return penalty

    def loss_metrics(self, states, actions, true_rewards):
        self.set_data(states, actions, true_rewards)
        scaled_states = self.scale_handler.scale_states(states)
        X = torch.cat([scaled_states, actions], dim=-1)
        with gpytorch.settings.prior_mode(True):
            # no need for unscaling here
            outputs = self.gp(X)
            neg_mll = -self.mll(outputs, true_rewards)

        snr_penalty = self._snr_penalty()
        loss_components = {
            "neg_mll": neg_mll,
            "snr_penalty": snr_penalty,
            "loss": neg_mll + snr_penalty,
        }
        return loss_components
