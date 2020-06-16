import logging
import math

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from sacred import Ingredient
from torch import nn

gp_ingredient = Ingredient("gp")


logger = logging.getLogger(__name__)
HYPERPARAMETER_PENALTY_DEFAULTS = dict(
    snr=1000,  # Original
    # snr=50,  # Highest confirmed
    ls=100,  # Original
    # ls=50,  # Highest confirmed
    p=30,  # Original
)


@gp_ingredient.capture
def build_gp(
    input_dim,
    output_dim,
    rescale,
    gp_kernel,
    outputscale_gamma_prior,
    lengthscale_gamma_prior,
    snr_type,
    snr_penalty_tolerance,
    snr_penalty_p,
    init_noise_factor,
    noise_bound,
    noise_bound_factor,
):
    gp_model = MultiIndepGP(
        input_dim=input_dim,
        output_dim=output_dim,
        rescale=rescale,
        kernel=gp_kernel,
        lengthscale_gamma_prior=lengthscale_gamma_prior,
        outputscale_gamma_prior=outputscale_gamma_prior,
        snr_type=snr_type,
        snr_penalty_tolerance=snr_penalty_tolerance,
        snr_penalty_p=snr_penalty_p,
        init_noise_factor=init_noise_factor,
        outputscale_bound=1e-2,
        noise_bound=noise_bound,
        noise_bound_factor=noise_bound_factor,
        num_inducing=0,
    )
    return gp_model


class Projection(gpytorch.means.Mean):
    def __init__(self, dim):
        super(Projection, self).__init__()
        self.dim = dim

    def set_data(self, X, y):
        pass

    def forward(self, input):
        return input[..., self.dim]


class TrainDatasetMean(gpytorch.means.Mean):
    def __init__(self):
        super(TrainDatasetMean, self).__init__()
        self.mean = None

    def set_data(self, X, y):
        assert y.dim() == 1
        self.mean = torch.mean(y)

    def forward(self, input):
        assert input.dim() == 2
        if self.mean is None:
            raise ValueError("Not initialized")
        return self.mean[None].expand(input.shape[0])


class TrainDatasetMin(gpytorch.means.Mean):
    def __init__(self):
        super(TrainDatasetMin, self).__init__()
        self.min = None

    def set_data(self, X, y):
        assert y.dim() == 1
        self.min = torch.min(y)

    def forward(self, input):
        assert input.dim() == 2
        if self.min is None:
            raise ValueError("Not initialized")
        return self.min[None].expand(input.shape[0])


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        dim_name,
        mean_module,
        outputscale_gamma_prior,
        lengthscale_gamma_prior,
        kernel,
        noise_bound,
        outputscale_bound,
        init_noise_factor,
        noise_bound_factor,
        num_inducing=0,
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.dim_name = dim_name
        self.mean_module = mean_module
        self.init_noise_factor = init_noise_factor
        self.noise_bound = noise_bound
        self.noise_bound_factor = noise_bound_factor

        if train_x is not None and train_y is not None:
            self.mean_module.set_data(train_x, train_y)

        self.input_dim = train_x.shape[1]

        if kernel.lower() == "rbf":
            inner_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=self.input_dim,
                lengthscale_prior=gpytorch.priors.GammaPrior(*lengthscale_gamma_prior)
                if lengthscale_gamma_prior is not None
                else None,
            )
        else:
            raise ValueError
        self.inner_kernel = inner_kernel

        scale_kernel = gpytorch.kernels.ScaleKernel(
            inner_kernel,
            outputscale_constraint=gpytorch.constraints.GreaterThan(outputscale_bound),
            outputscale_prior=gpytorch.priors.GammaPrior(*outputscale_gamma_prior)
            if outputscale_gamma_prior is not None
            else None,
        )
        self.scale_kernel = scale_kernel

        if num_inducing is not None and num_inducing > 0:
            self.covar_module = gpytorch.kernels.InducingPointKernel(
                self.scale_kernel,
                inducing_points=torch.randn((num_inducing, train_x.shape[-1])),
                likelihood=likelihood,
            )
        else:
            self.covar_module = self.scale_kernel

        self.hyp_initialized = False

    def set_train_data(self, inputs=None, targets=None, strict=True):
        if not self.hyp_initialized and self.training:
            signal_std = torch.std(targets).detach()
            self.scale_kernel.outputscale = signal_std ** 2
            noise_var = self.init_noise_factor * (signal_std ** 2)
            noise_constraint = None
            if self.noise_bound is not None:
                assert self.noise_bound_factor is None
                noise_constraint = gpytorch.constraints.GreaterThan(self.noise_bound)
            if self.noise_bound_factor is not None:
                assert self.noise_bound is None
                assert self.noise_bound_factor < self.init_noise_factor
                noise_bound = self.noise_bound_factor * (signal_std ** 2)
                noise_constraint = gpytorch.constraints.GreaterThan(noise_bound)
            if noise_constraint is not None:
                print(f"Setting noise lower-bound to {noise_constraint}")
                self.likelihood.noise_covar.register_constraint(
                    "raw_noise", noise_constraint
                )
            self.likelihood.noise = noise_var
            self.hyp_initialized = True
        self.mean_module.set_data(inputs, targets)
        super().set_train_data(inputs, targets, strict)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ScaleHandler(object):
    def __init__(self, rescale):
        self.rescale = rescale

    def save_scale(self, file):
        save_dict = {"rescale": self.rescale}
        if self.rescale == "minmax":
            save_dict["min_state"] = self.min_state.detach()
            save_dict["max_state"] = self.max_state.detach()
        elif self.rescale == "standard":
            save_dict["state_mean"] = self.state_mean.detach()
            save_dict["state_std"] = self.state_std.detach()
        elif self.rescale == "none":
            pass
        else:
            raise ValueError
        torch.save(save_dict, file)

    def load_scale(self, file):
        file_dict = torch.load(file)
        if file_dict["rescale"] != self.rescale:
            raise ValueError("rescale type must match")
        if self.rescale == "minmax":
            self.min_state = file_dict["min_state"]
            self.max_state = file_dict["max_state"]
        elif self.rescale == "standard":
            self.state_mean = file_dict["state_mean"]
            self.state_std = file_dict["state_std"]
        elif self.rescale == "none":
            pass
        else:
            raise ValueError

    def copy_from(self, scale_handler):
        if scale_handler.rescale != self.rescale:
            raise ValueError("rescale type must match")
        if self.rescale == "minmax":
            self.min_state = scale_handler.min_state.clone()
            self.max_state = scale_handler.max_state.clone()
        elif self.rescale == "standard":
            self.state_mean = scale_handler.state_mean.clone()
            self.state_std = scale_handler.state_std.clone()
        elif self.rescale == "none":
            pass
        else:
            raise ValueError

    def set_data(self, X):
        if self.rescale == "minmax":
            self.min_state = torch.min(X, dim=0).values
            self.max_state = torch.max(X, dim=0).values
        elif self.rescale == "standard":
            self.state_mean = torch.sum(X, dim=0) / X.shape[0]
            self.state_std = torch.std(X, dim=0) + 1e-2
        elif self.rescale == "none":
            pass
        else:
            raise ValueError

    def to(self, device):
        if self.rescale == "minmax":
            self.min_state = self.min_state.to(device)
            self.max_state = self.max_state.to(device)
        elif self.rescale == "standard":
            self.state_mean = self.state_mean.to(device)
            self.state_std = self.state_std.to(device)

    def scale_states(self, *states):
        if self.rescale == "minmax":
            if len(states) == 1:
                state = states[0]
                return (state - self.min_state) / (
                    self.max_state - self.min_state
                ) * 2 - 1
            else:
                return [
                    (state - self.min_state) / (self.max_state - self.min_state) * 2 - 1
                    for state in states
                ]
        elif self.rescale == "standard":
            if len(states) == 1:
                state = states[0]
                return (state - self.state_mean) / self.state_std
            else:
                return [(state - self.state_mean) / self.state_std for state in states]
        elif self.rescale == "none":
            if len(states) == 1:
                state = states[0]
                return state
            else:
                return states
        else:
            raise ValueError

    def unscale_outputs(self, outputs):
        if self.rescale == "minmax":
            scaled_outputs = [
                # (o + 1) / 2 * (self.max-self.min) + self.min
                (
                    (
                        o
                        + MultivariateNormal(
                            torch.ones_like(o.mean),
                            gpytorch.lazy.ZeroLazyTensor(
                                *o.mean.shape, o.mean.shape[-1], device=o.mean.device
                            ),
                        )
                    )
                    / 2
                    * (self.max_state[i].item() - self.min_state[i].item())
                    + MultivariateNormal(
                        torch.ones_like(o.mean) * self.min_state[i].item(),
                        gpytorch.lazy.ZeroLazyTensor(
                            *o.mean.shape, o.mean.shape[-1], device=o.mean.device
                        ),
                    )
                )
                for i, o in enumerate(outputs)
            ]
        elif self.rescale == "standard":
            scaled_outputs = [
                # (o * state_std) + state_mean
                (
                    (o * self.state_std[i].item())
                    + MultivariateNormal(
                        torch.ones_like(o.mean) * self.state_mean[i].item(),
                        gpytorch.lazy.ZeroLazyTensor(
                            *o.mean.shape, o.mean.shape[-1], device=o.mean.device
                        ),
                    )
                )
                for i, o in enumerate(outputs)
            ]
        elif self.rescale == "none":
            scaled_outputs = outputs
        else:
            raise ValueError
        return scaled_outputs


class MultiIndepGP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        rescale,
        outputscale_bound,
        outputscale_gamma_prior,
        lengthscale_gamma_prior,
        kernel,
        snr_type,
        snr_penalty_tolerance,
        snr_penalty_p,
        init_noise_factor,
        noise_bound,
        noise_bound_factor,
        num_inducing=0,
    ):
        super(MultiIndepGP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        _X = torch.zeros(2, input_dim)

        _y_state = torch.zeros(2, output_dim)
        self.models = nn.ModuleList(
            [
                ExactGPModel(
                    _X,
                    _y_state,
                    dim_name=str(d),
                    mean_module=Projection(d),
                    num_inducing=num_inducing,
                    outputscale_gamma_prior=outputscale_gamma_prior,
                    lengthscale_gamma_prior=lengthscale_gamma_prior,
                    kernel=kernel,
                    outputscale_bound=outputscale_bound,
                    init_noise_factor=init_noise_factor,
                    noise_bound=noise_bound,
                    noise_bound_factor=noise_bound_factor,
                )
                for d in range(output_dim)
            ]
        )

        self.mlls = [
            gpytorch.mlls.ExactMarginalLogLikelihood(m.likelihood, m)
            for m in self.models
        ]

        self.scale_handler = ScaleHandler(rescale)

        for i, model in enumerate(self.models):
            for constraint_name, constraint in model.named_constraints():
                logger.debug(
                    f"[GP {i}] Constraint name: {constraint_name:60} Constraint = {constraint}"
                )
            for prior_name, prior, *_ in model.named_priors():
                logger.debug(
                    f"[GP {i}] Prior name:      {prior_name:60} Prior = {prior}"
                )

        self.snr_type = snr_type
        self.snr_penalty_tolerance = snr_penalty_tolerance
        self.snr_penalty_p = snr_penalty_p

    def set_data(
        self, curr_states, next_states, actions, strict=False, fixed_scale=False
    ):
        """Set data for each"""
        assert all(
            [t.dim() == 2 for t in [curr_states, next_states, actions]]
        ), "Multi-batch states not implemented"
        # determine scale only based on curr_states
        if not fixed_scale:
            self.scale_handler.set_data(curr_states)
        curr_states, next_states = self.scale_handler.scale_states(
            curr_states, next_states
        )
        y = next_states
        X = torch.cat((curr_states, actions), dim=1)
        for i, _m in enumerate(self.models):
            _m.set_train_data(inputs=X, targets=y[:, i], strict=False)
        return X

    def add_data(self, curr_states, next_states, actions):
        assert all(
            [t.dim() == 2 for t in [curr_states, next_states, actions]]
        ), "Multi-batch states not implemented"
        curr_states, next_states = self.scale_handler.scale_states(
            curr_states, next_states
        )
        X = torch.cat((curr_states, actions), dim=1)
        y = next_states
        for i in range(len(self.models)):
            self.models[i] = self.models[i].get_fantasy_model(inputs=X, targets=y[:, i])
        assert next_states.shape[1] == self.output_dim

    def predict(self, scaled_X, suppress_eval_mode_warning=False):
        """ Run predictions on training data """
        if not suppress_eval_mode_warning and not self.training:
            logger.warning("Calling predict() in eval mode seems unusual")
        scaled_outputs = self._inference(scaled_X)
        outputs = self.scale_handler.unscale_outputs(scaled_outputs)
        return outputs

    def forward(self, curr_states, actions):
        """ Run predictions on arbitrary data """
        assert not self.training, (
            "forward() should be called for inference only,"
            "use predict() to get predictions for your training data"
        )
        assert all(
            [t.dim() == 2 for t in [curr_states, actions]]
        ), "Multi-batch states not implemented"

        scaled_states = self.scale_handler.scale_states(curr_states)
        X = torch.cat((scaled_states, actions), dim=-1)
        scaled_outputs = self._inference(X)
        outputs = self.scale_handler.unscale_outputs(scaled_outputs)
        return outputs

    def _inference(self, X):
        scaled_outputs = [model(X) for model in self.models]
        return scaled_outputs

    def get_mll(self, outputs, next_states):
        mlls = [mll(outputs[i], next_states[:, i]) for i, mll in enumerate(self.mlls)]
        return mlls

    def likelihood(self, outputs):
        likelihoods = [m.likelihood(outputs[i]) for i, m in enumerate(self.models)]
        return likelihoods

    def n_step_mean_prop_forward(self, mu, sigma=0, actions=[], **kwargs):
        state_mu, state_sigma = [], []
        for action in actions:
            with gpytorch.settings.fast_pred_var(True):
                prediction = self.likelihood(self(mu, actions=action))
                # prediction = self(mu, actions=actions[i])
                mu = torch.stack([p.mean for p in prediction], dim=1)
                sigma = torch.stack([p.variance for p in prediction], dim=1)
                state_mu.append(mu.clone())
                state_sigma.append(sigma.clone())

        return state_mu, state_sigma

    def hyperparameter_penalties(
        self,
        snr=HYPERPARAMETER_PENALTY_DEFAULTS["snr"],
        ls=HYPERPARAMETER_PENALTY_DEFAULTS["ls"],
        p=HYPERPARAMETER_PENALTY_DEFAULTS["p"],
    ):

        penalties = {}

        for m in self.models:
            std = torch.cat(m.train_inputs, dim=0).std(dim=0)

            if hasattr(m, "base_covar_module"):
                lengthscales = m.base_covar_module.base_kernel.lengthscale
                outputscale = m.base_covar_module.outputscale
            else:
                lengthscales = m.covar_module.base_kernel.lengthscale
                outputscale = m.covar_module.outputscale
            noise = m.likelihood.noise

            # Lengthscale penalty:
            ls_penalties = (
                ((lengthscales.log() - std.log()) / math.log(ls)).pow(p).flatten()
            )
            for input_dim, ls_pen in enumerate(ls_penalties):
                penalties[f"{m.dim_name}/lengthscale/{input_dim}"] = ls_pen

            # Signal to noise penalty
            signal_noise_penalty = (
                ((outputscale.sqrt().log() - noise.sqrt().log()) / math.log(snr))
                .pow(p)
                .sum()
            )
            penalties[f"{m.dim_name}/signal-noise-ratio"] = signal_noise_penalty

        return penalties

    def signal_to_noise(self):
        snrs = []
        for m in self.models:
            outputscale = m.covar_module.outputscale
            noise_variance = m.likelihood.noise
            snr = outputscale / noise_variance
            if self.snr_type == "pilco":
                snrs.append(snr.sqrt())
            elif self.snr_type == "squared":
                snrs.append(snr)
            else:
                raise ValueError("Unknown SNR type")
        return snrs

    def snr_penalty(self):
        penalties = []
        snrs = self.signal_to_noise()
        for snr in snrs:
            penalties.append(
                (snr.log() / math.log(self.snr_penalty_tolerance)).pow(
                    self.snr_penalty_p
                )
            )
        penalty = sum(penalties)
        return penalty
