import gpytorch
import numpy as np
import torch
from tqdm import trange


class TransitionModel(torch.nn.Module):
    def __init__(self, gp_model):
        super().__init__()
        self.gp_model = gp_model

    def predict(self, state_beliefs, actions):
        """
        Predict state belief forward in time

        Parameters
        ----------
        state_belief: tuple of ([batch, state_dim], [batch, state_dim, state_dim])
            Distribution of initial state (mean and covariance)
        actions: [horizon, batch, action_dim]
            Actions

        Returns
        ----------
        predicted_state_beliefs: tuple of ([horizon+1, batch, state_dim],
                                           [horizon+1, batch, state_dim, state_dim])
            Predicted states (including initial state)
        """
        state_belief_mean, state_belief_covariance = state_beliefs
        state_belief_mean, state_belief_covariance = (
            state_belief_mean.cuda(),
            state_belief_covariance.cuda(),
        )

        with gpytorch.settings.lazily_evaluate_kernels(state=True):
            mp_predictions, mp_variances = self.gp_model.n_step_mean_prop_forward(
                state_belief_mean, state_belief_covariance, actions=actions
            )

        mp_predictions = torch.stack([state_belief_mean] + mp_predictions, dim=0)
        mp_variances = torch.stack(mp_variances, dim=0)
        mp_variances = torch.diag_embed(mp_variances)
        mp_variances = torch.cat(
            (state_belief_covariance.unsqueeze(0), mp_variances), dim=0
        )

        return mp_predictions, mp_variances


class LearnedRewardModel(torch.nn.Module):
    def __init__(self, reward_model):
        super().__init__()
        self.reward_model = reward_model

    def forward(self, state_beliefs, actions, n_samples=5):
        state_belief_mean, state_belief_covariance = state_beliefs

        batch_size, state_dim = state_belief_mean.shape
        _, action_dim = actions.shape

        variances = torch.diagonal(state_belief_covariance, dim1=1, dim2=2)
        std = variances.sqrt()

        eps = torch.randn(
            (batch_size, n_samples, state_dim),
            dtype=state_belief_mean.dtype,
            device=state_belief_mean.device,
        )
        sampled = state_belief_mean.unsqueeze(1) + std.unsqueeze(1) * eps
        sampled = sampled.reshape(batch_size * n_samples, state_dim)
        repeated_actions = (
            actions.unsqueeze(1)
            .repeat((1, n_samples, 1))
            .reshape(batch_size * n_samples, action_dim)
        )

        reward = self.reward_model(sampled, repeated_actions, with_variance=False)
        reward = reward.reshape(batch_size, n_samples, 1).mean(dim=1)
        return reward


class MPCPlannerCem(torch.nn.Module):
    def __init__(
        self,
        action_dim,
        transition_model,
        reward_model,
        planning_horizon,
        optimisation_iters=30,
        candidates=1000,
        top_candidates=100,
        tol=0.01,
        verbose=True,
        return_planned_trajectory=False,
        seed=None,
    ):
        """
        MPCPlannerCem

        Parameters
        ----------
        action_dim: int
            Dimensionality of action
        transition_model: TransitionModel
            Transition model
        reward_model: RewardModel
            Reward model
        planning_horizon: int
            Planning horizon
        optimisation_iters: int
            Number of CEM iterations
        candidates: int
            Number of candidates per iteration
        top_candidates: int
            Number of best candidates to refit belief per iteration
        """
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.tol = tol
        self.verbose = verbose
        self.return_planned_trajectory = return_planned_trajectory
        self.rng = np.random.RandomState(seed)

    def forward(self, initial_state_belief, action_range):
        """
        Compute optimal action for current state

        Parameters
        ----------
        initial_state_belief: tuple of ([batch, state_dim], [batch, state_dim, state_dim])
            Distribution of initial state (mean and covariance)
        """
        B = initial_state_belief[0].shape[0]
        device = initial_state_belief[0].device
        expanded_state_belief = [
            (
                t.unsqueeze(dim=1)
                .expand(t.shape[0], self.candidates, *t.shape[1:])
                .reshape(-1, *t.shape[1:])
            )
            for t in initial_state_belief
        ]

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        assert action_range[0] == -action_range[1]
        action_mean, action_std_dev = (
            torch.zeros(self.planning_horizon, B, 1, self.action_dim, device=device),
            torch.ones(self.planning_horizon, B, 1, self.action_dim, device=device)
            * action_range[1],
        )
        bar = trange(self.optimisation_iters, disable=not self.verbose)
        for _ in bar:
            # Sample actions (time x batch x candidates x actions)
            random_samples = self.rng.randn(
                self.planning_horizon, B, self.candidates, self.action_dim
            ).astype(np.float32)
            random_samples = torch.from_numpy(random_samples).to(
                device=action_mean.device
            )
            actions = (action_mean + action_std_dev * random_samples).view(
                self.planning_horizon, B * self.candidates, self.action_dim
            )
            actions = actions.clamp(*action_range)

            # Predict in latent space
            state_belief = self.transition_model.predict(expanded_state_belief, actions)

            # Calculate expected returns
            A = actions.shape[-1]
            raw_returns = self.reward_model(
                [
                    t[:-1].view(-1, *t.shape[2:]) for t in state_belief
                ],  # Drop last state and collapse planning-horizon and candidates
                actions.view(-1, A),
            ).view(self.planning_horizon, -1)
            returns = raw_returns.sum(dim=0)

            # Select action sequences with highest return
            _, topk = returns.reshape(B, self.candidates).topk(
                self.top_candidates, dim=1, largest=True, sorted=False
            )
            topk += self.candidates * torch.arange(
                0, B, dtype=torch.int64, device=topk.device
            ).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(
                self.planning_horizon, B, self.top_candidates, self.action_dim
            )
            # best_actions = planning_horizon x batch x top_candidates x action_dim

            # Update action belief with statistics of best action sequences
            action_mean = best_actions.mean(
                dim=2, keepdim=True
            )  # Mean of all candidates
            # action_mean = planning_horizon x batch x 1 x action_dim
            action_std_dev = best_actions.std(dim=2, unbiased=False, keepdim=True)

            curr_best_action = action_mean[0]
            curr_best_action_std = action_std_dev[0]
            std = curr_best_action_std.item()
            bar.set_description(
                f"Best action: {curr_best_action.item():.3f} +- {curr_best_action_std.item():.5f}"
            )
            if std <= self.tol:
                bar.close()
                break

        selected_action = action_mean.squeeze(dim=1)[0]
        if self.return_planned_trajectory:
            state_belief = self.transition_model.predict(
                initial_state_belief, action_mean.squeeze(dim=1)
            )
            planned_trajectory = {"action": action_mean, "state_mean": state_belief[0]}
            return selected_action, planned_trajectory
        else:
            return selected_action
