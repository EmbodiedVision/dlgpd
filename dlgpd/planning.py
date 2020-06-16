""" Perform control """

import os
from pathlib import Path

import numpy as np
import plac
import torch
from torchvision import transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)  # for loading the shuffled train-data

from .data.loader import (
    load_data,
    get_env_info,
    get_random_collection_env,
    get_control_env,
    make_env,
    encode_batch_of_pairs,
)
from .models.gp import ExactGPModel
from .models.cem import TransitionModel, LearnedRewardModel, MPCPlannerCem
from .utils.loading import load_model_skeleton


def get_screen(env, with_noise, rendering_noise_rng):
    screen = env.render(mode="rgb_array")
    screen = screen.transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    if with_noise:
        screen += 0.01 * rendering_noise_rng.randn(*screen.shape)
        screen = np.clip(screen, 0, 1)
    screen = torch.from_numpy(screen)
    transform_chain = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform_chain)
    screen = transform(screen)
    return screen.float().cuda()


@torch.no_grad()
def run_rollout(
    env,
    planner,
    vae,
    n_warmup_observations,
    seed=None,
    max_steps=None,
    with_observation_noise=True,
    verbose=True,
):
    rendering_history = []
    latent_history = []
    action_history = []
    reward_history = []
    planning_history = []

    full_reward = 0
    if seed is None:
        seed = np.random.randint(0, int(1e8))

    env.seed(seed)
    env.action_space.seed(seed)
    rendering_noise_rng = np.random.RandomState(seed)

    env.reset()

    curr_image = get_screen(env, with_observation_noise, rendering_noise_rng)
    rendering_history.append(curr_image)

    done = False
    step = 0
    while not done:
        # not enough observations collected to infer current state
        if len(rendering_history) < n_warmup_observations:
            action = 0.1 * env.action_space.sample()
            planned_trajectory = {}
        else:
            step += 1
            # infer current state from history of observations
            observations = torch.stack(rendering_history, dim=0).unsqueeze(1)
            sequence_encoding = vae.encode_sequence(observations)
            state_belief = [
                sequence_encoding.mean[-1, 0, :].unsqueeze(0).cuda(),
                torch.diag_embed(sequence_encoding.std[-1, 0, :] ** 2)
                .unsqueeze(0)
                .cuda(),
            ]
            latent_history.append(sequence_encoding.mean[-1, 0, :])
            action_range = (env.action_space.low[0], env.action_space.high[0])
            planner_return = planner(state_belief, action_range=action_range)
            if isinstance(planner_return, tuple):
                action, planned_trajectory = planner_return
            else:
                action, planned_trajectory = planner_return, {}
            action = action.detach().cpu().flatten()
            planned_trajectory = {
                k: v.detach().cpu().numpy() for k, v in planned_trajectory.items()
            }

        next_state, reward, done, _ = env.step(action)
        full_reward += reward

        next_image = get_screen(env, with_observation_noise, rendering_noise_rng)
        action_history.append(action)
        rendering_history.append(next_image)
        reward_history.append(reward)
        planning_history.append(planned_trajectory)

        if verbose:
            print(dict(step=step, action=float(action), reward=float(reward)))

        if max_steps and step >= max_steps:
            break

    if verbose:
        print(f"Collected reward: {full_reward}")

    return {
        "rendering": [t.cpu().numpy() for t in rendering_history],
        "action": action_history,
        "latent": [t.cpu().numpy() for t in latent_history],
        "reward": reward_history,
        "total_reward": full_reward,
        "planned_trajectories": planning_history,
    }


def get_gp_evidence(evidence_data, vae):
    pair_data = evidence_data.get_pairs_as_batch()
    # use mean as evidence
    latents = encode_batch_of_pairs(pair_data, vae)
    curr_states = latents.current.mean.detach().cuda()
    actions = pair_data["action"].cuda()
    next_states = latents.next.mean.detach().cuda()
    rewards = pair_data["reward"].cuda()
    return curr_states, actions, next_states, rewards


def load_train_data(config, vae):
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
    pair_batch = train_data.get_pairs_as_batch(max_pairs=1024)
    latent = encode_batch_of_pairs(pair_batch, vae)
    current = latent.current.sample.detach().cuda()
    next = latent.next.sample.detach().cuda()
    action = pair_batch["action"].detach().cuda()
    reward = pair_batch["reward"].detach().cuda()
    return current, action, next, reward


def load_evidence(
    task_name,
    variation_name,
    bucket_size,
    bucket_seed,
    recollect_evidence,
    data_base_dir,
):
    if recollect_evidence:
        env_name, env_kwargs = get_random_collection_env(task_name, variation_name)
    else:
        env_name, env_kwargs = get_random_collection_env(task_name)
    data = load_data(
        env_name,
        env_kwargs,
        split_name=f"evidence_{bucket_seed}",
        n_rollouts_total=200,
        n_rollouts_subset=bucket_size,
        rollout_length=30,
        chunk_length=30,
        batch_size_chunks=1,
        batch_size_pairs=512,
        shuffle_batches_chunks=False,
        shuffle_batches_pairs=False,
        data_base_dir=data_base_dir,
        subset_shuffle_seed=None,
    )
    return data


def debug_gp(gp_model):
    for m in gp_model.modules():
        if isinstance(m, ExactGPModel):
            print("GP module: ")
            print(
                f"Noise: {m.likelihood.noise},"
                f"Scale: {m.scale_kernel.outputscale},"
                f"Lengthscales: {m.inner_kernel.lengthscale}"
            )


@plac.annotations(
    model_path=plac.Annotation(kind="positional", type=str),
    variation_name=plac.Annotation(kind="option", type=str),
    evidence_bucket_size=plac.Annotation(kind="option", type=int),
    evidence_bucket_seed=plac.Annotation(kind="option", type=int),
    evidence_subdirectory=plac.Annotation(kind="option", type=str),
    rollout_seed=plac.Annotation(kind="option", type=int),
    recollect_evidence=plac.Annotation(kind="flag"),
)
def main(
    model_path,
    variation_name="standard",
    evidence_bucket_size=50,
    evidence_bucket_seed=0,
    evidence_subdirectory="evidence_paper",
    rollout_seed=0,
    recollect_evidence=False,
):
    torch.manual_seed(rollout_seed)
    np.random.seed(rollout_seed)

    planning_horizon = 20
    max_steps = 150
    n_warmup_observations = 2
    with_observation_noise = True

    run_dir = Path(model_path).parent.parent
    rollouts_path = (
        run_dir.joinpath("rollouts")
        .joinpath(f"variation_{variation_name}")
        .joinpath(evidence_subdirectory)
        .joinpath(
            f"bucket_{evidence_bucket_size}_"
            f"{evidence_bucket_seed}_"
            f"recollect_{recollect_evidence}_"
            f"noise_{with_observation_noise}"
        )
    )
    rollout_file = os.path.join(rollouts_path, f"rollout_{rollout_seed}.npz")
    if os.path.isfile(rollout_file):
        print("Rollout already exists")
        exit(0)

    os.makedirs(str(rollouts_path), exist_ok=True)

    # 1) Load config and model skeleton (without loading state dict)
    config, model = load_model_skeleton(model_path)

    # 2) Load state dict to initialize VAE
    model.load_state_dict(torch.load(model_path))

    env_info = get_env_info(config["data"]["task_name"])

    # 3) Set train-data in train mode (to initialize constraints and scales)
    model.train()
    train_curr, train_actions, train_next, train_rewards = load_train_data(
        config, model.vae
    )
    model.gp.set_data(
        curr_states=train_curr, actions=train_actions, next_states=train_next
    )
    model.reward_model.set_data(
        states=train_curr, actions=train_actions, true_rewards=train_rewards
    )

    # 3) Load state_dict again to overwrite parameters initialized by set_data()
    model.load_state_dict(torch.load(model_path))

    # 4) Set evidence data in eval mode with fixed scale
    model.eval()

    data_dir = config["data"]["data_base_dir"]
    evidence_dir = os.path.join(data_dir, evidence_subdirectory)
    evidence = load_evidence(
        config["data"]["task_name"],
        variation_name,
        evidence_bucket_size,
        evidence_bucket_seed,
        recollect_evidence,
        data_base_dir=evidence_dir,
    )

    curr_states, actions, next_states, rewards = get_gp_evidence(evidence, model.vae)

    print(f"Using {curr_states.shape[0]} transitions as evidence")

    model.gp.set_data(
        curr_states=curr_states,
        actions=actions,
        next_states=next_states,
        fixed_scale=True,
    )

    model.reward_model.set_data(
        states=curr_states, actions=actions, true_rewards=rewards, fixed_scale=True
    )

    print(f"Transition GP: {debug_gp(model.gp)}")
    print(f"Reward GP: {debug_gp(model.reward_model)}")

    # Define MPC Model
    transition_model = TransitionModel(model.gp)
    transition_model.eval()
    reward_model = LearnedRewardModel(model.reward_model)
    reward_model.eval()
    planner = MPCPlannerCem(
        action_dim=env_info.action_size,
        transition_model=transition_model,
        reward_model=reward_model,
        planning_horizon=planning_horizon,
        verbose=False,
        return_planned_trajectory=True,
        seed=rollout_seed,
    )

    env_name, env_kwargs = get_control_env(config["data"]["task_name"], variation_name)
    env = make_env(env_name, env_kwargs)

    rollout = run_rollout(
        env=env,
        planner=planner,
        vae=model.vae,
        n_warmup_observations=n_warmup_observations,
        seed=rollout_seed,
        max_steps=max_steps,
        with_observation_noise=with_observation_noise,
    )

    env.close()

    rollout["model_path"] = model_path
    np.savez(rollout_file, rollout)


if __name__ == "__main__":
    with torch.no_grad():
        plac.call(main)
