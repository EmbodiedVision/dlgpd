import numpy as np
import os
import pandas as pd
from glob import glob
from joblib import delayed, Parallel
from tqdm import tqdm

# -------------------------------------------
# Define the experiments you want to evaluate
# (subdirectories in experiments/)
EXPERIMENTS = ["pretrained_1", "pretrained_2", "pretrained_3"]
# -------------------------------------------

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
os.chdir(MODULE_PATH)

rollout_base_dir = f"experiments/dlgpd/"

MAX_LENGTH = 150

EVIDENCE_ENV_KWARGS = {
    "invactions": "action_factor=-1",
    "heavierpole": "m=1.5",
    "lighterpole": "m=0.2",
}

SIM_ENV_KWARGS = {
    "standard": "init_type=bottom",
    "invactions": "action_factor=-1&init_type=bottom",
    "heavierpole": "init_type=bottom&m=1.5",
    "lighterpole": "init_type=bottom&m=0.2",
}


def collect_rollouts_for_experiment(experiment_name):
    rollouts = []

    for rollout_file in glob(
        os.path.join(rollout_base_dir, experiment_name, "rollouts/*/*/*/rollout_*.npz")
    ):
        (
            variation_dir,
            evidence_dir,
            bucket_dir,
            rollout_filename,
        ) = rollout_file.split("/")[-4:]
        rollout_data = np.load(rollout_file, allow_pickle=True)
        variation_name = variation_dir.split("_")[1]
        bucket_parts = bucket_dir.split("_")
        evidence_bucket_size = int(bucket_parts[1])
        evidence_bucket_seed = bucket_parts[2]

        if bucket_parts[4] == "True":
            method_name = "dlgpd_matching"
            evidence_env_kwargs = EVIDENCE_ENV_KWARGS[variation_name]
        elif bucket_parts[4] == "False":
            method_name = "dlgpd_mismatching"
            evidence_env_kwargs = "from_evidence_pool"
        else:
            raise ValueError

        if len(bucket_parts) == 5:
            noise = True
        else:
            noise = bucket_parts[6] == "True"

        if not noise:
            continue

        if evidence_dir != "evidence_paper":
            continue

        rollout_seed = int(rollout_filename.split(".")[0].split("_")[1])

        rewards = np.array([float(r) for r in rollout_data["arr_0"].item()["reward"]])
        rewards = rewards[:MAX_LENGTH]
        last_rewards = rewards[125:]
        working_mask = last_rewards > -1
        task_solved = working_mask.all(axis=0)
        rollout_return = np.sum(rewards)

        rollouts.append(
            {
                "experiment_name": experiment_name,
                "run_id": experiment_name,
                "variation_name": variation_name,
                "evidence_subdir": evidence_dir,
                "bucket_size": evidence_bucket_size,
                "bucket_seed": evidence_bucket_seed,
                "noise": noise,
                "method_name": method_name,
                "rollout_seed": rollout_seed,
                "return": rollout_return,
                "task_solved": task_solved,
                "sim_env_name": "NoActionRendererPendulum-v0",
                "sim_env_kwargs": SIM_ENV_KWARGS[variation_name],
                "evidence_env_name": "RandomInitPendulum-v0",
                "evidence_env_kwargs": evidence_env_kwargs,
            }
        )

    pd.DataFrame(rollouts).to_pickle(
        os.path.join(rollout_base_dir, experiment_name, "rollout_dataframe.pkl")
    )


Parallel(n_jobs=12)(
    delayed(collect_rollouts_for_experiment)(experiment_name)
    for experiment_name in tqdm(EXPERIMENTS)
)
