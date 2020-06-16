import tempfile
import itertools
import os
from pathlib import Path

# -------------------------------------------
# Define the experiments and training epochs which
# you want to run planning rollouts for (as tuple).
# If the 'epoch' field is None, use the model with
# lowest validation error.
# (subdirectories in experiments/)
ID_LIST = ["pretrained_1", "pretrained_2", "pretrained_3"]
EXPERIMENTS = [(str(id), 2000) for id in ID_LIST]
# -------------------------------------------

VARIATIONS = ["standard", "invactions", "heavierpole", "lighterpole"]
BUCKET_SIZES = [10, 20, 30, 50, 70, 100, 200]
BUCKET_SEEDS = [0, 1, 2]
RECOLLECT_EVIDENCE = [True, False]
ROLLOUT_SEEDS = [0, 1, 2]

EVIDENCE_SUBDIRECTORY = "evidence_paper"

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
os.chdir(MODULE_PATH)

commands = []
for (
    experiment,
    variation_name,
    bucket_size,
    bucket_seed,
    rollout_seed,
    recollect,
) in itertools.product(
    EXPERIMENTS,
    VARIATIONS,
    BUCKET_SIZES,
    BUCKET_SEEDS,
    ROLLOUT_SEEDS,
    RECOLLECT_EVIDENCE,
):
    if variation_name == "standard" and recollect:
        # No need to recollect evidence
        continue
    if experiment[1] is None:
        model_file = "dlgpd_model.pt"
    else:
        model_file = f"epoch{experiment[1]}_dlgpd_model.pt"
    model_path = f"experiments/dlgpd/{experiment[0]}/models/{model_file}"

    run_dir = Path(f"experiments/dlgpd/{experiment[0]}")
    rollouts_path = (
        run_dir.joinpath("rollouts")
        .joinpath(f"variation_{variation_name}")
        .joinpath(EVIDENCE_SUBDIRECTORY)
        .joinpath(
            f"bucket_{bucket_size}_"
            f"{bucket_seed}_"
            f"recollect_{recollect}_"
            f"noise_True"
        )
    )
    rollout_file = os.path.join(rollouts_path, f"rollout_{rollout_seed}.npz")
    if os.path.isfile(rollout_file):
        print(f"Rollout {rollout_file} already exists")
        continue

    command = (
        "python -m dlgpd.planning "
        f"-variation-name {variation_name} "
        f"-evidence-bucket-size {bucket_size} "
        f"-evidence-bucket-seed {bucket_seed} "
        f"-evidence-subdirectory {EVIDENCE_SUBDIRECTORY} "
        f"-rollout-seed {rollout_seed} "
        + ("-recollect-evidence " if recollect else " ")
        + model_path
    )
    commands.append(command)

# write commands to file
with open("scripts_planning_evaluation/planning_jobs.txt", "w") as file_handler:
    for item in commands:
        file_handler.write("{}\n".format(item))

print(
    f"List of {len(commands)} planning jobs (one per line) "
    f"written to planning_jobs.txt"
)
print("Use any job submission system of your favor to run these jobs.")
