import json
import os
import re
from pathlib import Path
import pandas as pd
import torch
import warnings
import inspect
import json
from ..models.vae import build_vae
from ..models.gp import build_gp
from ..models.reward import build_reward_model
from ..models.dlgpd import DLGPDModel
from ..data.loader import get_env_info


def kwargs_invoke(fcn, kwargs):
    sig = inspect.signature(fcn, follow_wrapped=True)
    reduced_kwargs = {k: kwargs[k] for k in sig.parameters.keys()}
    return fcn(**reduced_kwargs)


def load_model_skeleton(model_path):
    config_path = Path(model_path).parent.parent.joinpath("config.json")
    with open(str(config_path), "r") as f:
        config = json.load(f)
    env_info = get_env_info(config["data"]["task_name"])
    latent_size = config["latent_size"]
    action_size = env_info.action_size
    vae = kwargs_invoke(
        build_vae, dict(**config["vae"], latent_size=latent_size, image_channels=3)
    )
    gp_model = kwargs_invoke(
        build_gp,
        dict(
            **config["gp"], input_dim=latent_size + action_size, output_dim=latent_size
        ),
    )
    reward_model = kwargs_invoke(
        build_reward_model,
        dict(**config["reward"], latent_size=latent_size, action_size=action_size),
    )
    dlgpd_model = DLGPDModel(vae=vae, gp=gp_model, reward_model=reward_model)
    dlgpd_model.cuda()
    return config, dlgpd_model
