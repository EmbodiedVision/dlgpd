"""Utilities relating to logging: python logging, Tensorboard, Sacred"""
import logging
import os

import colorlog
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TqdmHandler(logging.StreamHandler):
    """
    Replacement for the standard logging.StreamHandler to handle tqdm well
    """

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(_run=None, logdir=".logs"):
    # fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
    fmt = "[%(levelname)s][%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handlers = []

    colored_fmt = (
        # '|%(green)s%(asctime)s%(reset)s'
        # '|%(log_color)s%(levelname)s%(reset)s'
        # '|%(message_log_color)s%(name)s%(reset)s'
        # '| '
        "[%(green)s%(asctime)s%(reset)s]"
        "[%(log_color)s%(levelname)s%(reset)s]"
        "[%(message_log_color)s%(name)s%(reset)s]"
        "\n  "
        "%(message_log_color)s%(message)s"
    )
    colored_formatter = colorlog.ColoredFormatter(
        fmt=colored_fmt,
        datefmt=datefmt,
        secondary_log_colors={
            "message": {"DEBUG": "cyan", "ERROR": "red", "CRITICAL": "red"}
        },
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "SUCCESS:": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    tqdm_handler = TqdmHandler()
    tqdm_handler.setFormatter(colored_formatter)
    handlers.append(tqdm_handler)

    os.makedirs(logdir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(logdir, "log"))
    file_handler.setFormatter(colored_formatter)
    handlers.append(file_handler)

    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(tqdm_handler)
    logger.addHandler(file_handler)
    if _run is None:
        level = "INFO"
    else:
        level = _run.meta_info["options"]["--loglevel"] or "INFO"
    logger.setLevel(level)
    logger.setLevel("INFO")

    logging.captureWarnings(True)
    # logging.basicConfig(handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

    return logger


def log_gp_hyperparameters(model, writer, step):
    """Log the GP hyperparameters in tensorboard histograms"""
    for name, value in model.named_parameters():
        name = name.replace("raw_", "")

        if name.startswith("models."):

            _, number, *params = name.split(".")

            curr = model.models[int(number)]
            for p in params:
                curr = getattr(curr, p)  # Recursive getattr
            value = curr

            # Single-item list to scalar
            # Lengthscales
            if "lengthscale" in name:
                for j, v in enumerate(value.flatten()):
                    writer.add_scalar(
                        f"z_{model.__class__.__name__}_parameters/"
                        + f"{number}/{params[-1]}/{j}",
                        v,
                        step,
                    )

            elif value.numel() == 1:
                value = value.item()
                writer.add_scalar(
                    f"z_{model.__class__.__name__}_parameters/{number}/{params[-1]}",
                    value,
                    step,
                )

            elif "inducing_points" in name:
                writer.add_embedding(
                    value, global_step=str(step), tag=f"{params[-1]}_{number}"
                )


def declare_custom_scalars(writer, **kwargs):
    # Group the GP parameters
    gp_parameter_grouping = {
        "Lengthscales 0": [
            "Multiline",
            [
                "z_MultiIndepGP_parameters/0/lengthscale/0",
                "z_MultiIndepGP_parameters/1/lengthscale/0",
                "z_MultiIndepGP_parameters/2/lengthscale/0",
            ],
        ],
        "Lengthscales 1": [
            "Multiline",
            [
                "z_MultiIndepGP_parameters/0/lengthscale/1",
                "z_MultiIndepGP_parameters/1/lengthscale/1",
                "z_MultiIndepGP_parameters/2/lengthscale/1",
            ],
        ],
        "Lengthscales 2": [
            "Multiline",
            [
                "z_MultiIndepGP_parameters/0/lengthscale/2",
                "z_MultiIndepGP_parameters/1/lengthscale/2",
                "z_MultiIndepGP_parameters/2/lengthscale/2",
            ],
        ],
        "Lengthscales 3": [
            "Multiline",
            [
                "z_MultiIndepGP_parameters/0/lengthscale/3",
                "z_MultiIndepGP_parameters/1/lengthscale/3",
                "z_MultiIndepGP_parameters/2/lengthscale/3",
            ],
        ],
        "Outputscales": [
            "Multiline",
            [
                "z_MultiIndepGP_parameters/0/outputscale",
                "z_MultiIndepGP_parameters/1/outputscale",
                "z_MultiIndepGP_parameters/2/outputscale",
            ],
        ],
        "Noises": [
            "Multiline",
            [
                "z_MultiIndepGP_parameters/0/noise",
                "z_MultiIndepGP_parameters/1/noise",
                "z_MultiIndepGP_parameters/2/noise",
            ],
        ],
    }

    prediction_losses = {
        "image predictions (mse) (first 5)": [
            "multiline",
            [f"loss/validation/sequence/prediction/mse/{k}-step" for k in range(6)],
        ],
        "image predictions (bce) (first 5)": [
            "multiline",
            [f"loss/validation/sequence/prediction/bce/{k}-step" for k in range(6)],
        ],
        "image predictions (mse) (first 10)": [
            "multiline",
            [f"loss/validation/sequence/prediction/mse/{k}-step" for k in range(11)],
        ],
        "image predictions (bce) (first 10)": [
            "multiline",
            [f"loss/validation/sequence/prediction/bce/{k}-step" for k in range(11)],
        ],
        "image predictions (mse)": [
            "multiline",
            [f"loss/validation/sequence/prediction/mse/{k}-step" for k in range(50)],
        ],
        "image predictions (bce)": [
            "multiline",
            [f"loss/validation/sequence/prediction/bce/{k}-step" for k in range(50)],
        ],
    }

    layout = {
        "GP Parameters": gp_parameter_grouping,
        "Prediction Losses": prediction_losses,
    }
    writer.add_custom_scalars(layout)
