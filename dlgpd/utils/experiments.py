""" Save/load models/experiments """
import logging
import os

import torch

logger = logging.getLogger(__name__)


def save_states(pytorch_object, path, _run=None):
    torch.save(pytorch_object.state_dict(), path)
    logger.debug("Saved {} to {}".format(pytorch_object.__class__.__name__, path))
    if _run:
        _run.add_artifact(path)
        logger.debug("Registered {} as an artifact".format(path))


def save_experiment(ex_objects, model_path, epoch=None, _run=None):
    for name, pobject in ex_objects.items():
        if epoch:
            name = f"epoch{epoch}_" + name
        save_states(pobject, os.path.join(model_path, name + ".pt"), _run)
