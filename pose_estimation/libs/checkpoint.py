import os
from logging import getLogger
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

logger = getLogger(__name__)


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_rmse: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_rmse": best_rmse,
    }

    torch.save(save_states, os.path.join(result_path, "checkpoint.pth"))
    logger.debug("successfully saved the ckeckpoint.")
    
def _save_average_snapshot(result_path, epoch, model, optimizer, num_snapshot=3):
    path = Path(os.path.join(result_path, "checkpoint.pth"))
    if path.exists():
        try:
            checkpoints = torch.load(str(path), map_location='cpu')['checkpoints']
        except Exception:
            checkpoints = []
    else:
        checkpoints = []

    if len(checkpoints) >= num_snapshot:
        del checkpoints[0]
    checkpoints.append({k: v.cpu() for k, v in model.state_dict().items()})

    # average checkpoints
    model_weights = checkpoints[-1].copy()
    for k, v in model_weights.items():
        model_weights[k] = v / len(checkpoints)
        for i in range(len(checkpoints)-1):
            model_weights[k] += checkpoints[i][k] / len(checkpoints)

    serialized = {
        "epoch":epoch,
        'model': model_weights,
        'checkpoints': checkpoints, 
        "optimizer": optimizer.state_dict(),
    }

    torch.save(serialized, str(path))


def resume(
    resume_path: str, model: nn.Module, optimizer: optim.Optimizer
) -> Tuple[int, nn.Module, optim.Optimizer, float]:
    try:
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        logger.info("loading checkpoint {}".format(resume_path))
    except FileNotFoundError("there is no checkpoint at the result folder.") as e:
        logger.exception(f"{e}")

    begin_epoch = checkpoint["epoch"]
    best_rmse = checkpoint["best_rmse"]
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info("training will start from {} epoch".format(begin_epoch))

    return begin_epoch, model, optimizer, best_rmse
