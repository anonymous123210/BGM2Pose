import os, time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

# from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter
from .metric2 import calc_accuracy, calc_rmse_mae_acc
from .loss_fn import mixup, ClipLoss
from .joint_list import list2joint, joint2list

from scipy.ndimage import gaussian_filter1d
def fast_gaussian_filter(data, sigma, axis=0):
    return gaussian_filter1d(data, sigma=sigma, axis=axis, mode='reflect')

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)


def do_one_iteration(
    sample: Dict[str, Any],
    model: nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    optimizer: Optional[optim.Optimizer] = None,
    do_mixup: bool = False,
    smooth_loss: bool = False,
    logit_scale_ = 0.07,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and optimizer is None:
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    x = sample["sound"].to(device)
    t = sample["targets"].to(device)
    music_type_map = {'arnor':0, 'mantron':1, 'cirrus':2, 'jazz':3}
    batch_size = x.shape[0]
    output, sound_emb, pose_emb, logit_scale, latent = model(x, t)
    if smooth_loss:
        t_diff = t[:,:-1,:] - t[:,1:,:]
        output_diff = output[:,:-1,:] - output[:,1:,:]
        loss = criterion(output, t) + criterion(t_diff, output_diff) * 100
    else:
        loss = criterion(output, t) 
    clip_loss = ClipLoss(logit_scale=logit_scale_)(sound_emb, pose_emb, sample['sound_path'], logit_scale_)
    sim = logit_scale * sound_emb @ pose_emb.T
    sim = torch.nn.Softmax(dim=-1)(sim)
    sim = sim.detach().cpu().numpy()
    loss += clip_loss
    # measure accuracy and record loss
    # accs = calc_accuracy(output, t, topk=(1,))
    # acc1 = accs[0]

    if iter_type == "train" and optimizer is not None:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # keep predicted results and gts for calculate F1 Score
    gt = t.to("cpu").numpy()
    pred = output.to("cpu").detach().numpy()
    latent = latent.to('cpu').detach().numpy()
    # ret_weight = ret_weight.to("cpu").detach().numpy()
    
    gt = gt.reshape(-1, gt.shape[-1])
    pred = pred.reshape(-1, pred.shape[-1])

    # return batch_size, loss.item(), acc1, gt, pred
    return batch_size, loss.item(), gt, pred, clip_loss.item(), logit_scale.item(), latent, sim


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    scheduler: Any, 
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
    aug:str = '',
    smooth_loss:bool = False,
    logit_scale_ = 0.07,
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    logit_scales = AverageMeter('Logit scale', ":.4e")
    clip_losses = AverageMeter("CLIP Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []
    # ret_weights = []
    sims = []
    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size, loss, gt, pred, clip_loss, logit_scale, latent, sim  = do_one_iteration(
            sample, model, criterion, device, "train", optimizer, do_mixup=False, smooth_loss=smooth_loss, logit_scale_=logit_scale_
        )
        losses.update(loss, batch_size)
        clip_losses.update(clip_loss, batch_size)
        logit_scales.update(logit_scale, batch_size)
        # top1.update(acc1, batch_size)

        # save the ground truths and predictions in lists
        gts += list(gt)
        preds += list(pred)
        if sim.shape[0] == 64:
            sims.append(sim)
        # ret_weights += list(ret_weight)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)
            
    if isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    # calculate F1 Score
    # f1s = f1_score(gts, preds, average="macro")
    rmse, mae, acc = calc_rmse_mae_acc(gts, preds)
    sims = np.array(sims)

    # return losses.get_average(), top1.get_average(), f1s
    return losses.get_average(), rmse, mae, acc, clip_losses.get_average(), logit_scales.get_average(), sims #ret_weights


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    device: str,
    mode: str = "train",
    image_dir: str = None,
    output_type:str = 'both',
) -> Tuple[float, float, float, np.ndarray]:
    losses = AverageMeter("Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []
    latents = []
    sims = []

    # calculate confusion matrix
    # n_classes = loader.dataset.get_n_classes()
    # c_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in loader:
            # batch_size, loss, acc1, gt, pred = do_one_iteration(
            #     sample, model, criterion, device, "evaluate", do_mixup=False
            # )
            batch_size, loss, gt, pred, _, _, latent, sim = do_one_iteration(
                sample, model, criterion, device, "evaluate", do_mixup=False
            )

            losses.update(loss, batch_size)
            # top1.update(acc1, batch_size)

            # keep predicted results and gts for calculate F1 Score
            gts += list(gt)
            preds += list(pred)
            latents += list(latent)
            if sim.shape[0] == 64:
                sims.append(sim)

            # c_matrix += confusion_matrix(
            #     gt,
            #     pred,
            #     labels=[i for i in range(n_classes)],
            # )

    # f1s = f1_score(gts, preds, average="macro")

    rmse, mae, acc = calc_rmse_mae_acc(gts, preds, mode=mode, image_dir=image_dir, output_type=output_type)
    # smoothed_preds = fast_gaussian_filter(np.array(preds), sigma=21)
    # rmse, mae, acc = calc_rmse_mae_acc(gts, smoothed_preds.tolist(), mode=mode, image_dir=image_dir, output_type=output_type)
    gts = np.array(gts)
    preds = np.array(preds)
    latents = np.array(latents)
    sims = np.array(sims)
    
    
    # return losses.get_average(), top1.get_average(), f1s, c_matrix
    return losses.get_average(), rmse, mae, acc, gts, preds, latents, sims
