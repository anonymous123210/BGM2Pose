import argparse
import datetime
import os
import time
from logging import DEBUG, INFO, basicConfig, getLogger
import pandas as pd
import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from libs.checkpoint import resume, save_checkpoint, _save_average_snapshot
from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper import evaluate, train
from libs.logger import TrainLogger
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.seed import set_seed
from libs.music_setting import get_music_cv
import glob
import numpy as np

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for sound pose estimation with Sound Pose Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Add --use_wandb option if you want to use wandb.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=0,
    #     help="random seed",
    # )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )
    # configuration
    config = get_config(args.config)
    # fix seed
    set_seed(config.seed)
    if config.music_cv_ver in [3, 8, 9, 10, 11, 12]:
        cv_setting = 'cross_music'
    else:
        cv_setting = 'single_music'
    # cv_setting = 'normal'
    val_subject = pd.read_csv(f"./csv/{config.dataset_name}/val.csv")['testee'].values[0]
    # cpu or cuda
    device = get_device(allow_only_gpu=False)
    train_loader = get_dataloader(
        config.dataset_name,
        config.spec_duration,
        config.input_feature,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        seq_len=config.seq_len,
        music=get_music_cv(config.music_cv_ver), 
        aug = config.aug,
        cv_setting=cv_setting,
        val_subject=val_subject
    )

    val_loader = get_dataloader(
        config.dataset_name,
        config.spec_duration,
        config.input_feature,
        "test",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        music=get_music_cv(config.music_cv_ver),
        seq_len=config.seq_len,
        cv_setting=cv_setting,
        val_subject=val_subject
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    # define a model
    model = get_model(
        config.model,
        n_classes,
        config.input_feature,
        pretrained=config.pretrained,
        dropout=config.dropout,
        seq_len=config.seq_len
    )
    if config.model=="retrieved_aug_pose_v2":
        ckpt_path = glob.glob(f'./result/exp001/*1_3_{val_subject}_all*-music_cv_ver={config.music_cv_ver}*')[0]
        print(ckpt_path)
        ckpt_path = os.path.join(ckpt_path, 'final_model.prm')
        ckpt = torch.load(ckpt_path)
        parameter_names = [name for name, _ in model.named_parameters()]
        logger.info(set(ckpt.keys()).intersection(parameter_names))
        model.load_state_dict(ckpt, strict=False)
        new_ckpt = {}
        for key_, val_ in ckpt.items():
            if "main_block2" in key_:
                new_key = key_.replace("main_block2", "main_block2_ret")
            elif "main_block" in key_:
                new_key = key_.replace("main_block", "main_block_ret")
            elif "music_block" in key_:
                new_key = key_.replace("music_block", "music_block_ret")
            elif "multi_head_attention1" in key_:
                new_key = key_.replace("multi_head_attention1", "multi_head_attention1_ret")
            else:
                new_key = None
            if new_key:
                new_ckpt[new_key] = val_
        logger.info(set(new_ckpt.keys()).intersection(parameter_names))
        model.load_state_dict(new_ckpt, strict=False)

    # send the model to cuda/cpu
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr_max)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=config.max_epoch, eta_min=config.lr_min, last_epoch=-1)
    # keep training and validation log
    begin_epoch = 0
    best_rmse = float("inf")

    # resume if you want
    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_rmse = resume(resume_path, model, optimizer)

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLogger(log_path, resume=args.resume)

    # criterion for loss
    criterion = get_criterion(config.ratio)

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="sound_pose_estimation",
            job_type="training",
            # dirs="./wandb_result/",
        )
        # Magic
        wandb.watch(model, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        train_loss, train_rmse, train_mae, train_acc, clip_loss, logit_scale, sims = train(
            train_loader, model, criterion, optimizer, scheduler, epoch, device, aug=config.aug, smooth_loss=config.smooth_loss, logit_scale_=config.logit_scale
        )
        train_time = int(time.time() - start)
        _save_average_snapshot(result_path, epoch, model, optimizer, num_snapshot=3)
        model.load_state_dict(torch.load(os.path.join(result_path, "checkpoint.pth"))['model'])
        # validation
        start = time.time()
        # val_loss, val_acc1, val_f1s, c_matrix = evaluate(
        #     val_loader, model, criterion, device
        # )
        val_loss, val_rmse, val_mae, val_acc, _, _, _, _ = evaluate(
            val_loader, model, criterion, device
        )
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_rmse > val_rmse["all"]:
            best_rmse = val_rmse["all"]
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )

        # save checkpoint every epoch
        # save_checkpoint(result_path, epoch, model, optimizer, best_rmse)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizer.param_groups[0]["lr"],
            train_time,
            train_loss,
            clip_loss,
            logit_scale,
            train_rmse,
            train_mae,
            train_acc,
            val_time,
            val_loss,
            val_rmse,
            val_mae,
            val_acc,
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_loss": train_loss,
                    "train_clip_loss": clip_loss,
                    'train_logit_scale': logit_scale,
                    "train_rmse": train_rmse,
                    "train_mae": train_mae,
                    "train_acc": train_acc,
                    "val_time[sec]": val_time,
                    "val_loss": val_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "val_acc": val_acc,
                },
                step=epoch,
            )
        # ret_weights = np.concatenate(ret_weights)
        np.save(os.path.join(result_path, f'epoch_{epoch}_sims.npy'), sims)

    # save models
    torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

    # delete checkpoint
    # os.remove(os.path.join(result_path, "checkpoint.pth"))

    logger.info("Done")


if __name__ == "__main__":
    main()
