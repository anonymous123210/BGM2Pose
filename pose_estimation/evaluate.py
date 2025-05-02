import argparse
import csv
import datetime
import os
from logging import DEBUG, INFO, basicConfig, getLogger

import pandas as pd
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper import evaluate
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.music_setting import get_music_cv
import numpy as np

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
                    train a network for sound pose estimation
                    with Sound Pose Dataset
                    """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("mode", type=str, help="validation or test")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""path to the trained model. If you do not specify, the trained model,
            'best_acc1_model.prm' in result directory will be used.""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    parser.add_argument(
        "--output_type",
        default='both',
    ) 
    parser.add_argument(
        "--eval_name",
        type=str,
        default='csv',
    )
    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    config = get_config(args.config)
    result_path = os.path.dirname(args.config)
    image_dir = os.path.join(result_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    if args.mode not in ["validation", "test"]:
        message = "args.mode is invalid. ['validation', 'test']"
        logger.error(message)
        raise ValueError(message)

    # setting logger configuration
    logname = os.path.join(
        result_path, f"{datetime.datetime.now():%Y-%m-%d}_{args.mode}.log"
    )
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # cpu or cuda
    device = get_device(allow_only_gpu=True)
    if config.music_cv_ver in [3, 8, 9, 10, 11, 12]:
        cv_setting = 'cross_music'
    else:
        cv_setting = 'single_music'
    # cv_setting = 'normal'
    val_subject = pd.read_csv(f"./{args.eval_name}/{config.dataset_name}/val.csv")['testee'].values[0]
    loader = get_dataloader(
        config.dataset_name,
        config.spec_duration,
        config.input_feature,
        "val" if args.mode == "validation" else "test",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        music=get_music_cv(config.music_cv_ver),
        seq_len=config.seq_len,
        cv_setting=cv_setting,
        val_subject=val_subject,
        eval_name=args.eval_name,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    model = get_model(
        config.model,
        n_classes,
        config.input_feature,
        pretrained=config.pretrained,
        seq_len = config.seq_len
    )

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(os.path.join(result_path, "checkpoint.pth"))['model']

    model.load_state_dict(state_dict)

    # criterion for loss
    criterion = get_criterion(config.ratio)

    # train and validate model
    logger.info(f"---------- Start evaluation for {args.mode} data ----------")
    if args.output_type == 'none':
        image_dir = None
    # evaluation
    # loss, acc1, f1s, c_matrix = evaluate(loader, model, criterion, device)
    loss, rmse, mae, acc, gts, preds, latents, sims = evaluate(
        loader, model, criterion, device, mode=args.mode, image_dir=image_dir, output_type=args.output_type
    )

    # logger.info("loss: {:.5f}\tacc1: {:.2f}\tF1 Score: {:.2f}".format(loss, acc1, f1s))
    logger.info(
        "loss: {:.5f}\tRMSE: {:.2f}\tMAE: {:.2f}\tAcc: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            loss,
            rmse["all"],
            mae["all"],
            acc["all"][0],
            acc["all"][1],
            acc["all"][2],
            acc["all"][3],
        )
    )
    logger.info(
        "arm RMSE: {:.2f}\tarm MAE: {:.2f}\tAcc: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            rmse["arm"],
            mae["arm"],
            acc["arm"][0],
            acc["arm"][1],
            acc["arm"][2],
            acc["arm"][3],
        )
    )
    logger.info(
        "leg RMSE: {:.2f}\tleg MAE: {:.2f}\tAcc: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            rmse["leg"],
            mae["leg"],
            acc["leg"][0],
            acc["leg"][1],
            acc["leg"][2],
            acc["leg"][3],
        )
    )
    logger.info(
        "body RMSE: {:.2f}\tbody MAE: {:.2f}\tAcc: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            rmse["body"],
            mae["body"],
            acc["body"][0],
            acc["body"][1],
            acc["body"][2],
            acc["body"][3],
        )
    )
    df = pd.DataFrame(
        {"loss": [loss], "rmse": [rmse], "mae": [mae], "acc": [acc]},
        columns=["loss", "rmse", "mae", "acc"],
        index=None,
    )

    df.to_csv(os.path.join(result_path, f"test_log_{args.eval_name}.csv"), index=False)
    np.save(os.path.join(result_path, f'pose_gts_{args.eval_name}.npy'), gts)
    np.save(os.path.join(result_path, f'pose_preds_{args.eval_name}.npy'), preds)
    np.save(os.path.join(result_path, f'latents_{args.eval_name}.npy'), latents)
    logger.info("Done.")


if __name__ == "__main__":
    main()
