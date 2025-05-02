from typing import Any, Dict, List, Tuple

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error

from libs.joint_list import (
    get_joint_names,
    get_leg,
    get_arm,
    get_body,
    joint2list,
    list2joint,
)


def calc_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[float]:
    """Computes the accuracy over the k top predictions.

    Args:
        output: (N, C). model output.
        target: (N, C). ground truth.
        topk: if you set (1, 5), top 1 and top 5 accuracy are calcuated.
    Return:
        res: List of calculated top k accuracy
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1)
            correct_k = correct_k.float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def get_visualize_config():
    joint_config = {}
    # 関節点のマーカー形状、サイズを決定
    for name in get_joint_names():
        joint_config[name] = {"marker":"o","marker_size":25}
    joint_config["Head"]["marker_size"] = 150
    # joint_config["Neck"]["marker_size"] = 0

    # 関節点間の線の太さを決定
    joint_config["line_width"] = 2

    # 人体の中央、右、左のそれぞれに色を設定(R,G,B)
    # 色は以下のサイトから探しました
    # https://www.color-site.com/types/red
    # https://www.color-site.com/types/blue

    color_config = {
        "gt":{"Center":[1,0,102/256],"Left":[240/256,86/256,110/256],"Right":[1,0,0]},
        # "gt":{"Center":[255/256,200/256,0],"Left":[255/256,200/256,0],"Right":[255/256,200/256,0]}, #黄色
        # "gt":{"Center":[127/256, 17/256,132/256],"Left":[127/256, 17/256,132/256],"Right":[127/256, 17/256,132/256]}, #紫
        # "gt":{"Center":[64/256, 186/256,141/256],"Left":[64/256, 186/256,141/256],"Right":[64/256, 186/256,141/256]}, #緑
        "pred":{"Center":[0,103/256,1],"Left":[98/256,131/256,194/256],"Right":[0,0,1]},
    }
    for name in get_joint_names():
        if "Right" in name:
            key = "Right"
        elif "Left" in name:
            key = "Left"
        else:
            key = "Center"
        color_config["gt"][name] = color_config["gt"][key]
        color_config["pred"][name] = color_config["pred"][key]
    
    return joint_config, color_config

def make_pose_image(
    idx: int,
    joints: Dict[str, Any],
    sub_joints: Dict[str, Any] = None,
    output_type = 'both',
) -> None:

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("white") # これはいらないかも１
    # ax.set_facecolor("white")
    ax.patch.set_facecolor("white") # これはいらないかも2
    # 以下で背景を白にする
    white = (1, 1, 1, 0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    # 以下を有効にするとgridが消える
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # 以下を有効にすると3次元平面が完全に消える
    # ax.set_axis_off()

    lim = 10
    # lim = 5
    # lim = 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 2 * lim)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    # joints = list2joint(joints)
    joint_config,color_config = get_visualize_config() 

    # 関節点をプロットするかどうかを決める。bothで表示するときはFalseにしたほうが良さそう
    plot_joint_point = True
    # plot_joint_point = False

    if plot_joint_point & ((output_type == 'both') | (output_type=='gt')):
        for name in get_joint_names():
            ax.scatter(
                joints[name + "_x"],
                -joints[name + "_z"],
                joints[name + "_y"],
                color=color_config["gt"][name],
                marker=joint_config[name]["marker"],
                s=joint_config[name]["marker_size"],
            )


    for lr in ["Left", "Right"]:
        for parts in [get_leg(lr), get_arm(lr)]:
            points = {
                "_x": [],
                "_y": [],
                "_z": [],
            }
            for name in parts:
                for dim in ["_x", "_y", "_z"]:
                    points[dim].append(joints[name + dim])
            if (output_type == 'both') | (output_type=='gt'):
                # ax.plot(points["_x"], -np.array(points["_z"]), points["_y"], color="red")
                ax.plot(points["_x"], -np.array(points["_z"]), points["_y"], color=color_config["gt"][lr], lw=joint_config["line_width"])
    points = {
        "_x": [],
        "_y": [],
        "_z": [],
    }
    for name in get_body():
        for dim in ["_x", "_y", "_z"]:
            points[dim].append(joints[name + dim])
    if (output_type == 'both') | (output_type == 'gt'):
        ax.plot(
            points["_x"],
            -np.array(points["_z"]),
            # np.array(points["_z"]),
            points["_y"],
            # color="red",
            color=color_config["gt"]["Center"],
            label="Ground Truth",
            lw=joint_config["line_width"],
        )
    if not sub_joints is None:
        if plot_joint_point & ((output_type == 'both') | (output_type == 'prediction')):
            for name in get_joint_names():
                if "Right" in name:
                    color = color_config["pred"]["Right"]
                elif "Left" in name:
                    color = color_config["pred"]["Left"]
                else:
                    color = color_config["pred"]["Center"]

                ax.scatter(
                    sub_joints[name + "_x"],
                    -sub_joints[name + "_z"],
                    sub_joints[name + "_y"],
                    color=color,
                    marker=joint_config[name]["marker"],
                    s=joint_config[name]["marker_size"],
                )

        for lr in ["Left", "Right"]:
            for parts in [get_leg(lr), get_arm(lr)]:
                points = {
                    "_x": [],
                    "_y": [],
                    "_z": [],
                }
                for name in parts:
                    for dim in ["_x", "_y", "_z"]:
                        points[dim].append(sub_joints[name + dim])
                if (output_type == 'both') | (output_type == 'prediction'):
                    # ax.plot(points["_x"], -np.array(points["_z"]), points["_y"], color="blue")
                    ax.plot(points["_x"], -np.array(points["_z"]), points["_y"], color=color_config["pred"][lr], lw=joint_config["line_width"])
        points = {
            "_x": [],
            "_y": [],
            "_z": [],
        }
        for name in get_body():
            for dim in ["_x", "_y", "_z"]:
                points[dim].append(sub_joints[name + dim])
        if (output_type == 'both') | (output_type == 'prediction'):
            ax.plot(
                points["_x"],
                -np.array(points["_z"]),
                points["_y"],
                # color="blue",
                color=color_config["pred"]["Center"],
                label="Predict",
                lw = joint_config["line_width"],
            )
        if output_type == 'both':
            plt.legend(fontsize=9)

    fig.canvas.draw()
    data = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    c = len(data) // (w * h)
    plt.close()

    img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return {"idx": idx, "img": img}


def make_time_score(
    gt_lists: Dict[str, List[Any]],
    pred_lists: Dict[str, List[Any]],
    mode: str,
    image_dir: str,
) -> None:

    for part in ["all", "arm", "leg", "body"]:

        rmse = []
        mae = []

        for gt, pred in zip(gt_lists[part], pred_lists[part]):
            rmse.append(np.sqrt(mean_squared_error(gt, pred)))
            mae.append(mean_absolute_error(gt, pred))

        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.scatter(range(len(rmse)), rmse)
        ax1.set_ylabel("rmse (cm)")
        ax1.set_ylim(0, 5)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.scatter(range(len(mae)), mae)
        ax2.set_xlabel("time (-)")
        ax2.set_ylabel("mae (cm)")
        ax2.set_ylim(0, 5)

        path = os.path.join(image_dir, "%s_%s.png" % (mode, part))
        fig.savefig(path)
        plt.close()


def calc_dis(
    gt_dict: Dict[str, float], pred_dict: Dict[str, float], joint: str
) -> float:
    dis = 0.0
    for dim in ["_x", "_y", "_z"]:
        dis += (gt_dict[joint + dim] - pred_dict[joint + dim]) ** 2
    return np.sqrt(dis)


def calc_acc(corrects: Dict[str, List[float]], part: str) -> float:

    acc = []

    for i in range(4):
        if part == "all":
            ac = np.mean(
                np.array(
                    [np.mean(corrects[joint][:, i]) for joint in get_joint_names()]
                )
            )
        elif part == "arm":
            ac = np.mean(
                np.array(
                    [
                        np.mean(corrects[joint][:, i])
                        for joint in get_arm("Right") + get_arm("Left")
                    ]
                )
            )
        elif part == "leg":
            ac = np.mean(
                np.array(
                    [
                        np.mean(corrects[joint][:, i])
                        for joint in get_leg("Right") + get_leg("Left")
                    ]
                )
            )
        elif part == "body":
            ac = np.mean(
                np.array([np.mean(corrects[joint][:, i]) for joint in get_body()])
            )

        acc.append(ac)

    return acc


def gt_pred_process(idx, gt, pred):
    gt_dict = list2joint(gt)
    pred_dict = list2joint(pred)

    out = {
        "idx": idx,
        "gt_dicts": gt_dict,
        "pred_dicts": pred_dict,
        "corrects": {},
        "gt_lists": {},
        "pred_lists": {},
    }

    threshold = 0.0
    for dim in ["_x", "_y", "_z"]:
        threshold += (gt_dict["Neck" + dim] - gt_dict["Head" + dim]) ** 2
    threshold = np.sqrt(threshold) * 0.5
    for joint in get_joint_names():
        out["corrects"][joint] = []
        for i in range(1, 5):
            cor = float(calc_dis(gt_dict, pred_dict, joint) < (threshold * i))
            out["corrects"][joint].append(cor)

    for part in ["all", "arm", "leg", "body"]:
        gt_list = joint2list(gt_dict, part)
        pred_list = joint2list(pred_dict, part)
        out["gt_lists"][part] = gt_list
        out["pred_lists"][part] = pred_list

    return out


def calc_rmse_mae_acc(gts, preds, mode=None, image_dir: str = None, output_type:str = 'both'):
    lists = Parallel(n_jobs=5)(
        [delayed(gt_pred_process)(i, gts[i], preds[i]) for i in range(len(gts))]
    )
    lists = sorted(lists, key=lambda x: x["idx"])
    # lists = [gt_pred_process(i, gts[i], preds[i]) for i in range(len(gts))]
    corrects = {
        name: [data["corrects"][name] for data in lists] for name in get_joint_names()
    }
    corrects = {k: np.array(v) for k, v in corrects.items()}
    gt_lists = {}
    pred_lists = {}
    for part in ["all", "arm", "leg", "body"]:
        gt_lists[part] = [data["gt_lists"][part] for data in lists]
        pred_lists[part] = [data["pred_lists"][part] for data in lists]

    if not image_dir is None:
        fps = 20
        w = 500
        h = 500
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            os.path.join(image_dir, mode + "_" + output_type + ".mp4"), codec, fps, (w, h)
        )
        # output_type = 'prediction'
        imgs = Parallel(n_jobs=10)(
            [
                delayed(make_pose_image)(
                    lists[i]["idx"], lists[i]["gt_dicts"], lists[i]["pred_dicts"], output_type = output_type
                )
                for i in range(len(lists)//4)
            ]
        )
        # imgs = Parallel(n_jobs=16)(
        #     [
        #         delayed(make_pose_image)(
        #             data["idx"], data["gt_dicts"], data["pred_dicts"], output_type = output_type
        #         )
        #         for data in lists
        #     ]
        # )
        imgs = sorted(imgs, key=lambda x: x["idx"])

        for img in imgs:
            video.write(img["img"])
        video.release()

    if not image_dir is None:
        make_time_score(gt_lists, pred_lists, mode, image_dir)

    rmse = {}
    mae = {}
    acc = {}

    for part in ["all", "arm", "leg", "body"]:
        rmse[part] = np.sqrt(
            mean_squared_error(
                np.concatenate(gt_lists[part]), np.concatenate(pred_lists[part])
            )
        )
        mae[part] = mean_absolute_error(
            np.concatenate(gt_lists[part]), np.concatenate(pred_lists[part])
        )
        acc[part] = calc_acc(corrects, part)

    return rmse, mae, acc
