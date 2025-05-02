from logging import getLogger
from typing import Optional

import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
# from ..dataset_csv import DATASET_CSVS
# from .class_weight import get_class_weight

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    adver_ratio:float = 0
) -> nn.Module:
    if adver_ratio > 0:
        criterion = dict()
        criterion['model'] = nn.MSELoss()
        criterion['D'] = nn.CrossEntropyLoss()
        criterion['ratio'] = adver_ratio
    else:
        criterion = nn.MSELoss()
    return criterion

def compute_binary_cross_entropy_matrix(logits, targets, bgm_weights):
    """
    バイナリクロスエントロピー損失を各ラベルに適用し、[batch_size, label_size] の行列を返します。

    Args:
        logits: テンソル [batch_size, label_size]、モデルの出力（未正規化のスコア）
        targets: テンソル [batch_size, label_size]、バイナリターゲットラベル（0または1）
        label_smoothing: float、ラベルスムージングの係数（デフォルトは0.0）

    Returns:
        loss_matrix: テンソル [batch_size, label_size]、各サンプル・各ラベルごとの損失値
    """

    # バイナリクロスエントロピー損失の計算
    num_classes = logits.shape[0]

    # ワンホットエンコーディングの作成
    one_hot_labels = F.one_hot(targets, num_classes=num_classes).float()
    loss_matrix = F.binary_cross_entropy_with_logits(logits, one_hot_labels, reduction='none')
    loss_matrix[bgm_weights==1] *= 5
    return loss_matrix



class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            logit_scale=0.07
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.logit_scale = logit_scale

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, music_types, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, np.log(1 / self.logit_scale))
        total_loss = 0
        # for music_type in music_types_unique:
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss
