from logging import getLogger

import torch.nn as nn
import torchvision
from .wisppn_resnet import get_wisppn
import torch
from .speech2pose import Speech2pose
from .speech2pose_ad import Speech2pose_P, Speech2Pose_D
from .wipose import Wipose_LSTM
from .speech2pose_freq_atten_ver10 import Speech2pose_freq_atten_ver10

__all__ = ["get_model"]

model_names = [
    "speech2pose",
    "speech2pose_logmel",
    "wipose_lstm",
    "wisppn",
    "speech2pose_ad",
    "music2pose_freq_atten_ver10",
]
logger = getLogger(__name__)


def get_model(
    name: str, n_classes: int, input_feature: str, pretrained: bool = True, dropout:float = 0.0, seq_len:int = 12
) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "You have to choose %s as a model." % (", ").join(model_names)
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))
    out_cha = 21 * 3
    try:
        model = getattr(torchvision.models, name)(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    except:
        in_cha = 7
        if input_feature == "logmel":
            in_cha = 4
        elif input_feature == "intensity":
            in_cha = 3
        elif input_feature == 'raw':
            in_cha = 4
        elif input_feature == 'mono_channel':
            in_cha = 1
        if name == "wisppn":
            model = get_wisppn(in_cha=in_cha, out_cha=out_cha)
        elif name == 'speech2pose':
            model = Speech2pose(in_cha=in_cha, out_cha=out_cha,dropout=dropout)
        elif name == "speech2pose_logmel":
            model = Speech2pose(in_cha=6, out_cha=out_cha,dropout=dropout)
        elif name == 'wipose_lstm':
            model = Wipose_LSTM(in_cha = 6, out_cha = out_cha)
        elif name == 'speech2pose_ad':
            model = dict()
            model['model'] = Speech2pose_P(in_cha = in_cha, out_cha=out_cha)
            model['D'] = Speech2Pose_D(in_cha = 256, out_cha = 5)
        elif name == 'music2pose':
            model = Speech2pose(in_cha=9, out_cha=out_cha, dropout=dropout)
        elif name == 'music2pose_ad':
            model = dict()
            model['model'] = Speech2pose_P(in_cha = 9, out_cha=out_cha)
            model['D'] = Speech2Pose_D(in_cha = 256, out_cha = 5)
        elif name == 'music2pose_freq_atten_ver10':
            model = Speech2pose_freq_atten_ver10(in_cha=11, out_cha=out_cha, dropout=dropout)
        else:
            message = (
                "There is no model appropriate to your choice. "
                "You have to choose %s as a model." % (", ").join(model_names)
            )
            logger.error(message)
            raise ValueError(message)

    return model

