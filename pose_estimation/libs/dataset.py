from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset_csv import DATASET_CSVS
from .mean_std import get_mean, get_std, get_raw_mean, get_raw_std, get_mean_music, get_std_music, get_pose_mean_std
from .sound_length_list import get_sound_length_list, get_sound_duration_list
from .joint_list import joint2list, get_joints
from .data_aug import filt_aug, frame_shift
from torch.utils.data.sampler import BatchSampler

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(
    dataset_name: str,
    spec_duration: float,
    input_feature: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None,
    music = None,
    seq_len=12,
    aug:str = '',
    cv_setting:str = 'same_music',
    val_subject: str = 'subject_0',
    eval_name:str = 'csv'
) -> DataLoader:
    if split not in ["train", "valid", "test"]:
        message = "split should be selected from ['train', 'valid', 'test']."
        logger.error(message)
        raise ValueError(message)

    logger.info(f"Dataset: {dataset_name}\tSplit: {split}\tBatch size: {batch_size}.")
    try:
        csv_file = getattr(DATASET_CSVS[dataset_name], split)
    except:
        csv_file = f"{eval_name}/{dataset_name}/{split}.csv"
    
    is_train = split=='train'
    
    if input_feature == 'raw':
        data = SoundPoseLSTMDataset(
            csv_file, spec_duration, input_feature, mean=np.array(get_raw_mean()).astype("float32"), std=np.array(get_raw_std()).astype("float32"), transform=transform, music=music, seq_len=seq_len, aug=aug
        )
    elif input_feature == 'music_intensity':
        data = SoundPose2DDataset(
            csv_file, spec_duration, input_feature, transform=transform, music=music[split], seq_len=seq_len, aug = aug, cv_setting=cv_setting, val_subject=val_subject, is_train=is_train, test_music = music['test'][0], 
        )
    else:
        data = SoundPose2DDataset(
            csv_file, spec_duration, input_feature, transform=transform, music = music[split], seq_len=seq_len, is_train=is_train, test_music = music['test'][0], 
        )
    if split == 'train':
        bgmsampler = BGMBatchSampler(data, batch_size=batch_size)
        dataloader = DataLoader(
            data,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_sampler=bgmsampler,
        )
    else:
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    return dataloader


class SoundPoseDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        sound_length: int,
        input_feature: str,
        mean: np.array,
        std: np.array,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        sound_length_list = get_sound_length_list()

        if sound_length not in sound_length_list:
            message = (
                "There is no sound length appropriate to your choice. "
                "You have to choose %s as sound length." % sound_length_list
            )
            logger.error(message)
            raise ValueError(message)
        self.input_feature = input_feature
        if self.input_feature != 'logmel':
            self.df = self.df[self.df['preprocess'] == self.input_feature]
            self.df = self.df.reset_index(drop=True)
        self.df = (
            self.df[self.df["sound_length"] == sound_length]
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        self.sound = np.array(
            [(np.load(path) - mean) / std for path in self.df["sound_path"]]
        )
        self.sound = self.sound.transpose(0, 2, 1)
        if input_feature == "logmel":
            self.sound = self.sound[:, :4]
        elif input_feature == "intensity":
            self.sound = self.sound[:, 4:]

        self.transform = transform

        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sound = self.sound[idx]

        if self.transform is not None:
            sound = self.transform(sound)

        targets = torch.Tensor(joint2list(self.df.iloc[idx]))
        testee = self.df.iloc[idx * self.seq_len]['testee'].split('_')[-1]
        testee_num = int(testee) - 1
        sample = {
            "sound": sound,
            "targets": targets,
            "testee":testee_num,
        }

        return sample
    
class SoundPoseLSTMDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        sound_length: int,
        input_feature: str,
        mean: np.array,
        std: np.array,
        transform: Optional[transforms.Compose] = None,
        is_train = False,
        aug:str = '',
        seq_len:int =12,
    ) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        sound_length_list = get_sound_length_list()
        self.seq_len = seq_len
        if sound_length not in sound_length_list:
            message = (
                "There is no sound length appropriate to your choice. "
                "You have to choose %s as sound length." % sound_length_list
            )
            logger.error(message)
            raise ValueError(message)
        self.input_feature = input_feature
        if (self.input_feature != 'logmel') & (self.input_feature != 'all')& (self.input_feature != 'intensity'):
            self.df = self.df[self.df['preprocess'] == self.input_feature]
            self.df = self.df.reset_index(drop=True)
        else:
            # Logmel特徴量はintensityとともに作られる。特徴量選択は self.sound = self.sound[:, :, :4]によって実施
            self.df = self.df[self.df['preprocess'] == 'intensity']
            self.df = self.df.reset_index(drop=True)
        self.df = (
            self.df[self.df["sound_length"] == sound_length]
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        self.df = self.df.iloc[:self.seq_len * (self.df.shape[0]//self.seq_len)]
        self.sound = np.array(
            [(np.load(path) - mean) / std for path in self.df["sound_path"]]
        )
        self.sound = self.sound.transpose(0, 2, 1) #[sample_size, channels, sound_length]
        self.sound = self.sound.reshape(self.df.shape[0]//self.seq_len, self.seq_len, self.sound.shape[1], self.sound.shape[-1])
        self.targets = self.df[get_joints()].values.reshape(self.df.shape[0]//self.seq_len, self.seq_len, len(get_joints()))
        assert self.sound.shape[0] == self.targets.shape[0]
        if input_feature == "logmel":
            self.sound = self.sound[:, :, :4]
        elif input_feature == "intensity":
            self.sound = self.sound[:, :, 4:]
        self.input_feature = input_feature
        self.transform = transform
        self.is_train = is_train
        self.aug = aug
        if is_train:
            self.max_noise_amp = 0.2
        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.sound)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sound = torch.Tensor(self.sound[idx]).float() #[seq_len, channels, 1024]
        if self.transform is not None:
            sound = self.transform(sound)
        targets = torch.Tensor(self.targets[idx]) #[seq_len, num_joints * 3]
        testee = self.df.iloc[idx * self.seq_len]['testee'].split('_')[-1]
        testee_num = int(testee) - 1
        sample = {
            "sound": sound,
            "targets": targets,
            "testee":testee_num
        }
        return sample
    
class SoundPose2DDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        spec_duration: float,
        input_feature: str,
        transform: Optional[transforms.Compose] = None,
        is_train = False,
        aug:str = '',
        music=None,
        seq_len:int = 12,
        cv_setting:str = 'same_music',
        val_subject: str = 'subject_0',
        test_music:str=None
    ) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        spec_duration_list = get_sound_duration_list()
        self.seq_len = seq_len
        self.test_music = test_music
        if spec_duration not in spec_duration_list:
            message = (
                "There is no sound length appropriate to your choice. "
                "You have to choose %s as sound length." % spec_duration_list
            )
            logger.error(message)
            raise ValueError(message)
        self.input_feature = input_feature
        if (self.input_feature == 'mono_channel'):
            self.df = self.df[self.df['preprocess'] == 'music_intensity'].reset_index(drop=True)
        elif (self.input_feature != 'logmel') & (self.input_feature != 'all')& (self.input_feature != 'intensity'):
            self.df = self.df[self.df['preprocess'] == self.input_feature]
            self.df = self.df.reset_index(drop=True)
        else:
            # Logmel特徴量はintensityとして作られる。特徴量選択は self.sound = self.sound[:, :, :4]によって実施
            self.df = self.df[self.df['preprocess'] == 'intensity']
            self.df = self.df.reset_index(drop=True)
        if music!=None:
            self.df = self.df[self.df.music_type.isin(music)]
        self.df = (
            self.df[self.df["spec_duration"] == spec_duration]
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        # ToDo: logmelとallで場合分け
        if input_feature == 'logmel':
            self.sound = np.array(
                [(np.load(path)[:-9, :, :56]) for path in self.df["sound_path"]]
            )
        else:
            self.sound = np.array(
                [(np.load(path)[:-6, :, :56]) for path in self.df["sound_path"]]
            )
        self.sound_path = [path for path in self.df["sound_path"]]
        self.sound = self.sound.transpose(0, 1, 3, 2)
        if 'wo_mocap' in csv_file:
            self.targets = np.zeros((self.df.shape[0], 12, 63))
        else:
            # self.targets = self.df[get_joints()].values.reshape(self.df.shape[0]//self.seq_len, self.seq_len, len(get_joints()))
            self.targets = np.array(
                [(np.load(path)[:56,:]) if path is not np.nan else np.zeros(12, 63) for path in self.df["joint_path"]]
            )
        assert self.sound.shape[0] == self.targets.shape[0]
        self.input_feature = input_feature
        self.transform = transform
        self.is_train = is_train
        logger.info(f"the number of samples: {len(self.df)}")
        self.cv_setting = cv_setting
        self.val_subject = val_subject
        self.pose_mean, self.pose_std = get_pose_mean_std()
    def __len__(self) -> int:
        return len(self.sound)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sound = self.sound[idx]
        sound = torch.Tensor(sound).float() #[in_cha, seq_len, freq_bin]
        sound_path = self.sound_path[idx]
        sound_path = '_'.join(sound_path.split("/")[-1].split('_')[2:4])
        testee = self.df.iloc[idx]['testee'].split('_')[-1]
        if self.transform is not None:
            sound = self.transform(sound)
        targets = torch.Tensor(self.targets[idx]).to(sound.device) #[seq_len, num_joints * 3]
        testee_num = int(testee) #- 1
        music_type = self.df.iloc[idx]['music_type']
        mean_ = torch.Tensor(get_mean_music(music_type=music_type)).to(sound.device)
        std_ = torch.Tensor(get_std_music(music_type=music_type)).to(sound.device)
        sound = (sound - mean_.reshape(-1, 1, 1)) / std_.reshape(-1, 1, 1)
        if testee_num > 4:
            testee_num -= 5
        sample = {
            "sound": sound,
            "targets": targets,
            'testee':testee_num,
            "sound_path":sound_path,
        }
        return sample
    
class BGMBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, batch_size, eval_type='single_music'):
        loader = DataLoader(dataset)
        self.sensing_bgms = []
        self.original_bgms = []
        # for sample in loader:
        #     print(sample['sound_path'])
        #     self.original_bgms.append(sample['sound_path'])
        self.original_bgms = list(dataset.sound_path)
        # /home/shibatie/SSD/Music2Pose/pose_estimation_with_noise/dataset_spec_2400/music_with_intensity/subject_0_mantron_13_0.6_music_intensity.npy
        self.original_bgms = ["_".join(sound_path.split('/')[-1].split('_')[2:4]) for sound_path in self.original_bgms]
        self.original_bgms = np.array(self.original_bgms)
        self.sensing_bgms = np.unique(self.original_bgms)
        self.sensing_bgm_to_indices = {bgm: np.where(self.original_bgms == bgm)[0]
                                 for bgm in self.sensing_bgms}
        # for l in self.sensing_bgms:
        #     np.random.shuffle(self.sensing_bgm_to_indices[l])
        self.sensing_bgm_set = list(set(self.sensing_bgm_to_indices.keys()))
        print(self.sensing_bgm_set[0], self.sensing_bgm_to_indices[self.sensing_bgm_set[0]])
        self.count = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.eval_type = eval_type

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            tmp_count = 0
            indices = []
            while tmp_count < self.batch_size:
                tmp_bgm = np.random.choice(self.sensing_bgm_set)
                tmp_indices = self.sensing_bgm_to_indices[tmp_bgm]
                tmp_size = min(self.batch_size - tmp_count, len(tmp_indices))
                np.random.shuffle(tmp_indices)
                indices.extend(tmp_indices[:tmp_size])
                tmp_count += tmp_size
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size
