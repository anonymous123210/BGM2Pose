from logging import getLogger
from typing import List
import glob
import numpy as np
logger = getLogger(__name__)

def get_mean() -> List[float]:
    # in a classroom
    mean = [
        -23.539913, 
        -25.119139, 
        -26.711174, 
        -22.011652, 
        0.0044820304, 
        0.00080580305, 
        -0.040803947
        ]
    

    logger.info(f"mean value: {mean}")
    return mean


def get_std() -> List[float]:
    # in a classroom
    std = [
        12.967929, 
        10.571223, 
        10.273731, 
        11.8457985, 
        0.022749536, 
        0.019968906, 
        0.037085325
        ]

    logger.info(f"std value: {std}")
    return std

def get_mean_music(music_type:str='jazz') -> List[float]:
    if music_type == 'jazz':
        mean = [
            -16.957348, 
            -18.000713, 
            -20.53041, 
            -15.92767, 
            -19.950762, 
            -20.871494, 
            0.0006076998, 
            0.0009762824, 
            -0.013828985
        ]
    elif music_type == 'cirrus':
        mean = [-21.437536, -21.861847, -24.83102, -20.684359, -21.42624, -22.515173, 0.0005496154, 0.00031469387, -0.01345951]
    elif music_type == 'arnor':
        mean = [-22.34753, -22.290886, -25.803625, -21.670645, -31.663267, -30.171076, -7.713504e-05, 0.000403049, -0.011210225]
    elif music_type == 'mantron':
        mean = [-24.252245, -24.13487, -27.592958, -23.681503, -35.573185, -36.0816, 0.00010231032, 0.00018740784, -0.010841111]
        
    # logger.info(f"mean value: {mean}")
    return mean

def get_std_music(music_type:str='jazz') -> List[float]:
    if music_type == 'jazz':
        std = [
                20.589516, 
                17.214449, 
                15.242527, 
                18.578787, 
                28.552048, 
                28.646671, 
                0.015363253, 
                0.011566004, 
                0.018870942
            ]
    elif music_type == 'cirrus':
        std = [21.075201, 18.122711, 15.19214, 18.57458, 26.82804, 26.986845, 0.016524797, 0.011419602, 0.017496226]
    elif music_type == 'arnor':
        std = [21.993052, 19.654186, 16.043749, 19.641361, 32.62192, 31.799248, 0.017382737, 0.011129919, 0.017447662]
    elif music_type == 'mantron':
        std = [22.483595, 20.395716, 16.392046, 19.99123, 35.62522, 35.553818, 0.017658168, 0.011187353, 0.017098213]
    else:
        assert False
        
    # logger.info(f"std value: {std}")
    return std

def get_raw_mean() -> List[float]:
    mean = [
        -1.99718730e-05, -1.60138708e-08, -5.86972669e-08, -1.24919584e-07
    ]

    logger.info(f"mean value: {mean}")
    return mean


def get_raw_std() -> List[float]:
    std = [
        0.00657787, 0.00519175, 0.00472543, 0.00824625
    ]

    logger.info(f"std value: {std}")
    return std

def get_logmel_mean() -> List[float]:
    mean = [
        -23.539913, 
        -25.119139, 
        -26.711174, 
        -22.011652, 
        ]

    logger.info(f"mean value: {mean}")
    return mean


def get_logmel_std() -> List[float]:
    std = [
        12.967929, 
        10.571223, 
        10.273731, 
        11.8457985, 
        ]
    return std

def get_wo_norm_mean() -> List[float]:
    mean = [
        -16.566772,
        -13.443462,
        -18.72402,
        -15.498938,
        -17.582392,
        -17.18739,
        -15.774
    ]
    return mean

def get_wo_norm_std() -> List[float]:
    std = [
        15.025663, 
        12.787112, 
        17.318056, 
        14.863319, 
        16.7786, 
        15.965399, 
        14.254401
    ]
    return std

def get_pose_mean_std():
    data = []
    files = glob.glob("dataset_spec/music_with_intensity/*joint*")
    for file in files:
        data.append(np.load(file))
    data = np.array(data)
    mean_ = data.reshape(-1, 63).mean(axis=0)
    std_ = data.reshape(-1, 63).std(axis=0)
    return mean_, std_

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import tqdm

    csv_path = "./csv/kfold_same_music_wo_subject_1_3_subject_6_all/train.csv"
    csv = pd.read_csv(csv_path)
    csv = csv[csv["sound_length"] == 2400]
    csv = csv[csv['preprocess'] == 'music_intensity'].reset_index()
    csv = csv[csv['music_type']=='cirrus']
    data = [np.load(path) for path in  tqdm.tqdm(csv["sound_path"])]
    data = np.array(data) #[sample_num, freq_bin, channels]
    print("mean: [", end="")
    for i in range(data.shape[-1]):
        if i:
            print(", ", end="")
        print(np.mean(data[:, :, i]), end="")
    print("]")
    print("std: [", end="")
    for i in range(data.shape[-1]):
        if i:
            print(", ", end="")
        print(np.std(data[:, :, i]), end="")
    print("]")
