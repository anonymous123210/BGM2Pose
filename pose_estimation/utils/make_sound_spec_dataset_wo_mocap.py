from logging import getLogger

import argparse
import glob
import os
import sys
from typing import Dict, List, Union
import tqdm

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import warnings
warnings.simplefilter('ignore')

logger = getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from libs.sound_length_list import get_sound_length_list
from libs.joint_list import get_joints, get_joint_names


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make sound dataset for sound pose estimation"
    )
    parser.add_argument(
        "--sound_dir",
        type=str,
        default="./data_0724/sound_cut/",
        help="path to a sound dirctory",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="./data_0724/annotations/",
        help="a path where annotations are",
    )
    parser.add_argument(
        "--position_data_path",
        type=str,
        default="./data_0724/poses_synchronized",
        help="a path where position csv file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dataset_spec_2400_wo_mocap",
        help="a directory where sound dataset will be saved",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="sampling rate of sound data",
    )
    parser.add_argument(
        "--processed_method",
        type=str,
        default="music_intensity",
        help="a path where processed mic data",
    )
    parser.add_argument(
        "--denoise_method",
        type=str,
        default="",
        help="description of denoise method ex. wpe, inv_conv_subtract",
    )
    return parser.parse_args()


class SpecExtractor:
    def __init__(self, fs, nfft):

        self.nfft = nfft
        self.melW = librosa.filters.mel(
            sr=fs,
            n_fft=nfft,
        )

    def logmel(self, sig):
        S = (
            np.abs(
                librosa.stft(
                    y=sig,
                    n_fft=self.nfft,
                    hop_length=self.nfft,
                    center=False
                )
            )
            ** 2
        )
        S_mel = np.dot(self.melW, S).T
        S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
        S_logmel = np.expand_dims(S_logmel, axis=0)

        return S_logmel

    def intensity(self, sig):

        ref = sig[0]
        x = sig[1]
        y = sig[2]
        z = sig[3]
        music_r = sig[4]
        music_l = sig[5]

        Pref = librosa.stft(
            y=ref,
            n_fft=self.nfft,
            hop_length=self.nfft,
            center=False,
        )
        Px = librosa.stft(
            y=x,
            n_fft=self.nfft,
            hop_length=self.nfft,
            center=False,
        )
        Py = librosa.stft(
            y=y,
            n_fft=self.nfft,
            hop_length=self.nfft,
            center=False,
        )
        Pz = librosa.stft(
            y=z,
            n_fft=self.nfft,
            hop_length=self.nfft,
            center=False,
        )
        
        music_r = librosa.stft(
            y=music_r,
            n_fft=self.nfft,
            hop_length=self.nfft,
            center=False,
        )
        music_l = librosa.stft(
            y=music_l,
            n_fft=self.nfft,
            hop_length=self.nfft,
            center=False,
        )
        
        I1 = np.real(np.conj(Pref) * Px)
        I2 = np.real(np.conj(Pref) * Py)
        I3 = np.real(np.conj(Pref) * Pz)
        normal = np.sqrt(I1 ** 2 + I2 ** 2 + I3 ** 2)
        I1 = np.dot(self.melW, I1 / normal).T
        I2 = np.dot(self.melW, I2 / normal).T
        I3 = np.dot(self.melW, I3 / normal).T
        intensity = np.array([I1, I2, I3])
        
        phasew = np.dot(self.melW, np.angle(Pref)).T
        phasex = np.dot(self.melW, np.angle(Px)).T
        phasey = np.dot(self.melW, np.angle(Py)).T
        phasez = np.dot(self.melW, np.angle(Pz)).T
        phaser = np.dot(self.melW, np.angle(music_r)).T
        phasel = np.dot(self.melW, np.angle(music_l)).T
        phase = np.array([phasew, phasex, phasey, phasez, phaser, phasel])
        return intensity, phase

    def transform(self, audio):
        """ Extract Lomel and Intensity vector features

        Args:
            audio: multi channel audio [channel, sound_length] 

        Returns:
           features: Logmel spectrogram (4 channels) + Intensity Vector (3 channels), [7, melbin, time]
        """
        channel_num = audio.shape[0]
        feature_logmel = []
        for n in range(0, channel_num):
            feature_logmel.append(self.logmel(audio[n]))
        feature_intensity, phase = self.intensity(sig=audio)

        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature = np.concatenate([feature_logmel, feature_intensity, phase], axis=0)
        feature = feature.transpose(0, 2, 1)
        return feature


def read_annotation(path: str):
    anno = {}
    with open(path) as f:
        for line in f.readlines():
            start, end, pose = line.split(" ")
            pose = pose.split("\n")[0]
            anno[pose] = (float(start), float(end))

    return anno


def joint_norm(joint_locations: Dict[str, float]):
    shift = {}
    magnification = 0
    for dim in ["_x", "_y", "_z"]:
        shift[dim[-1]] = joint_locations["Hips" + dim]

        magnification += (
            joint_locations["Spine" + dim] - joint_locations["Hips" + dim]
        ) ** 2
    magnification = np.sqrt(magnification)
    for k, v in joint_locations.items():
        joint_locations[k] = (v - shift[k[-1]]) / magnification
    for joint in get_joint_names():
        shift["y"] = min(shift["y"], joint_locations[joint + "_y"])
    for joint in get_joint_names():
        joint_locations[joint + "_y"] = joint_locations[joint + "_y"] - shift["y"]
        
def timeseries_joint_norm(joint_locations):
    output = pd.DataFrame(joint_locations)
    output['scale'] = 0
    for dim_ in ["_x", '_y', '_z']:
        output['shift' + dim_] = output["Hips" + dim_].values
        output['scale'] += (
            output["Spine" + dim_].values - output["Hips" + dim_].values
        ) ** 2
    output['scale'] = np.sqrt(output['scale'].values)
    for joint in joint_locations.keys():
        dim_ = joint.split('_')[-1]
        output[joint] = (output[joint] - output['shift_' + dim_]) / output['scale'].values
    for joint in get_joint_names():
        output['shift_y'] = np.minimum(output['shift_y'], output[joint + '_y'])
    for joint in get_joint_names():
        output[joint + "_y"] = output[joint + "_y"] - output['shift_y']
    return output
        


def main() -> None:
    args = get_arguments()

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)
    if args.processed_method == 'music_intensity':
        dataset_name = "music_with_intensity"
    else:
        dataset_name = "sound_with_intensity"
    dataset_dir = os.path.join(args.save_dir, dataset_name)
    DEBUG = False
    anno_path_list = glob.glob(os.path.join(args.annotation_dir, "*"))
    anno_path_list = [anno_path for anno_path in anno_path_list if "wo_mocap" in anno_path]
    target_music = ['cirrus', 'arnor', 'mantron', 'jazz', 'techno']
    if args.processed_method == 'music_intensity':
        csv_file_name = 'music_intensity.csv'
    else:
        csv_file_name = "sound_intensity.csv"
    # すでにデータセットが存在しているなら終了
    if os.path.exists(dataset_dir):
        exist_flg = True
        print("Sound dataset exists.")
        prev_data = pd.read_csv(os.path.join(args.save_dir, csv_file_name))
        prev_data['annto_path_'] = args.annotation_dir + prev_data['testee'] + '_' + prev_data['music_type'] + '.txt'
        exist_anno_path = prev_data['annto_path_'].unique()
        anno_path_list = [anno_path for anno_path in anno_path_list if anno_path not in exist_anno_path]
        if DEBUG==False:
            return
    else:
        exist_flg = False
        os.mkdir(dataset_dir)
    anno_path_list = glob.glob(os.path.join(args.annotation_dir, "*"))
    anno_path_list = [anno_path for anno_path in anno_path_list if "wo_mocap" in anno_path]
    if DEBUG:
        anno_path_list = anno_path_list[:3]
    print(len(anno_path_list))
    # sound_length_list = get_sound_length_list()
    spec_duration_list = [0.6]
    joint_names = get_joints()

    columns = ["sound_path", "label", "testee", "music_type", "spec_duration", "joint_path"]

    data: Dict[str, List[Union[int, str]]] = {name: [] for name in columns}
    for anno_idx, anno_path in enumerate(anno_path_list):
        if ('techno' in anno_path) | ('subject_1' not in anno_path):
            continue
        data_name = anno_path.split("/")[-1].split(".")[0]
        if args.denoise_method != '':
            mic_path = os.path.join(args.sound_dir, data_name + "_" + args.denoise_method + ".wav")
        else:
            mic_path = os.path.join(args.sound_dir, data_name + "_cut.wav")
        position_path = os.path.join(args.position_data_path, data_name + "_position.csv")
        music_type = anno_path.split('/')[-1].split('_')[-1].split('.')[0]
        if music_type not in target_music:
            continue
        if "demo" in data_name:
            continue
        print("anno path:", anno_path)
        print(
            "%s/%s Making dataset from %s"
            % (anno_idx + 1, len(anno_path_list), data_name)
        )
        for spec_duration in spec_duration_list:
            print(spec_duration)
            # Read OptiTrack
            sound, sr_target = sf.read(mic_path)
            anno = read_annotation(anno_path)
            testee = 'subject_' + anno_path.split('/')[-1].split('_')[1]
            try:
                music_path = os.path.join(args.sound_dir, music_type + "_input.mp3")
                music_sound, sr_orig = sf.read(music_path)
            except:
                music_path = os.path.join(args.sound_dir, music_type + "_input.WAV")
                music_sound, sr_orig = sf.read(music_path)
            music_sound = librosa.resample(y = music_sound, orig_sr=sr_orig, target_sr=sr_target, res_type='kaiser_fast', axis=0)
            sample_num = int(np.round(len(sound)/sr_target / spec_duration))  # シーケンス数
            feature_extractor = SpecExtractor(args.sr, nfft=2400)

            for frame in tqdm.tqdm(range(sample_num)):  # データセットの各フレームに対して処理を実行
                if DEBUG:
                    if frame > 50:break
                sound_ = sound[int(np.round(sr_target * spec_duration * frame)) : int(np.round(sr_target * spec_duration * (frame + 1)))]
                second = frame * spec_duration  # 秒数timestampを取得
                for pose, (start, end) in anno.items():
                    if pose == "no_people":
                        continue
                    if second < start:
                        continue
                    if end < second:
                        continue
                    if args.processed_method == 'music_intensity':
                        processed_file_name = "%s_%s_%s_music_intensity.npy"
                        sound_ = sound_.T # [channels, time]
                        music_sound_ = music_sound[int(np.round(sr_target * spec_duration * frame)) : int(np.round(sr_target * spec_duration * (frame + 1)))]
                        music_sound_ = music_sound_.T 
                        sound_ = np.concatenate([sound_, music_sound_], axis=0)
                        sound_ = feature_extractor.transform(sound_)
                    else:
                        message = (
                            "There is no preprocessing method appropriate to your choice. "
                        )
                        raise ValueError(message)
                    data["label"].append(pose)
                    sound_path = os.path.join(
                        dataset_dir,
                        processed_file_name % (data_name, frame, spec_duration),
                    )
                    data['joint_path'].append("")
                    data["sound_path"].append(sound_path)
                    data["testee"].append(testee)
                    data["spec_duration"].append(spec_duration)
                    data["music_type"].append(music_type)
                    np.save(sound_path, sound_.astype("float32"))

    # list を DataFrame に変換
    df = pd.DataFrame(
        data,
        columns=columns,
    )
    if (DEBUG==False):
        if (exist_flg == True):
            df = pd.concat([df, prev_data])
            df = df.reset_index(drop=True)
    # 保存
    df.to_csv(os.path.join(args.save_dir, csv_file_name), index=None)

    print("Finished making sound dataset.")


if __name__ == "__main__":
    main()
