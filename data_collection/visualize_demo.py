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
import matplotlib.pyplot as plt


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make sound dataset for sound pose estimation"
    )
    parser.add_argument(
        "--sound_path",
        type=str,
        default="./data/sound_cut/",
        help="path to a sound dirctory",
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
                    n_fft=self.nfft
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

        Pref = librosa.stft(
            y=ref,
            n_fft=self.nfft,
        )
        Px = librosa.stft(
            y=x,
            n_fft=self.nfft,
        )
        Py = librosa.stft(
            y=y,
            n_fft=self.nfft,
        )
        Pz = librosa.stft(
            y=z,
            n_fft=self.nfft,
        )

        I1 = np.real(np.conj(Pref) * Px)
        I2 = np.real(np.conj(Pref) * Py)
        I3 = np.real(np.conj(Pref) * Pz)
        normal = np.sqrt(I1 ** 2 + I2 ** 2 + I3 ** 2)
        I1 = np.dot(self.melW, I1 / normal).T
        I2 = np.dot(self.melW, I2 / normal).T
        I3 = np.dot(self.melW, I3 / normal).T
        intensity = np.array([I1, I2, I3])
        return intensity

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
        feature_intensity = self.intensity(sig=audio)

        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature = np.concatenate([feature_logmel, feature_intensity], axis=0)
        feature = feature.transpose(0, 2, 1)
        return feature

def main():
    args = get_arguments()
    sound_, sr = sf.read(args.sound_path)
    sound_ = sound_.T #[channel, sound_length]
    sound_ = sound_[:, sr * 10:sr * 15]
    spec_extractor = SpecExtractor(fs=sr, nfft=2048)
    features = spec_extractor.transform(sound_)

    plt.figure(figsize=(18, 4))
    plt.subplot(1,2,1)
    ticks_loc = np.linspace(0, 128 - 8 - 1, len([0, 20, 40, 60, 80, 100, 110, 120]))
    plt.yticks(ticks_loc+8, reversed([0, 20, 40, 60, 80, 100, 110, 120]))
    plt.ylabel("frequency (mel scale)")
    plt.imshow(features[0][::-1,:], aspect='auto')

    plt.subplot(1,2,2)
    plt.yticks(ticks_loc+8, reversed([0, 20, 40, 60, 80, 100, 110, 120]))
    plt.ylabel("frequency (mel scale)")
    plt.imshow(features[4][::-1,:], aspect='auto')
    plt.show()
    plt.tight_layout(pad=0.1)
    plt.savefig('mel_intensity.png')

if __name__ == '__main__':
    main()
