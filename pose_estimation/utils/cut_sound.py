import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import glob


def main():
    delay_dict = {
            # 'subject_0_cirrus':21.78, 'subject_0_arnor':27.37, 'subject_0_mantron':18.62,
            # 'subject_1_techno':14.26, 'subject_1_jazz':15.09, 'subject_1_cirrus':13.47, 'subject_1_arnor':13.42, 'subject_1_mantron':13.28,
            # 'subject_2_techno':18.83, 'subject_2_jazz':40.87, 'subject_2_cirrus':15.28,
            # 'subject_3_techno':23.21, 'subject_3_jazz':13.88, 'subject_3_cirrus':16.76,
            # 'subject_4_jazz':37.07, 'subject_4_cirrus':15.35,
            # 'subject_5_jazz':17.77, 'subject_5_cirrus':16.02,
            # 'subject_6_cirrus':12.93, 'subject_6_arnor':14.85, 'subject_6_mantron':21.17,
            # 'subject_7_cirrus':15.78, 'subject_7_arnor':23.15, 'subject_7_mantron':22.76,
            # 'subject_8_cirrus':15.46, 'subject_8_arnor':16.70, 'subject_8_mantron':14.15,
            # 'subject_9_cirrus':17.71, 'subject_9_arnor':22.03, 'subject_9_mantron':13.98,
            'subject_0_wo_mocap_mantron':16.91,'subject_0_wo_mocap_arnor':16.71,'subject_0_wo_mocap_cirrus':19.57,
            'subject_1_wo_mocap_mantron':12.36,'subject_1_wo_mocap_arnor':12.59,'subject_1_wo_mocap_cirrus':12.74,
        }
    for name, delay in delay_dict.items():
        tmp_path = f'data/sound/{name}.WAV'
        tmp_sound, sr = sf.read(tmp_path)
        start = int(np.round(delay * sr))
        print(name + ':' + str(start))
        tmp_sound = tmp_sound[start:, :]
        filepath = f"data/sound_cut/{name}_cut.wav" # 23/5/8 パスを変更
        _format = "WAV"
        subtype = 'PCM_24'
        sf.write(filepath, tmp_sound, sr, format=_format, subtype=subtype)

    
if __name__ == "__main__":
    main()