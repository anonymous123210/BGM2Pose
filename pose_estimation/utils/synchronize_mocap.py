import pandas as pd
import numpy as np
import glob
import os

def main():
    testee_delay_dict = {
        'subject_0_cirrus_position':1.040878, 'subject_0_jazz_position':0.64422,'subject_0_arnor_position':0.992760, 'subject_0_mantron_position':0.587795, 
        'subject_1_techno_position':0.6517, 'subject_1_jazz_position':0.79664897, 'subject_1_cirrus_position':0.6339414, 'subject_1_arnor_position':0.75258564, 'subject_1_mantron_position':0.634313821,
        'subject_2_techno_position':0.744547, 'subject_2_jazz_position':0.757358, 'subject_2_cirrus_position':0.63982,
        'subject_3_techno_position':0.7248535, 'subject_3_jazz_position':0.25171, 'subject_3_cirrus_position':0.2414190,
        'subject_4_jazz_position':0.649291, 'subject_4_cirrus_position':0.660280,
        'subject_5_jazz_position':0.651196, 'subject_5_cirrus_position':0.981673,
        'subject_6_cirrus_position':0.62620, 'subject_6_arnor_position':0.61568, 'subject_6_mantron_position':0.62575,
        'subject_7_cirrus_position':0.62928, 'subject_7_arnor_position':0.66509, 'subject_7_mantron_position':0.69609,
        'subject_8_cirrus_position':0.62069, 'subject_8_arnor_position':0.65937, 'subject_8_mantron_position':0.6338586,
        'subject_9_cirrus_position':0.69904923, 'subject_9_arnor_position':0.639220, 'subject_9_mantron_position':0.6427383,
        }
    pos_path_list = glob.glob("data/position_data_2023_1012_csv/*.csv")
    opti_fps = 120
    save_dir = 'data/poses_synchronized/'
    os.makedirs(save_dir, exist_ok=True)
    for pos_path in pos_path_list:
        opti_df = pd.read_csv(pos_path)
        opti_df = opti_df.reset_index(drop=True)
        testee = pos_path.split('position_data_2023_1012_csv/')[-1].split('.')[0]
        print(testee)
        delay = testee_delay_dict[testee]
        start = int(np.round(delay * opti_fps))
        new_opti_df = opti_df.iloc[start:].reset_index(drop=True)
        new_opti_df.to_csv(save_dir + testee + '.csv')
        
        
        
if __name__ == "__main__":
    main()