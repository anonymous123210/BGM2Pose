import pandas as pd
import numpy as np
import glob 
import argparse
import cv2
import os
import shutil


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make csv files for Sound Pose dataset"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="./pose_estimation/result/exp001/exp001/model=music2pose_freq_atten_ver4-dataset_name=kfold_same_music_wo_subject_1_3_subject_6_all-input_feature=music_intensity-seq_len=12-music_cv_ver=1/images/test.mp4",
        help="path to class label csv",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./for_paper/images",
        help="path to class label csv",
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gt',
    )
    parser.add_argument(
        '--start_time',
        type=int,
        default=120,
    )
    parser.add_argument(
        '--video_title',
        type=str,
        default='subject_6_arnor_single',
    )

    return parser.parse_args()
def main():
    args = get_arguments()
    cap = cv2.VideoCapture(args.img_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total frame num:', total_frame)
    max_frame = 5
    img_count = 0
    if fps > 20:
        fps = 30
    start_frame = args.start_time * fps
    if os.path.exists(os.path.join(args.save_path, args.video_title, args.model, "start_" + str(args.start_time))):
        shutil.rmtree(os.path.join(args.save_path, args.video_title, args.model, "start_" + str(args.start_time)))
    os.makedirs(
        os.path.join(args.save_path, args.video_title, args.model, "start_" + str(args.start_time)),
        exist_ok=True,
    )
    print('fps:', fps)
    if "RGB" in args.video_title:
        start_frame += 0
    else:
        start_frame += 20
    hop_length = int(fps * 16)
    print(start_frame)
    for i in range(int(start_frame)):
        ret, frame = cap.read()
    for i in range(0, int(total_frame)):
        ret, frame = cap.read()
        # if i < start_frame:
        #     continue
        if img_count > max_frame:
            break
        if i %  hop_length == 0:
            tmp_save_path = os.path.join(args.save_path, args.video_title, args.model, "start_" + str(args.start_time) ,f'{i}.jpg')
            # print(tmp_save_path)
            try:
                if "RGB" in args.model:
                    h,w,c = frame.shape
                    frame = frame[450:450+500, 800:800+500,:]
                    # print(frame.shape)
                    # assert False
                else:
                    h,w,c = frame.shape
                    frame = frame[h//2 - 110:h//2+110, w//2 - 110:w//2+110,:]
                cv2.imwrite(tmp_save_path, frame)
                img_count += 1
            except:
                pass
    return 

if __name__ == '__main__':
    main()