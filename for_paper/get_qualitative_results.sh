1. single-music, mocap
start_time=42
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/subject_6_arnor_demo.mp4 --model RGB
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp078/model=music2pose_freq_atten_ver10-dataset_name=kfold_same_music_wo_subject_1_3_subject_6_all-input_feature=music_intensity-music_cv_ver=1-seed=0/images/test_gt.mp4 --model gt
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp078/model=music2pose_freq_atten_ver10-dataset_name=kfold_same_music_wo_subject_1_3_subject_6_all-input_feature=music_intensity-music_cv_ver=1-seed=0/images/test_prediction.mp4 --model ours
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp095_subject_6/images/test.mp4 --model sound2pose
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp104_subject_6/images/test.mp4 --model speech2pose
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp116-121_wipose/exp116/model=wipose_lstm-max_epoch=50-dataset_name=kfold_same_music_wo_subject_1_3_subject_6_all-input_feature=music_intensity-music_cv_ver=1/images/test_prediction.mp4 --model wipose
python ./for_paper/draw_figure.py --start_time $start_time

2. cross-music, mocap
start_time=60
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/subject_6_arnor_demo.mp4 --model RGB --video_title subject_6_arnor_cross_music_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp071_080/exp074/model=music2pose_freq_atten_ver4-dataset_name=kfold_same_music_wo_subject_1_3_subject_6_all-input_feature=music_intensity-seq_len=12-music_cv_ver=1/images/test.mp4 --model gt --video_title subject_6_arnor_cross_music_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp090/model=music2pose_freq_atten_ver4-max_epoch=50-dataset_name=kfold_same_music_wo_subject_1_3_subject_6_all-input_feature=music_intensity-music_cv_ver=8/images/test_prediction.mp4 --model ours --video_title subject_6_arnor_cross_music_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp093_subject_6_mocap/images/test_prediction.mp4 --model sound2pose --video_title subject_6_arnor_cross_music_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp108_subject_6_mocap/images/test_prediction.mp4 --model speech2pose --video_title subject_6_arnor_cross_music_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp116-121_wipose/exp120/model=wipose_lstm-max_epoch=50-dataset_name=kfold_same_music_wo_subject_1_3_subject_6_all-input_feature=music_intensity-music_cv_ver=8/images/test_prediction.mp4 --model wipose --video_title subject_6_arnor_cross_music_mocap
python ./for_paper/draw_figure.py --start_time $start_time --video_title subject_6_arnor_cross_music_mocap

# start_time=80
start_time=80
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/shibatie/SSD/Music2Pose/pose_estimation_with_noise/subject_1_arnor_wo_mocap2_demo.mp4 --model RGB --video_title subject_1_arnor_wo_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/shibatie/SSD/Music2Pose/pose_estimation_with_noise/result/exp063_subject6_music_11_wo_mocap/images/test_prediction.mp4 --model ours --video_title subject_1_arnor_wo_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp093_subject_1_wo_mocap/images/test.mp4 --model sound2pose --video_title subject_1_arnor_wo_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp120_subject_1_wo_mocap/images/test.mp4 --model wipose --video_title subject_1_arnor_wo_mocap
python ./for_paper/get_prediction_img.py --start_time $start_time --img_path /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/exp108_subject_1_wo_mocap/images/test.mp4 --model speech2pose --video_title subject_1_arnor_wo_mocap
python ./for_paper/draw_figure.py --start_time $start_time --video_title subject_1_arnor_wo_mocap