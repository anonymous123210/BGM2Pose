export CUDA_VISIBLE_DEVICES=0

python utils/make_sound_dataset_wo_mocap.py --processed_method music_intensity
python utils/make_csv_files.py --dataset_name kfold_wo_mocap --music arnor cirrus mantron jazz --subject_name subject_0 --sound_csv_path ./dataset_wo_mocap/music_intensity.csv --preprocess music_intensity
python utils/make_configs.py --batch_size 256 --lr_max 0.003 --lr_min 0.001 --max_epoch 50 --sound_length 2400 --input_feature music_intensity --dataset_name kfold_wo_mocap_subject_0 kfold_wo_mocap_subject_0 --music_cv_ver 6 6 --model music2pose_freq_atten_ver4 music2pose_freq_atten_ver4 --smooth_loss True --seq_len 12
python utils/make_configs.py --batch_size 256 --lr_max 0.003 --lr_min 0.001 --max_epoch 50 --sound_length 2400 --input_feature music_intensity --dataset_name kfold_wo_mocap_subject_0 kfold_wo_mocap_subject_0 --music_cv_ver 8 8 --model wipose_lstm wipose_lstm --smooth_loss True --seq_len 12
python evaluate.py /home/ubuntu/slocal/Music2Pose/pose_estimation_with_noise/result/model=music2pose_freq_atten_ver4-dataset_name=kfold_wo_mocap_subject_1-music_cv_ver=8/config.yaml test --output_type prediction