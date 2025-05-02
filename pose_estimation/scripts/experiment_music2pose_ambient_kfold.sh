export CUDA_VISIBLE_DEVICES=0
python utils/make_sound_spec_dataset.py --processed_method music_intensity --save_dir ./dataset_spec_2400

python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_0 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity
python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_6 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity
python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_7 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity
python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_8 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity
python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_9 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity
python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_2 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity
python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_4 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity
python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_1_3 --music arnor cirrus mantron jazz --subject_name subject_5 --sound_csv_path ./dataset_spec_2400/music_intensity.csv --preprocess music_intensity

python utils/make_configs.py --batch_size 64 --lr_max 0.003 --lr_min 0.001 --max_epoch 30 --spec_duration 0.6 --input_feature music_intensity music_intensity --dataset_name  kfold_same_music_wo_subject_1_3_subject_0_all kfold_same_music_wo_subject_1_3_subject_6_all kfold_same_music_wo_subject_1_3_subject_7_all kfold_same_music_wo_subject_1_3_subject_8_all kfold_same_music_wo_subject_1_3_subject_9_all --music_cv_ver 10 10 --model music2pose_freq_atten_ver10 music2pose_freq_atten_ver10 --smooth_loss True --seq_len 12 12 --logit_scale 0.07 --seed 0 1 2

# python utils/make_configs.py --batch_size 256 --lr_max 0.003 --lr_min 0.001 --max_epoch 50 --sound_length 2400 --input_feature music_intensity music_intensity --dataset_name kfold_same_music_wo_subject_3_subject_1_arnor kfold_same_music_wo_subject_3_subject_6_arnor kfold_same_music_wo_subject_3_subject_7_arnor kfold_same_music_wo_subject_3_subject_8_arnor kfold_same_music_wo_subject_3_subject_9_arnor --model speech2pose speech2pose --smooth_loss True True --dropout 0.0 0.0 --seq_len 12

files="./result/*music_cv_ver=10*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
        flag2="${filepath}/config.yaml"
        if [ ! -e $flag2 ] ; then
            continue
        fi
        echo "${filepath}"
        python train.py "${filepath}/config.yaml" #--use_wandb
        python evaluate.py "${filepath}/config.yaml" test #--music arnor
    fi
done