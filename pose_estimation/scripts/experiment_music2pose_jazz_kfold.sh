export CUDA_VISIBLE_DEVICES=0
python utils/make_sound_dataset.py --processed_method intensity 
python utils/make_sound_dataset.py --processed_method music_intensity

# python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_3 --music jazz --subject_name subject_0 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
# python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_3 --music jazz --subject_name subject_1 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
# python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_3 --music jazz --subject_name subject_2 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
# python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_3 --music jazz --subject_name subject_4 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
# python utils/make_csv_files.py --dataset_name kfold_same_music_wo_subject_3 --music jazz --subject_name subject_5 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity

python utils/make_configs.py --batch_size 256 --lr_max 0.003 --lr_min 0.001 --max_epoch 50 --sound_length 2400 --input_feature all all --dataset_name kfold_same_music_wo_subject_3_subject_0_jazz kfold_same_music_wo_subject_3_subject_1_jazz kfold_same_music_wo_subject_3_subject_2_jazz  kfold_same_music_wo_subject_3_subject_4_jazz kfold_same_music_wo_subject_3_subject_5_jazz --model speech2pose speech2pose --smooth_loss True True --seq_len 12 12
files="./result/*seq_len=12*"
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
        cat "${filepath}"
        python train.py "${filepath}/config.yaml" --use_wandb
        # python evaluate.py "${filepath}/config.yaml" validation
        python evaluate.py "${filepath}/config.yaml" test
    fi
done