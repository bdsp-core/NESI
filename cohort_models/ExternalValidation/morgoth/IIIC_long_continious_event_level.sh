password="ayushbidmc" # add you password for sudo


echo "$password" | sudo -S  $(which python) segment_long_eeg.py segment \
            --data_format mat  \
            --segment_duration 600 \
            --eeg_dir /media/ayush/Expansion/abiee_eegs/dummy   \
            --eval_sub_dir /media/ayush/Expansion/abiee_eegs/MORGOTH_EMBEDDINGS_ABIEE/IIIC  \


# IIIC-------------------------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model checkpoints/IIIC.pth \
            --dataset IIIC \
            --data_format mat \
            --sampling_rate 200 \
            --already_format_channel_order yes \
            --already_average_montage no \
            --allow_missing_channels no \
             --max_length_hour no \
            --eval_sub_dir  /media/ayush/Expansion/abiee_eegs/dummy \
            --eval_results_dir /media/ayush/Expansion/abiee_eegs/MORGOTH_EMBEDDINGS_ABIEE/IIIC\
            --prediction_slipping_step_second 1 \
            --rewrite_results no

