#!/bin/bash

## Required parameters
#--dataset IIIC or SPIKES or FOC_GEN_SPIKES or BS or SLOWING or NORMAL or SLEEPPSG or MGBSLEEP3stages
#--data_format edf or mat
#--eval_sub_dir /xxx/xxx/ (input EEG Mat file data dir)
#--eval_results_dir /xxx/xxx/ (output result dir for MORGOTH ACTIVATION)
#--prediction_slipping_step xxx (Step size in points; if original hz>200 prediction_slipping_step better to be 100 or 128)
#  or use --prediction_slipping_step_second xxx (step size in seconds)

##### Optional parameters: If the original data contains channel names and sampling rate information, the following parameters can be omitted from the command.
#--sampling_rate 0 or xxx (If the raw data does not contain this information, it should be assigned here; 0 indicates that the information is present in the data.)
#--already_format_channel_order yes (If the data does not include channel information, it needs to be sorted as required before being input.)
#--already_average_montage yes (If the data has already been average montaged, it should be specified.)
#--allow_missing_channels yes or no (If the data does not include all 19 channels, processing is still allowed — the missing channels will be zero-filled.)

##### Optional parameters: For 1-second spike detection
#--smooth_result ema or window_ema or ''
#--need_spikes_10s_result yes (summarize 10-second results from 1-second predictions.)
#--spikes_10s_result_slipping_step_second xx (sliding step in second for 10-second spike detection)

##### More optional parameters:
#--polarity 1 or -1 with default 1 (If set -1, the signal is inverted)
#--max_length_hour no or 1,2,3...(Only analyze the first n hours of the EEG)
#--leave_one_hemisphere_out no or left or right or middel (Set the EEG signals to 0 for the left, right, or middle hemisphere)
#--rewrite_results no | yes (Default no Overwrite the original results when new results are available.)

########## PLEASE READ THIS ##################
# Please check the paths of EEG and Results directories
# IF YOU DO NOT HAVE GPU THEN USE "--device cpu" and "--distributed False"

# 1. IIIC-------------------------------------------------------------------------

OMP_NUM_THREADS=1 python finetune_classification.py \
    --abs_pos_emb \
    --model base_patch200_200 \
    --predict \
    --task_model checkpoints/IIIC.pth \
    --dataset IIIC \
    --data_format mat \
    --sampling_rate 200 \
    --already_format_channel_order no \
    --already_average_montage yes \
    --allow_missing_channels no \
    --max_length_hour no \
    --leave_one_hemisphere_out no \
    --polarity 1 \
    --eval_sub_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/ErikaSegmentEEG" \
    --eval_results_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/MorgothAcivtaions_Segments/IIIC" \
    --prediction_slipping_step_second 1 \
    --device cuda \
    --distributed False

# 2. Focal/Generalized Spikes--------------------------------------------------------
OMP_NUM_THREADS=1 python finetune_classification.py \
    --abs_pos_emb \
    --model base_patch200_200 \
    --predict \
    --task_model checkpoints/FOCGENSPIKES.pth \
    --dataset FOC_GEN_SPIKES \
    --data_format mat \
    --sampling_rate 200 \
    --already_format_channel_order no \
    --already_average_montage yes \
    --allow_missing_channels no \
    --max_length_hour no \
    --polarity 1 \
    --leave_one_hemisphere_out no \
    --eval_sub_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/ErikaSegmentEEG" \
    --eval_results_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/MorgothAcivtaions_Segments/FOCGEN" \
    --prediction_slipping_step_second 1 \
    --device cuda \
    --distributed False

# 3. Slowing--------------------------------------------------------
OMP_NUM_THREADS=1 python finetune_classification.py \
    --abs_pos_emb \
    --model base_patch200_200 \
    --predict \
    --task_model checkpoints/SLOWING.pth \
    --dataset SLOWING \
    --data_format mat \
    --sampling_rate 200 \
    --already_format_channel_order no \
    --already_average_montage yes \
    --allow_missing_channels no \
    --max_length_hour no \
    --polarity 1 \
    --leave_one_hemisphere_out no \
    --eval_sub_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/ErikaSegmentEEG" \
    --eval_results_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/MorgothAcivtaions_Segments/SLOWING" \
    --prediction_slipping_step_second 1 \
    --device cuda \
    --distributed False

# 4. BS--------------------------------------------------------
OMP_NUM_THREADS=1 python finetune_classification.py \
    --abs_pos_emb \
    --model base_patch200_200 \
    --predict \
    --task_model checkpoints/BS.pth \
    --dataset BS \
    --data_format mat \
    --sampling_rate 200 \
    --already_format_channel_order no \
    --already_average_montage yes \
    --allow_missing_channels no \
    --max_length_hour no \
    --leave_one_hemisphere_out no \
    --polarity 1 \
    --eval_sub_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/ErikaSegmentEEG" \
    --eval_results_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/MorgothAcivtaions_Segments/BS" \
    --prediction_slipping_step_second 1 \
    --device cuda \
    --distributed False

# 5. NORMAL--------------------------------------------------------
OMP_NUM_THREADS=1 python finetune_classification.py \
    --abs_pos_emb \
    --model base_patch200_200 \
    --predict \
    --task_model checkpoints/NORMAL.pth \
    --dataset NORMAL \
    --data_format mat \
    --sampling_rate 200 \
    --already_format_channel_order no \
    --already_average_montage yes \
    --allow_missing_channels no \
    --max_length_hour no \
    --polarity 1 \
    --leave_one_hemisphere_out no \
    --eval_sub_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/ErikaSegmentEEG" \
    --eval_results_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/MorgothAcivtaions_Segments/NORMAL" \
    --prediction_slipping_step_second 1 \
    --device cuda \
    --distributed False

# 6. SLEEP--------------------------------------------------------
OMP_NUM_THREADS=1 python finetune_classification.py \
    --predict \
    --model base_patch200_200 \
    --task_model checkpoints/SLEEP.pth \
    --abs_pos_emb \
    --dataset MGBSLEEP3stages \
    --data_format mat \
    --sampling_rate 200 \
    --already_format_channel_order no \
    --already_average_montage yes \
    --allow_missing_channels no \
    --max_length_hour no \
    --polarity 1 \
    --leave_one_hemisphere_out no \
    --eval_sub_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/ErikaSegmentEEG" \
    --eval_results_dir "/z/ClinicalProjects/ClinicalVideoEEG/DataVideoEEG_2023/ArkaProject_FilestoShare/Github_Repository/NESI/cohort_models/ExternalValidation/ProductsV2/MorgothAcivtaions_Segments/SLEEP" \
    --prediction_slipping_step_second 1 \
    --device cuda \
    --distributed False