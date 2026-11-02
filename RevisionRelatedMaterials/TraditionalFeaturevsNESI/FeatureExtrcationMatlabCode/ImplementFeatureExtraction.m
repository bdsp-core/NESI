clc;
clear;
close all;

%% SETTINGS
csv_file='/home/ayush/Desktop/NESI_review_material/TraditionalFeaturevsNESI/Final_NESIscores_full_dataset_result.csv';

dataset_name='GCS';     % 'RASS', 'GCS', 'CAMS', 'ICANS', etc.
split_name='All';       % 'Train', 'Val', 'Test', or 'All'

eeg_root='/media/ayush/Expansion/GCS_EEGs_10min_restcases';
output_dir='/home/ayush/Desktop/NESI_review_material/TraditionalFeaturevsNESI';

Fs=200;

%% RUN
[SEF95,RelativeDelta,DeltaAlpha,Filename,channels]=extract_traditional_qEEG_features2(...
    csv_file,eeg_root,output_dir,dataset_name,split_name,Fs);

