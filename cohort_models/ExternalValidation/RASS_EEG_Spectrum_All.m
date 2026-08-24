clc;
close all;
clear all;
%%
% Change these Paths as per your computer's directory
source_eeg_pathname = '/home/ayush/Desktop/KIMCHI_LAB_DATA/ErikaSegmentEEG';  
jj_callback_pathname = '/home/ayush/Desktop/EEG_viewing_codes/Callbacks';
prediction_pathname = '/home/ayush/Desktop/KIMCHI_LAB_DATA/RASSPredictions';
image_saving_dir = '/home/ayush/Desktop/KIMCHI_LAB_DATA/Plots_RASSPred';

addpath(jj_callback_pathname);

subject_dirs = dir(source_eeg_pathname);
subject_dirs = subject_dirs([subject_dirs.isdir]);
subject_dirs = subject_dirs(~ismember({subject_dirs.name},{'.','..'}));

%%
close all;

for s = 1:length(subject_dirs)
    tic
    subject_id = subject_dirs(s).name;
    subject_path = fullfile(source_eeg_pathname, subject_id);
    disp(['Processing subject ==> ', subject_id]);

    eeg_files = dir(fullfile(subject_path, '*.mat'));

    if isempty(eeg_files)
        disp(['No MAT file found for ', subject_id, '. Skipping...']);
        continue;
    end

    input_eegfile = fullfile(subject_path, eeg_files(1).name);
    input_RASSpred_CORNlogit_file = fullfile(prediction_pathname, [subject_id '_predictions.csv']);

    if ~isfile(input_RASSpred_CORNlogit_file)
        disp(['No prediction CSV found for ', subject_id, '. Skipping...']);
        continue;
    end

    RASS_EEG_Prediction_viz(input_eegfile, input_RASSpred_CORNlogit_file, ...
        image_saving_dir, "yes");

    toc
end
