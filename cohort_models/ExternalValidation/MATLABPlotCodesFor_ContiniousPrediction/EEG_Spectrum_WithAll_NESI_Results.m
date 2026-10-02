clc;
close all;
clear;
%%
THIS_FILE = matlab.desktop.editor.getActiveFilename;

if isempty(THIS_FILE)
    error('Save this MATLAB script inside the NESI repository before running it.');
end

current_dir = fileparts(THIS_FILE);
EXTERNAL_VALIDATION_ROOT = '';

while true
    [parent_dir,folder_name] = fileparts(current_dir);

    if strcmpi(folder_name,'ExternalValidation')
        EXTERNAL_VALIDATION_ROOT = current_dir;
        break;
    end

    if isempty(parent_dir) || strcmp(parent_dir,current_dir)
        break;
    end

    current_dir = parent_dir;
end

if isempty(EXTERNAL_VALIDATION_ROOT)
    error('Could not find ExternalValidation above: %s',THIS_FILE);
end

COHORT_MODELS_ROOT = fileparts(EXTERNAL_VALIDATION_ROOT);
NESI_ROOT = fileparts(COHORT_MODELS_ROOT);

fprintf('MATLAB file:              %s\n',THIS_FILE);
fprintf('NESI root:                %s\n',NESI_ROOT);
fprintf('ExternalValidation root:  %s\n',EXTERNAL_VALIDATION_ROOT);

%% AUTO-DETECT PATHS
THIS_FILE = matlab.desktop.editor.getActiveFilename;

if isempty(THIS_FILE)
    error('Save this MATLAB script inside the NESI repository before running it.');
end

current_dir = fileparts(THIS_FILE);
EXTERNAL_VALIDATION_ROOT = '';

while true
    [parent_dir,folder_name] = fileparts(current_dir);

    if strcmpi(folder_name,'ExternalValidation')
        EXTERNAL_VALIDATION_ROOT = current_dir;
        break;
    end

    if isempty(parent_dir) || strcmp(parent_dir,current_dir)
        break;
    end

    current_dir = parent_dir;
end

if isempty(EXTERNAL_VALIDATION_ROOT)
    error('Could not locate ExternalValidation above MATLAB script: %s',THIS_FILE);
end

%% BUILD ALL PATHS RELATIVE TO ExternalValidation
PRODUCTS_ROOT = fullfile(EXTERNAL_VALIDATION_ROOT,'Products');
source_eeg_pathname      = fullfile(PRODUCTS_ROOT,'ErikaSegmentEEG');
RASS_prediction_pathname = fullfile(PRODUCTS_ROOT,'RASSPredictions');
GCS_prediction_pathname  = fullfile(PRODUCTS_ROOT,'GCSPredictions');
CAMS_prediction_pathname = fullfile(PRODUCTS_ROOT,'CAMSPredictions');
NESI_prediction_pathname = fullfile(PRODUCTS_ROOT,'NESIPredictions');

EEG_Viewing_ROOT = fullfile(EXTERNAL_VALIDATION_ROOT,'EEG_viewing_codes');


image_saving_dir = fullfile(EXTERNAL_VALIDATION_ROOT,'Plots_RASS_GCS_CAMS_NESI');

% Put Callbacks somewhere inside ExternalValidation
jj_callback_pathname = fullfile(EEG_Viewing_ROOT,'Callbacks');

fprintf('\nExternalValidation root:\n%s\n\n',EXTERNAL_VALIDATION_ROOT);

% ADD CALLBACKS
if isfolder(jj_callback_pathname)
    addpath(jj_callback_pathname);
else
    warning('Callbacks folder not found: %s',jj_callback_pathname);
end

% CHECK MAIN EEG DIRECTORY
if ~isfolder(source_eeg_pathname)
    error('EEG folder does not exist: %s',source_eeg_pathname);
end

% CREATE OUTPUT DIRECTORY
if ~isfolder(image_saving_dir)
    mkdir(image_saving_dir);
end

% FIND ALL MAT FILES RECURSIVELY
eeg_files = dir(fullfile(source_eeg_pathname,'**','*.mat'));

fprintf('Total MAT files found: %d\n',length(eeg_files));

if isempty(eeg_files)
    error('No MAT files found recursively inside: %s',source_eeg_pathname);
end

%% PROCESS ALL MAT FILES
close all;

n_files = length(eeg_files);
n_completed = 0;
n_skipped = 0;
n_failed = 0;

for s = 1:n_files
    tic

    input_eegfile = fullfile(eeg_files(s).folder,eeg_files(s).name);
    [~,subject_id,~] = fileparts(eeg_files(s).name);

    fprintf('\n================================================================================\n');
    fprintf('Processing %d/%d ==> %s\n',s,n_files,subject_id);
    fprintf('EEG: %s\n',input_eegfile);
    fprintf('================================================================================\n');

    input_RASSpred_CORNlogit_file = fullfile(RASS_prediction_pathname,[subject_id '_predictions.csv']);
    input_GCSpred_CORNlogit_file  = fullfile(GCS_prediction_pathname,[subject_id '_predictions.csv']);
    input_CAMSpred_CORNlogit_file = fullfile(CAMS_prediction_pathname,[subject_id '_predictions.csv']);
    input_NESI_file               = fullfile(NESI_prediction_pathname,[subject_id '_predictions.csv']);

    missing_files = {};

    if ~isfile(input_RASSpred_CORNlogit_file)
        missing_files{end+1} = 'RASS';
    end
    if ~isfile(input_GCSpred_CORNlogit_file)
        missing_files{end+1} = 'GCS';
    end
    if ~isfile(input_CAMSpred_CORNlogit_file)
        missing_files{end+1} = 'CAMS';
    end
    if ~isfile(input_NESI_file)
        missing_files{end+1} = 'NESI';
    end

    if ~isempty(missing_files)
        fprintf('Missing prediction file(s): %s\n',strjoin(missing_files,', '));
        fprintf('Skipping %s...\n',subject_id);
        n_skipped = n_skipped + 1;
        continue;
    end

    try
        fprintf('All prediction files found. Generating figure...\n');

        RASS_GCS_CAMS_NESI_EEG_Prediction_viz( ...
            input_eegfile, ...
            input_RASSpred_CORNlogit_file, ...
            input_GCSpred_CORNlogit_file, ...
            input_CAMSpred_CORNlogit_file, ...
            input_NESI_file, ...
            image_saving_dir, ...
            "yes");

        n_completed = n_completed + 1;
        fprintf('Completed: %s\n',subject_id);

    catch ME
        n_failed = n_failed + 1;
        fprintf(2,'FAILED: %s\n',subject_id);
        fprintf(2,'Error: %s\n',ME.message);
    end

    fprintf('Time: %.2f seconds\n',toc);
    close all;
end

fprintf('\n================================================================================\n');
fprintf('FINISHED PROCESSING ALL MAT FILES\n');
fprintf('Total     : %d\n',n_files);
fprintf('Completed : %d\n',n_completed);
fprintf('Skipped   : %d\n',n_skipped);
fprintf('Failed    : %d\n',n_failed);
fprintf('================================================================================\n');