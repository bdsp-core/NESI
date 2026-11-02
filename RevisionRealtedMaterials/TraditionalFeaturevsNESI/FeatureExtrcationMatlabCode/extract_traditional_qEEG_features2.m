
function [SEF95,RelativeDelta,DeltaAlpha,Filename,channels]=extract_traditional_qEEG_features2(csv_file,eeg_root,output_dir,dataset_name,split_name,Fs)

df=readtable(csv_file,'VariableNamingRule','preserve');
dataset_col=string(df.("Dataset"));
split_col=string(df.("Split"));
idx=strcmpi(strtrim(dataset_col),dataset_name);

if ~strcmpi(split_name,'All')
    idx=idx & strcmpi(strtrim(split_col),split_name);
end

df=df(idx,:);
if height(df)==0
    error('No matching rows found.');
end
if ~ismember('MorgothOutputFilename',df.Properties.VariableNames)
    error('MorgothOutputFilename column not found.');
end

Filename=string(df.("MorgothOutputFilename"));
N=height(df);
Nch=19;
channels={};
SEF95=NaN(N,Nch);
RelativeDelta=NaN(N,Nch);
DeltaAlpha=NaN(N,Nch);
FeatureStatus=zeros(N,1,'uint8');

if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

dataset_safe=regexprep(dataset_name,'[^a-zA-Z0-9]','_');
split_safe=regexprep(split_name,'[^a-zA-Z0-9]','_');
prefix=sprintf('%s_%s',dataset_safe,split_safe);

sef_file=fullfile(output_dir,[prefix '_SEF95.mat']);
reldelta_file=fullfile(output_dir,[prefix '_RelativeDelta.mat']);
deltaalpha_file=fullfile(output_dir,[prefix '_DeltaAlpha.mat']);
checkpoint_file=fullfile(output_dir,[prefix '_checkpoint.mat']);

backup_sef=fullfile(output_dir,[prefix '_SEF951.mat']);
backup_reldelta=fullfile(output_dir,[prefix '_RelativeDelta1.mat']);
backup_deltaalpha=fullfile(output_dir,[prefix '_DeltaAlpha1.mat']);

start_idx=1;

if exist(checkpoint_file,'file')
    C=load(checkpoint_file);
    if ~isequal(string(C.Filename(:)),Filename(:))
        error('Checkpoint filename order does not match CSV.');
    end
    SEF95=C.SEF95;
    RelativeDelta=C.RelativeDelta;
    DeltaAlpha=C.DeltaAlpha;
    FeatureStatus=C.FeatureStatus;
    channels=C.channels;
    start_idx=C.last_completed+1;
    fprintf('RESUMING CHECKPOINT AT %d/%d\n',start_idx,N);

elseif exist(backup_sef,'file') && exist(backup_reldelta,'file') && exist(backup_deltaalpha,'file')
    fprintf('LOADING ORIGINAL BACKUP FILES...\n');

    A=load(backup_sef,'SEF95','Filename','channels');
    B=load(backup_reldelta,'RelativeDelta','Filename');
    C=load(backup_deltaalpha,'DeltaAlpha','Filename');

    if ~isequal(string(A.Filename(:)),Filename(:)) || ...
       ~isequal(string(B.Filename(:)),Filename(:)) || ...
       ~isequal(string(C.Filename(:)),Filename(:))
        error('Backup filename order does not match current CSV.');
    end

    if ~isequal(size(A.SEF95),[N Nch]) || ...
       ~isequal(size(B.RelativeDelta),[N Nch]) || ...
       ~isequal(size(C.DeltaAlpha),[N Nch])
        error('Backup feature matrix dimensions do not match.');
    end

    SEF95=A.SEF95;
    RelativeDelta=B.RelativeDelta;
    DeltaAlpha=C.DeltaAlpha;
    channels=A.channels;

    completed=any(isfinite(SEF95),2) | ...
              any(isfinite(RelativeDelta),2) | ...
              any(isfinite(DeltaAlpha),2);

    first_unprocessed=find(~completed,1,'first');

    if isempty(first_unprocessed)
        start_idx=N+1;
    else
        start_idx=first_unprocessed;
    end

    if any(completed(start_idx:end))
        error('Noncontiguous saved rows detected. Inspect backup before resuming.');
    end

    FeatureStatus(1:start_idx-1)=1;
    fprintf('BACKUP RECOVERED: %d completed rows\n',start_idx-1);
    fprintf('RESUMING FROM: %d/%d\n',start_idx,N);

elseif exist(sef_file,'file') || exist(reldelta_file,'file') || exist(deltaalpha_file,'file')
    error('Existing output files found without checkpoint. Inspect before proceeding.');

else
    fprintf('No backup found. Starting from row 1.\n');
end

if start_idx>1
    last_completed=start_idx-1;
    save(checkpoint_file,'SEF95','RelativeDelta','DeltaAlpha', ...
        'Filename','channels','FeatureStatus','last_completed', ...
        'dataset_name','split_name','Fs','-v7.3');
end

for i=start_idx:N
    fprintf('================================================================================\n');
    fprintf('Processing %d/%d\n',i,N);
    fprintf('Morgoth filename: %s\n',Filename(i));

    mat_filename=regexprep(char(Filename(i)),'\.csv$','.mat','ignorecase');
    mat_file=fullfile(eeg_root,mat_filename);

    row_SEF=zeros(1,Nch);
    row_RelDelta=zeros(1,Nch);
    row_DeltaAlpha=zeros(1,Nch);

    try
        if ~exist(mat_file,'file')
            D=dir(fullfile(eeg_root,'**',mat_filename));
            if isempty(D)
                error('EEG MAT file not found.');
            end
            mat_file=fullfile(D(1).folder,D(1).name);
        end

        fprintf('MAT: %s\n',mat_file);
        S=load(mat_file,'data','channels');

        if ~isfield(S,'data') || ~isfield(S,'channels')
            error('Missing data or channels variable.');
        end

        EEG=double(S.data);
        current_channels=S.channels;

        if size(EEG,1)~=Nch || length(current_channels)~=Nch
            error('Invalid EEG dimensions or channel count.');
        end

        fprintf('EEG size: %d x %d\n',size(EEG,1),size(EEG,2));

        if isempty(channels)
            channels=current_channels;
        end

        for ch=1:Nch
            try
                x=double(EEG(ch,:));
                x=x(:);
                valid=isfinite(x);

                if sum(valid)<Fs
                    fprintf('Channel %d: insufficient samples; zero-filled.\n',ch);
                    continue;
                end

                if any(~valid)
                    x=fillmissing(x,'linear','EndValues','nearest');
                end

                P1=bandpower(x,Fs,[0.5 4]);
                P2=bandpower(x,Fs,[4 7]);
                P3=bandpower(x,Fs,[8 15]);
                P4=bandpower(x,Fs,[16 31]);
                P5=bandpower(x,Fs,[32 64]);
                P=P1+P2+P3+P4+P5;

                row_RelDelta(ch)=P1/(P+eps);
                row_DeltaAlpha(ch)=P1/(P3+eps);

                window_length=4*Fs;
                overlap=round(0.5*window_length);
                nfft=max(2048,2^nextpow2(window_length));
                [pxx,f]=pwelch(x,hamming(window_length),overlap,nfft,Fs);

                use_idx=f>=0.5 & f<=64;
                pxx_use=pxx(use_idx);
                f_use=f(use_idx);

                cumulative_power=cumtrapz(f_use,pxx_use);
                total_power=cumulative_power(end);

                if total_power>0
                    target=0.95*total_power;
                    kk=find(cumulative_power>=target,1,'first');

                    if kk==1
                        row_SEF(ch)=f_use(kk);
                    elseif ~isempty(kk)
                        p_low=cumulative_power(kk-1);
                        p_high=cumulative_power(kk);
                        f_low=f_use(kk-1);
                        f_high=f_use(kk);

                        if p_high>p_low
                            row_SEF(ch)=f_low+(target-p_low)*(f_high-f_low)/(p_high-p_low);
                        else
                            row_SEF(ch)=f_use(kk);
                        end
                    end
                end
            catch MEch
                fprintf(2,'Channel %d failed: %s\n',ch,MEch.message);
            end
        end

        SEF95(i,:)=row_SEF;
        RelativeDelta(i,:)=row_RelDelta;
        DeltaAlpha(i,:)=row_DeltaAlpha;
        FeatureStatus(i)=1;

        fprintf('SEF95 median         : %.4f Hz\n',median(row_SEF,'omitnan'));
        fprintf('RelativeDelta median : %.4f\n',median(row_RelDelta,'omitnan'));
        fprintf('DeltaAlpha median    : %.4f\n',median(row_DeltaAlpha,'omitnan'));

    catch ME
        SEF95(i,:)=zeros(1,Nch);
        RelativeDelta(i,:)=zeros(1,Nch);
        DeltaAlpha(i,:)=zeros(1,Nch);
        FeatureStatus(i)=2;

        fprintf(2,'FAILED FILE: %s\n',mat_file);
        fprintf(2,'REASON: %s\n',ME.message);
        fprintf(2,'All 19 channels zero-filled. Continuing.\n');
    end

    if mod(i,25)==0 || i==N
        last_completed=i;

        save(checkpoint_file,'SEF95','RelativeDelta','DeltaAlpha', ...
            'Filename','channels','FeatureStatus','last_completed', ...
            'dataset_name','split_name','Fs','-v7.3');

        save(sef_file,'SEF95','Filename','channels','FeatureStatus', ...
            'dataset_name','split_name','Fs','-v7.3');
        save(reldelta_file,'RelativeDelta','Filename','channels','FeatureStatus', ...
            'dataset_name','split_name','Fs','-v7.3');
        save(deltaalpha_file,'DeltaAlpha','Filename','channels','FeatureStatus', ...
            'dataset_name','split_name','Fs','-v7.3');

        fprintf('CHECKPOINT SAVED: %d/%d\n',i,N);
    end
end

fprintf('\n================================================================================\n');
fprintf('FINISHED\n');
fprintf('Successful/recovered: %d\n',sum(FeatureStatus==1));
fprintf('Failed/zero-filled: %d\n',sum(FeatureStatus==2));
fprintf('SEF95: %d x %d\n',size(SEF95));
fprintf('RelativeDelta: %d x %d\n',size(RelativeDelta));
fprintf('DeltaAlpha: %d x %d\n',size(DeltaAlpha));
fprintf('Saved:\n%s\n%s\n%s\n',sef_file,reldelta_file,deltaalpha_file);

end
