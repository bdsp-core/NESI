function [SEF95,RelativeDelta,DeltaAlpha,Filename,channels]=extract_traditional_qEEG_features(csv_file,eeg_root,output_dir,dataset_name,split_name,Fs)

df=readtable(csv_file,'VariableNamingRule','preserve');

dataset_col=string(df.("Dataset"));
split_col=string(df.("Split"));

idx=strcmpi(strtrim(dataset_col),dataset_name);

if ~strcmpi(split_name,'All')
    idx=idx & strcmpi(strtrim(split_col),split_name);
end

df=df(idx,:);

fprintf('\nDataset : %s\n',dataset_name);
fprintf('Split   : %s\n',split_name);
fprintf('EEGs    : %d\n\n',height(df));

if height(df)==0
    error('No rows found for Dataset=%s, Split=%s.',dataset_name,split_name);
end

if ~ismember('MorgothOutputFilename',df.Properties.VariableNames)
    error('MorgothOutputFilename column not found in CSV.');
end

Filename=string(df.("MorgothOutputFilename"));
N=height(df);
Nch=19;

SEF95=NaN(N,Nch);
RelativeDelta=NaN(N,Nch);
DeltaAlpha=NaN(N,Nch);
channels={};

if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

dataset_safe=regexprep(dataset_name,'[^a-zA-Z0-9]','_');
split_safe=regexprep(split_name,'[^a-zA-Z0-9]','_');

sef_file=fullfile(output_dir,sprintf('%s_%s_SEF95.mat',dataset_safe,split_safe));
reldelta_file=fullfile(output_dir,sprintf('%s_%s_RelativeDelta.mat',dataset_safe,split_safe));
deltaalpha_file=fullfile(output_dir,sprintf('%s_%s_DeltaAlpha.mat',dataset_safe,split_safe));

for i=1:N

    fprintf('================================================================================\n');
    fprintf('Processing %d/%d\n',i,N);
    fprintf('Morgoth filename: %s\n',Filename(i));

    % Replace final .csv with .mat
    mat_filename=regexprep(char(Filename(i)),'\.csv$','.mat','ignorecase');

    fprintf('EEG filename    : %s\n',mat_filename);

    % First check directly inside EEG root
    mat_file=fullfile(eeg_root,mat_filename);

    % If not directly present, search recursively
    if ~exist(mat_file,'file')
        D=dir(fullfile(eeg_root,'**',mat_filename));

        if ~isempty(D)
            mat_file=fullfile(D(1).folder,D(1).name);
        else
            fprintf('NOT FOUND: %s\n',mat_filename);
            continue;
        end
    end

    fprintf('MAT: %s\n',mat_file);

    S=load(mat_file,'data','channels');

    if ~isfield(S,'data')
        fprintf('FAILED: Variable "data" not found.\n');
        continue;
    end

    if ~isfield(S,'channels')
        fprintf('FAILED: Variable "channels" not found.\n');
        continue;
    end

    EEG=double(S.data);
    current_channels=S.channels;

    if size(EEG,1)~=Nch
        fprintf('FAILED: Expected 19 channels but data has size %d x %d.\n',size(EEG,1),size(EEG,2));
        continue;
    end

    if length(current_channels)~=Nch
        fprintf('FAILED: Expected 19 channel names but found %d.\n',length(current_channels));
        continue;
    end

    fprintf('EEG size: %d x %d\n',size(EEG,1),size(EEG,2));

    if isempty(channels)
        channels=current_channels;
    end

    % if isempty(channels)
    %     channels=current_channels;
    %     fprintf('Reference channel order stored.\n');
    % else
    %     ref_channels=string(channels(:));
    %     this_channels=string(current_channels(:));
    % 
    %     if ~isequal(ref_channels,this_channels)
    %         error('Channel ordering mismatch in file: %s',mat_file);
    %     end
    % end

    for ch=1:Nch

        x=EEG(ch,:);
        x=double(x(:));

        valid=isfinite(x);

        if sum(valid)<Fs
            fprintf('Channel %d (%s): insufficient valid samples.\n',ch,string(current_channels{ch}));
            continue;
        end

        if any(~valid)
            x=fillmissing(x,'linear','EndValues','nearest');
        end

        % Same frequency bands as fcn_computeFeatures_powers_complex_v2
        P1=bandpower(x,Fs,[0.5 4]);
        P2=bandpower(x,Fs,[4 7]);
        P3=bandpower(x,Fs,[8 15]);
        P4=bandpower(x,Fs,[16 31]);
        P5=bandpower(x,Fs,[32 64]);

        P=P1+P2+P3+P4+P5;

        % Relative delta
        RelativeDelta(i,ch)=P1/(P+eps);

        % Delta/alpha
        DeltaAlpha(i,ch)=P1/(P3+eps);

        % SEF95
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
                SEF95(i,ch)=f_use(kk);

            elseif ~isempty(kk)

                p_low=cumulative_power(kk-1);
                p_high=cumulative_power(kk);
                f_low=f_use(kk-1);
                f_high=f_use(kk);

                if p_high>p_low
                    SEF95(i,ch)=f_low+(target-p_low)*(f_high-f_low)/(p_high-p_low);
                else
                    SEF95(i,ch)=f_use(kk);
                end
            end
        end
    end

    fprintf('SEF95 median         : %.4f Hz\n',median(SEF95(i,:),'omitnan'));
    fprintf('RelativeDelta median : %.4f\n',median(RelativeDelta(i,:),'omitnan'));
    fprintf('DeltaAlpha median    : %.4f\n',median(DeltaAlpha(i,:),'omitnan'));

    if mod(i,25)==0
        fprintf('Saving checkpoint...\n');
        save(sef_file,'SEF95','Filename','channels','dataset_name','split_name','Fs','-v7.3');
        save(reldelta_file,'RelativeDelta','Filename','channels','dataset_name','split_name','Fs','-v7.3');
        save(deltaalpha_file,'DeltaAlpha','Filename','channels','dataset_name','split_name','Fs','-v7.3');
    end
end

save(sef_file,'SEF95','Filename','channels','dataset_name','split_name','Fs','-v7.3');
save(reldelta_file,'RelativeDelta','Filename','channels','dataset_name','split_name','Fs','-v7.3');
save(deltaalpha_file,'DeltaAlpha','Filename','channels','dataset_name','split_name','Fs','-v7.3');

fprintf('\n================================================================================\n');
fprintf('FINISHED\n');
fprintf('================================================================================\n');
fprintf('SEF95         : %d x %d\n',size(SEF95,1),size(SEF95,2));
fprintf('RelativeDelta : %d x %d\n',size(RelativeDelta,1),size(RelativeDelta,2));
fprintf('DeltaAlpha    : %d x %d\n',size(DeltaAlpha,1),size(DeltaAlpha,2));

fprintf('\nChannel order:\n');
for ch=1:length(channels)
    fprintf('%2d: %s\n',ch,string(channels{ch}));
end

fprintf('\nSaved:\n%s\n%s\n%s\n',sef_file,reldelta_file,deltaalpha_file);

end