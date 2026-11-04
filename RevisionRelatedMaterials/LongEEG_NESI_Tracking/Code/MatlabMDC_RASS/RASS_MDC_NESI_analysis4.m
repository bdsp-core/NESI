function RASS_MDC_NESI_analysis4(input_eegfile,input_NESI_file,outDir,want_to_save,plot_start_time,plot_end_time)

    %%%% read data
    tmp=load(input_eegfile);

    data=tmp.data; data_=data;Fs=double(tmp.Fs);

    RASSRecordedDTS=tmp.RASSRecordedDTS;
    RawScore=double(tmp.RawScore(:));
    Dataset=tmp.Dataset;

    % ---- Convert EEG start/end times ----
    EEGStartDTS=local_to_datetime(tmp.EEGStartDTS);
    EEGEndDTS=local_to_datetime(tmp.EEGEndDTS);

    % ---- Convert RASS recording times ----
    RASSRecordedDTS=local_to_datetime_array(RASSRecordedDTS);

    %%% Image file name
    [~,img_file_name,~]=fileparts(input_eegfile);

    %%%% denoise
    data(isnan(data))=eps;
    [B1,A1]=butter(3,[0.5,70]/(Fs/2));[B2,A2]=iirnotch(60/(Fs/2),60/(35*Fs/2));
    data=filtfilt(B1,A1,data')';data=filtfilt(B2,A2,data')';data(isnan(data_))=NaN;

    %%%% compute spectrograms
    ww=1*Fs;num_seg=ceil(size(data,2)/ww)+1;stime=(0:1:(num_seg-1));
    params.movingwin=[4,1];params.tapers=[2,3];params.fpass=[.5,20];params.Fs=Fs;
    [sdata,~,sfreq]=fcn_compute_spec(fcn_bipolar(data),params);
    for j=1:size(sdata,1)
        s=sdata{j,2};
        sdata{j,2}=[s(:,1),s(:,1:end-1),repmat(s(:,end-1),1,num_seg-size(s,2))];
    end
    sdata=cell2mat(sdata(:,2));sdata(isnan(sdata))=eps;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % OPTIONAL DISPLAY TIME RANGE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargin<5 || isempty(plot_start_time)
        plot_start_sec=stime(1);
    else
        t0=datetime(string(plot_start_time),'InputFormat','HH:mm:ss');
        t0.Year=EEGStartDTS.Year;
        t0.Month=EEGStartDTS.Month;
        t0.Day=EEGStartDTS.Day;

        if t0<EEGStartDTS
            t0=t0+days(1);
        end

        plot_start_sec=seconds(t0-EEGStartDTS);
    end

    if nargin<6 || isempty(plot_end_time)
        plot_end_sec=stime(end);
    else
        t1=datetime(string(plot_end_time),'InputFormat','HH:mm:ss');
        t1.Year=EEGStartDTS.Year;
        t1.Month=EEGStartDTS.Month;
        t1.Day=EEGStartDTS.Day;

        if t1<EEGStartDTS
            t1=t1+days(1);
        end

        plot_end_sec=seconds(t1-EEGStartDTS);
    end

    plot_start_sec=max(plot_start_sec,stime(1));
    plot_end_sec=min(plot_end_sec,stime(end));

    if plot_start_sec>=plot_end_sec
        error('Invalid plot time range: start must occur before end.');
    end

    %%%% plot spectrum code
    close all;
    f=figure('units','normalized','position',[0.0182,0,0.4818,0.9467], ...
        'color','w','MenuBar','none','ToolBar','none','HitTest','off');

    % ---- Layout: 4 spectrograms + True RASS + NESI ----
    h=0.125;
    y=[0.865,0.725,0.585,0.445];

    ax_spec={
        subplot('position',[.040,y(1),.93,h]);
        subplot('position',[.040,y(2),.93,h]);
        subplot('position',[.040,y(3),.93,h]);
        subplot('position',[.040,y(4),.93,h])
    };

    total_samples=length(stime);
    x_start=0;
    x_end=total_samples-1;

    n_ticks=min(10,total_samples);
    x_ticks=round(linspace(x_start,x_end,n_ticks));

    % tick labels: elapsed hours from the start of the recording (no dates in figures)
    x_tick_labels=cell(1,length(x_ticks));

    for k=1:length(x_ticks)
        x_tick_labels{k}=sprintf('%.1f',x_ticks(k)/3600);
    end

    nn=size(sdata,1)/4;reg_tag={'LL','RL','LP','RP'};colormap('jet');

    for i=1:size(ax_spec,1)
        set(f,'CurrentAxes',ax_spec{i});cla(ax_spec{i})
        hold(ax_spec{i},'on')

        spec=pow2db(sdata((i-1)*nn+1:i*nn,:)+eps);

        imagesc(ax_spec{i},stime,sfreq,spec,[-10,25]);
        axis(ax_spec{i},'xy');

        yticks=get(ax_spec{i},'ytick');
        yticklabels=get(ax_spec{i},'yticklabel');

        yticklabels{end}=reg_tag{i};

        ylabel(ax_spec{i},'Freq (Hz)','fontsize',12);

        set(ax_spec{i}, ...
            'ylim',[sfreq(1),sfreq(end)+.1], ...
            'xlim',[stime(1),stime(end)], ...
            'yticklabel',yticklabels, ...
            'ytick',yticks, ...
            'xtick',[], ...
            'box','on');

        if i==4
            set(ax_spec{i}, ...
                'XTick',x_ticks, ...
                'XTickLabel',x_tick_labels, ...
                'TickLabelInterpreter','tex', ...
                'FontSize',8);

            xlabel(ax_spec{i},'Elapsed time (hours)','FontSize',9,'FontWeight','bold');
        end

        set(ax_spec{i}, ...
            'Box','on', ...
            'LineWidth',1.5, ...
            'XColor','k', ...
            'YColor','k');

        hold(ax_spec{i},'off')
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TRUE RASS + CONTINUOUS NESI
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    T_NESI=readtable(input_NESI_file);
    nesi_score=T_NESI.NESI;
    nesi_score=nesi_score(:)';

    if length(nesi_score)>1
        nesi_time=linspace(stime(1),stime(end),length(nesi_score));
    else
        nesi_time=stime(1);
    end

    % ---- RASS axis ----
    ax_rass=axes('position',[.040,.075,.93,.255],'Parent',f);
    hold(ax_rass,'on');

    rass_levels=[-5,-4,-3,-2,-1,0];

    for k=1:length(rass_levels)
        plot(ax_rass, ...
            [stime(1),stime(end)], ...
            [rass_levels(k),rass_levels(k)], ...
            'Color',[0.85 0.85 0.85], ...
            'LineStyle','--', ...
            'LineWidth',0.8);
    end

    % True RASS: preceding 30 min
    for k=1:min(length(RASSRecordedDTS),length(RawScore))

        rass_time=RASSRecordedDTS(k);
        rass_value=RawScore(k);

        if isnat(rass_time) || isnan(rass_value)
            continue
        end

        rass_end_sec=seconds(rass_time-EEGStartDTS);
        rass_start_sec=rass_end_sec-(30*60);

        plot_start=max(rass_start_sec,stime(1));
        plot_end=min(rass_end_sec,stime(end));

        if plot_end>=stime(1) && ...
           plot_start<=stime(end) && ...
           plot_end>=plot_start

            plot(ax_rass, ...
                [plot_start,plot_end], ...
                [rass_value,rass_value], ...
                'k-', ...
                'LineWidth',5);
        end
    end

    set(ax_rass, ...
        'Position',[.040,.075,.93,.255], ...
        'Color','none', ...
        'YAxisLocation','left', ...
        'YLim',[-5.5,0.5], ...
        'YDir','reverse', ...
        'XLim',[stime(1),stime(end)], ...
        'YTick',rass_levels, ...
        'YTickLabel',{'-5','-4','-3','-2','-1','0'}, ...
        'XTick',x_ticks, ...
        'XTickLabel',x_tick_labels, ...
        'TickLabelInterpreter','tex', ...
        'FontSize',8, ...
        'Box','on', ...
        'LineWidth',1.5, ...
        'XColor','k', ...
        'YColor','k');

    ylabel(ax_rass,'True RASS','FontSize',10,'FontWeight','bold');
    xlabel(ax_rass,'Elapsed time (hours)','FontSize',9,'FontWeight','bold');
    title(ax_rass,'True RASS and NESI Estimate','FontSize',10,'FontWeight','bold');

    % ---- NESI axis ----
    ax_nesi=axes('position',ax_rass.Position,'Parent',f);
    hold(ax_nesi,'on');

    plot(ax_nesi,nesi_time,nesi_score,'b-','LineWidth',1.2);

    set(ax_nesi, ...
        'Position',ax_rass.Position, ...
        'Color','none', ...
        'YAxisLocation','right', ...
        'XAxisLocation','bottom', ...
        'YLim',[-3,3], ...
        'YDir','normal', ...
        'XLim',[stime(1),stime(end)], ...
        'YTick',-3:1:3, ...
        'XTick',[], ...
        'FontSize',8, ...
        'Box','off', ...
        'XColor','none', ...
        'YColor','b');

    ylabel(ax_nesi,'NESI','FontSize',10,'FontWeight','bold','Color','b');

    uistack(ax_rass,'top');
    set(ax_rass,'Color','none');

    hold(ax_nesi,'off');
    hold(ax_rass,'off');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Link axes
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    linkaxes([ax_spec{1},ax_spec{2},ax_spec{3},ax_spec{4},ax_rass,ax_nesi],'x');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % APPLY DISPLAY RANGE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    xlim(ax_rass,[plot_start_sec,plot_end_sec]);

    n_display_ticks=10;

    x_ticks_display=round( ...
        linspace(plot_start_sec,plot_end_sec,n_display_ticks));

    % tick labels: elapsed hours from the start of the recording (no dates in figures)
    x_tick_labels_display=cell(1,length(x_ticks_display));

    for k=1:length(x_ticks_display)
        x_tick_labels_display{k}=sprintf('%.1f',x_ticks_display(k)/3600);
    end

    % Spectrogram axis
    set(ax_spec{4}, ...
        'XTick',x_ticks_display, ...
        'XTickLabel',x_tick_labels_display, ...
        'TickLabelInterpreter','tex');

    % RASS/NESI axis
    set(ax_rass, ...
        'XTick',x_ticks_display, ...
        'XTickLabel',x_tick_labels_display, ...
        'TickLabelInterpreter','tex');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if ~exist(outDir,'dir')
        mkdir(outDir);
    end

    filePath=fullfile( ...
        outDir, ...
        [img_file_name '_EEG_TrueRASS_NESICombinedPlot.png']);

    set(gcf,'WindowState','maximized');

    if strcmpi(string(want_to_save),"yes")
        saveas(gcf,filePath);
    else
        disp('Figure did not save')
    end
end


function dt=local_to_datetime(x)

    if iscell(x)
        x=x{1};
    end

    if isstring(x)
        x=x(1);
    end

    if ischar(x)
        x=strtrim(x);
    end

    formats={ ...
        'yyyy-MM-dd HH:mm:ss', ...
        'yyyy-MM-dd HH:mm:ss.SSS', ...
        'yyyy-MM-dd''T''HH:mm:ss', ...
        'yyyy-MM-dd''T''HH:mm:ss.SSS'};

    dt=NaT;

    for k=1:length(formats)
        try
            dt=datetime(x,'InputFormat',formats{k});

            if ~isnat(dt)
                return
            end
        catch
        end
    end

    try
        dt=datetime(x);
    catch
        error( ...
            'Could not convert datetime value: %s', ...
            string(x));
    end
end


function dt=local_to_datetime_array(x)

    if iscell(x)

        x=x(:);
        dt=NaT(size(x));

        for k=1:numel(x)
            dt(k)=local_to_datetime(x{k});
        end

    elseif isstring(x)

        x=x(:);
        dt=NaT(size(x));

        for k=1:numel(x)
            dt(k)=local_to_datetime(x(k));
        end

    elseif ischar(x)

        if size(x,1)>1

            dt=NaT(size(x,1),1);

            for k=1:size(x,1)
                dt(k)=local_to_datetime(strtrim(x(k,:)));
            end

        else
            dt=local_to_datetime(x);
        end

    else
        error( ...
            'Unsupported RASSRecordedDTS format: %s', ...
            class(x));
    end

    dt=dt(:);
end