function RASS_GCS_CAMS_NESI_EEG_Prediction_viz(input_eegfile, ...
    input_RASSpred_CORN_logitfile, input_GCSpred_CORN_logitfile, ...
    input_CAMSpred_CORN_logitfile, input_NESI_file, outDir, want_to_save)

    %%%% read data
    tmp=load(input_eegfile);
    data=tmp.data; data_= data;Fs=double(tmp.Fs);

    %%% Image file name
    [~, img_file_name, ~] = fileparts(input_eegfile);

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

    %%%% plot spectrum code
    close all;
    f = figure('units','normalized','position',[0.0182,0,0.4818,0.9467], ...
        'color','w','MenuBar','none','ToolBar','none','HitTest','off');

    % ---- Layout: 4 spectrograms + RASS + GCS + CAMS + NESI ----
    h = 0.125;
    y = [0.865, 0.725, 0.585, 0.445];
    ax_spec = {
        subplot('position',[.040, y(1), .93, h]);
        subplot('position',[.040, y(2), .93, h]);
        subplot('position',[.040, y(3), .93, h]);
        subplot('position',[.040, y(4), .93, h])
    };

    total_samples = length(stime);
    x_start = 0;
    x_end = total_samples - 1;

    % Automatically choose a reasonable number of X-axis ticks
    n_ticks = min(10, total_samples);
    x_ticks = round(linspace(x_start, x_end, n_ticks));
    x_tick_labels = string(x_ticks);

    nn=size(sdata,1)/4;reg_tag={'LL','RL','LP','RP'};colormap('jet');

    for i=1:size(ax_spec,1)
        set(f,'CurrentAxes',ax_spec{i});cla(ax_spec{i})
        hold(ax_spec{i},'on')
        spec=pow2db(sdata((i-1)*nn+1:i*nn,:)+eps);
        imagesc(ax_spec{i},stime,sfreq,spec,[-10,25]);axis(ax_spec{i},'xy');
        yticks=get(ax_spec{i},'ytick');yticklabels=get(ax_spec{i},'yticklabel');
        yticklabels{end}=reg_tag{i};ylabel(ax_spec{i},'Freq (Hz)', 'fontsize',12);
        set(ax_spec{i},'ylim',[sfreq(1),sfreq(end)+.1],'xlim',[stime(1),stime(end)], ...
            'yticklabel',yticklabels,'ytick',yticks,'xtick',[],'box','on')
        if i==4
            set(ax_spec{i},'xtick',x_ticks,'xticklabel',x_tick_labels,'fontsize',8);
            xlabel(ax_spec{i}, 'Time (s)','FontSize',9,'FontWeight','bold');
        end
        set(ax_spec{i},'Box','on','LineWidth',1.5,'XColor','k','YColor','k');
        hold(ax_spec{i},'off')
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % RASS CORN
    % ORIGINAL COMPUTATION FLOW UNCHANGED
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    T = readtable(input_RASSpred_CORN_logitfile);

    rass_preds = T.RASSMappingClass;

    % ---- Compute continuous ordinal score from CORN logits ----
    corn_logits = T{:, {'logit_0','logit_1','logit_2','logit_3','logit_4'}};
    cond_probs = 1 ./ (1 + exp(-corn_logits));   % sigmoid, size N x 5
    ordinal_score = sum(cond_probs, 2);          % N x 1, continuous in (0,5)
    rass_score = ordinal_score - 5;              % shift: class 5 -> RASS 0, class 0 -> RASS -5
    rass_score = rass_score(:)';                 % force row vector

    % CORN logits come from a sliding window over the EEG signal
    corn_idx = 1:length(rass_score);

    % ---- New axes for continuous score strip plot ----
    ax_corn = axes('position',[.040, .310, .93, .075], 'Parent', f);
    hold(ax_corn, 'on');

    rass_centers = [-5, -4, -3, -2, -1, 0];
    rass_labels  = {'RASS -5','RASS -4','RASS -3','RASS -2','RASS -1','RASS 0'};
    strip_colors = [
        0.780 0.780 1.000;
        0.682 0.733 0.941;
        0.702 0.933 0.878;
        0.953 0.953 0.690;
        0.961 0.812 0.620;
        0.890 0.702 0.659
    ];

    y_min = -5.5; y_max = 0.5;

    % background strips
    for k = 1:numel(rass_centers)
        yl = rass_centers(k) - 0.5;
        yu = rass_centers(k) + 0.5;
        patch(ax_corn,[corn_idx(1) corn_idx(end) corn_idx(end) corn_idx(1)], ...
            [yl yl yu yu],strip_colors(k,:),'EdgeColor','none');
    end

    % dashed boundary lines
    for boundary = y_min:1:y_max
        plot(ax_corn,[corn_idx(1) corn_idx(end)],[boundary boundary], ...
            'k--','LineWidth',1);
    end

    % continuous score trace
    plot(ax_corn,corn_idx,rass_score,'b-','LineWidth',0.8);

    set(ax_corn,'YLim',[y_min,y_max],'XLim',[corn_idx(1),corn_idx(end)], ...
        'YTick',rass_centers,'YTickLabel',rass_labels,'XTick',[], ...
        'FontSize',7,'Box','on','LineWidth',1.5,'XColor','k','YColor','k');

    title(ax_corn,'RASS Continuous Score','FontSize',9,'FontWeight','bold');
    hold(ax_corn,'off');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GCS CORN
    % Class 0 = Severe
    % Class 1 = Moderate
    % Class 2 = Mild
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    T_GCS = readtable(input_GCSpred_CORN_logitfile);

    gcs_preds = T_GCS.GCSMappingClass;

    % ---- Compute continuous ordinal score from CORN logits ----
    gcs_corn_logits = T_GCS{:, {'logit_0','logit_1'}};
    gcs_cond_probs = 1 ./ (1 + exp(-gcs_corn_logits));
    gcs_ordinal_score = sum(gcs_cond_probs, 2);
    gcs_score = gcs_ordinal_score;
    gcs_score = gcs_score(:)';
    gcs_corn_idx = 1:length(gcs_score);

    % ---- GCS axes ----
    ax_gcs = axes('position',[.040, .215, .93, .065], 'Parent', f);
    hold(ax_gcs, 'on');

    gcs_centers = [0, 1, 2];
    gcs_labels = {'Severe','Moderate','Mild'};
    gcs_strip_colors = [
        0.890 0.702 0.659;
        0.953 0.953 0.690;
        0.702 0.933 0.878
    ];

    gcs_y_min = -0.5;
    gcs_y_max = 2.5;

    % background strips
    for k = 1:numel(gcs_centers)
        yl = gcs_centers(k) - 0.5;
        yu = gcs_centers(k) + 0.5;
        patch(ax_gcs, ...
            [gcs_corn_idx(1) gcs_corn_idx(end) gcs_corn_idx(end) gcs_corn_idx(1)], ...
            [yl yl yu yu],gcs_strip_colors(k,:),'EdgeColor','none');
    end

    % dashed boundary lines
    for boundary = gcs_y_min:1:gcs_y_max
        plot(ax_gcs,[gcs_corn_idx(1) gcs_corn_idx(end)], ...
            [boundary boundary],'k--','LineWidth',1);
    end

    % continuous GCS score
    plot(ax_gcs,gcs_corn_idx,gcs_score,'b-','LineWidth',0.8);

    set(ax_gcs,'YLim',[gcs_y_min,gcs_y_max], ...
        'XLim',[gcs_corn_idx(1),gcs_corn_idx(end)], ...
        'YTick',gcs_centers,'YTickLabel',gcs_labels,'XTick',[], ...
        'FontSize',7,'Box','on','LineWidth',1.5,'XColor','k','YColor','k');

    title(ax_gcs,'GCS Continuous Score','FontSize',9,'FontWeight','bold');
    hold(ax_gcs,'off');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CAMS CORN
    % Class 0 = Mild
    % Class 1 = Moderate
    % Class 2 = Severe
    % 3 ordinal classes -> 2 CORN logits
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    T_CAMS = readtable(input_CAMSpred_CORN_logitfile);

    cams_preds = T_CAMS.CAMSMappingClass;

    % ---- Compute continuous ordinal score from CORN logits ----
    cams_corn_logits = T_CAMS{:, {'logit_0','logit_1'}};
    cams_cond_probs = 1 ./ (1 + exp(-cams_corn_logits));
    cams_ordinal_score = sum(cams_cond_probs, 2);
    cams_score = cams_ordinal_score;
    cams_score = cams_score(:)';
    cams_corn_idx = 1:length(cams_score);

    % ---- CAMS axes ----
    ax_cams = axes('position',[.040, .120, .93, .065], 'Parent', f);
    hold(ax_cams, 'on');

    cams_centers = [0, 1, 2];
    cams_labels = {'Mild','Moderate','Severe'};
    cams_strip_colors = [
        0.702 0.933 0.878;
        0.953 0.953 0.690;
        0.890 0.702 0.659
    ];

    cams_y_min = -0.5;
    cams_y_max = 2.5;

    % background strips
    for k = 1:numel(cams_centers)
        yl = cams_centers(k) - 0.5;
        yu = cams_centers(k) + 0.5;
        patch(ax_cams, ...
            [cams_corn_idx(1) cams_corn_idx(end) cams_corn_idx(end) cams_corn_idx(1)], ...
            [yl yl yu yu],cams_strip_colors(k,:),'EdgeColor','none');
    end

    % dashed boundary lines
    for boundary = cams_y_min:1:cams_y_max
        plot(ax_cams,[cams_corn_idx(1) cams_corn_idx(end)], ...
            [boundary boundary],'k--','LineWidth',1);
    end

    % continuous CAMS score
    plot(ax_cams,cams_corn_idx,cams_score,'b-','LineWidth',0.8);

    set(ax_cams,'YLim',[cams_y_min,cams_y_max], ...
        'XLim',[cams_corn_idx(1),cams_corn_idx(end)], ...
        'YTick',cams_centers,'YTickLabel',cams_labels,'XTick',[], ...
        'FontSize',7,'Box','on','LineWidth',1.5,'XColor','k','YColor','k');

    title(ax_cams,'CAMS-LF Continuous Score','FontSize',9,'FontWeight','bold');
    hold(ax_cams,'off');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NESI PREDICTION
    % CSV contains one column: NESI
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    T_NESI = readtable(input_NESI_file);
    nesi_score = T_NESI.NESI;
    nesi_score = nesi_score(:)';
    nesi_idx = 1:length(nesi_score);

    ax_nesi = axes('position',[.040, .025, .93, .065], 'Parent', f);
    hold(ax_nesi,'on');

    plot(ax_nesi,nesi_idx,nesi_score,'b-','LineWidth',0.8);

    nesi_min = min(nesi_score);
    nesi_max = max(nesi_score);
    if nesi_min == nesi_max
        nesi_min = nesi_min - 0.1;
        nesi_max = nesi_max + 0.1;
    end

    set(ax_nesi,'YLim',[-3, 3], ...
        'XLim',[nesi_idx(1),nesi_idx(end)], ...
        'XTick',[],'FontSize',7,'Box','on', ...
        'LineWidth',1.5,'XColor','k','YColor','k');

    title(ax_nesi,'NESI Prediction','FontSize',9,'FontWeight','bold');
    hold(ax_nesi,'off');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    filePath = fullfile(outDir,[img_file_name '_EEG_RASS_GCS_CAMS_NESIPredPlot.png']);
    set(gcf,'WindowState','maximized');

    if want_to_save == "yes"
        saveas(gcf,filePath);
    else
        disp('Figure did not save')
    end
end