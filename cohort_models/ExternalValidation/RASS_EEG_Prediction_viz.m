function RASS_EEG_Prediction_viz(input_eegfile, input_RASSpred_CORN_logitfile,  outDir, want_to_save)
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
    
    % ---- Layout (recomputed so 4 spectrograms + CORN panel + hard-pred
    % strip all fit without overlapping) ----
    h = 0.175;
    y = [0.80, 0.605, 0.41, 0.215];
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
                set(ax_spec{i},'ylim',[sfreq(1),sfreq(end)+.1],'xlim',[stime(1),stime(end)],'yticklabel',yticklabels,'ytick',yticks,'xtick',[],'box','on')        
                if i==4
                    set(ax_spec{i},'xtick',xticks,'xticklabel',x_tick_labels,'fontsize',11);
                end
                set(ax_spec{i}, 'Box', 'on', ...
                    'LineWidth', 1.5, ...
                    'XColor', 'k', ...
                    'YColor', 'k');
            hold(ax_spec{i},'off')
    end    
    
    %%% RASS CORN Continuous logits across various RASS level horizontal
    %%% strips (placed between bottom spectrogram and hard-pred strip)
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
    ax_corn = axes('position',[.040, .025, .93, .155], 'Parent', f);
    hold(ax_corn, 'on');
    rass_centers = [-5, -4, -3, -2, -1, 0];
    rass_labels  = {'RASS -5','RASS -4','RASS -3','RASS -2','RASS -1','RASS 0'};
    strip_colors = [
        0.780 0.780 1.000;   % RASS 0   - lavender/grey
        0.682 0.733 0.941;   % RASS -1  - light blue
        0.702 0.933 0.878;   % RASS -2  - light teal
        0.953 0.953 0.690;   % RASS -3  - light yellow
        0.961 0.812 0.620;   % RASS -4  - light orange
        0.890 0.702 0.659    % RASS -5  - light salmon
    ];
    
    y_min = -5.5; y_max = 0.5;
    
    % background strips
    for k = 1:numel(rass_centers)
        yl = rass_centers(k) - 0.5;
        yu = rass_centers(k) + 0.5;
        patch(ax_corn, [corn_idx(1) corn_idx(end) corn_idx(end) corn_idx(1)], ...
                        [yl yl yu yu], strip_colors(k,:), ...
                        'EdgeColor','none');
    end
    
    % dashed boundary lines
    for boundary = y_min:1:y_max
        plot(ax_corn, [corn_idx(1) corn_idx(end)], [boundary boundary], ...
             'k--', 'LineWidth', 1);
    end
    
    % continuous score trace, plotted against its own index
    plot(ax_corn, corn_idx, rass_score, 'b-', 'LineWidth', 0.8);
    
    set(ax_corn, 'YLim', [y_min, y_max],'XLim', [corn_idx(1), corn_idx(end)], ...
        'YTick', rass_centers, 'YTickLabel', rass_labels, ...
        'XTick', [], 'FontSize', 9, 'Box', 'on', ...
        'LineWidth', 1.5, 'XColor','k', 'YColor','k');
    
    ylabel(ax_corn, 'Continious Score', 'fontsize', 9);
    hold(ax_corn, 'off');
    
    %%%% Plot Hard RASS Predictions
%     ax_pred = axes('position',[.040, .025, .93, .09], 'Parent', f);
% 
%     strip_colors2 = [
%     0.55 0.55 0.85;   % RASS 0   - lavender/grey
%     0.45 0.55 0.85;   % RASS -1  - light blue
%     0.40 0.78 0.65;   % RASS -2  - light teal
%     0.82 0.80 0.30;   % RASS -3  - light yellow
%     0.85 0.60 0.30;   % RASS -4  - light orange
%     0.75 0.45 0.40    % RASS -5  - light salmon
% ];
% 
%     imagesc(ax_pred, stime, 1, rass_preds', [0 5]);
%     colormap(ax_pred, strip_colors2);
% 
%     set(ax_pred, 'YTick', [], 'XTick',xticks, 'XTickLabel', time_stamps, 'FontSize', 11, 'TickDir', 'none');
% 
%     text(ax_pred, ...
%         -0.034, 0.35, 'RASS Head', ...
%         'Units', 'normalized', ...
%         'Rotation', 90, ...
%         'FontSize', 10, ...
%         'HorizontalAlignment', 'center', ...
%         'VerticalAlignment', 'middle');
% 
%     hold(ax_pred, 'on');
%     p0 = plot(ax_pred, nan, nan, 's', 'MarkerFaceColor', strip_colors2(1,:), 'MarkerEdgeColor', 'k');  % RASS 0
%     p1 = plot(ax_pred, nan, nan, 's', 'MarkerFaceColor', strip_colors2(2,:), 'MarkerEdgeColor', 'k');  % RASS -1
%     p2 = plot(ax_pred, nan, nan, 's', 'MarkerFaceColor', strip_colors2(3,:), 'MarkerEdgeColor', 'k');  % RASS -2
%     p3 = plot(ax_pred, nan, nan, 's', 'MarkerFaceColor', strip_colors2(4,:), 'MarkerEdgeColor', 'k');  % RASS -3
%     p4 = plot(ax_pred, nan, nan, 's', 'MarkerFaceColor', strip_colors2(5,:), 'MarkerEdgeColor', 'k');  % RASS -4
%     p5 = plot(ax_pred, nan, nan, 's', 'MarkerFaceColor', strip_colors2(6,:), 'MarkerEdgeColor', 'k');  % RASS -5
% 
%     lgd = legend( ...
%         [p0 p1 p2 p3 p4 p5], ...
%         {'RASS 0','RASS -1','RASS -2','RASS -3','RASS -4','RASS -5'}, ...
%         'Orientation', 'horizontal', ...
%         'Location', 'southoutside', ...
%         'FontSize', 10, ...
%         'Box', 'off' ...
%     );
% 
%     lgd.ItemTokenSize = [8, 8];
    
    % keep zoom/pan synced across spectrograms + hard-pred strip
    % (ax_corn is excluded -- it uses CORN sample index, not EEG time,
    % since rass_score has a different length than stime)
    % linkaxes([ax_spec{1}, ax_spec{2}, ax_spec{3}, ax_spec{4}, ax_pred], 'x');
    % xlim([stime(1), stime(end)]);
    
    filePath = fullfile(outDir, [img_file_name '_EEG_RASSPredPlot.png']);
    set(gcf, 'WindowState', 'maximized');
    if want_to_save == "yes"
        saveas(gcf, filePath);
        
    else
        disp('Figure did not save')
    end
    
end