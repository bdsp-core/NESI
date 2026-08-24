clc;
close all;
clear all;
%%
addpath('/home/ayush/Desktop/EEG_viewing_codes/Callbacks')
input_file='sample_EEG_10min.mat';

%%%% read data
tmp=load(input_file);
data=tmp.data;data_=data;Fs=double(tmp.Fs);

%%%% denoise
data(isnan(data))=eps;
[B1,A1]=butter(3,[0.5,70]/(Fs/2));[B2,A2]=iirnotch(60/(Fs/2),60/(35*Fs/2));
data=filtfilt(B1,A1,data')';data=filtfilt(B2,A2,data')';data(isnan(data_))=NaN;

%%%% compute spectrograms
ww=1*Fs;num_seg=ceil(size(data,2)/ww)+1;stime=(0:1:(num_seg-1));params.movingwin=[4,1];params.tapers=[2,3];params.fpass=[.5,20];params.Fs=Fs;
[sdata,~,sfreq]=fcn_compute_spec(fcn_bipolar(data),params);
for j=1:size(sdata,1)
    s=sdata{j,2};
    sdata{j,2}=[s(:,1),s(:,1:end-1),repmat(s(:,end-1),1,num_seg-size(s,2))];
end
sdata=cell2mat(sdata(:,2));sdata(isnan(sdata))=eps;

%% plot figure
tt=300;event_t=Fs*tt;w=10;data=data(:,(event_t-w*Fs/2+1):(event_t+w*Fs/2));

f=figure('units','normalized','position',[0.0182,0,0.4818,0.9467],'color','w','MenuBar','none','ToolBar','none','HitTest','off');
ax_eeg=subplot('position',[.355,.03,.63,.945],'Parent',f);
ax_spec={subplot('position',[.040,.75,.28,.225]);subplot('position',[.040,.51,.28,.225]);subplot('position',[.040,.27,.28,.225]);subplot('position',[.040,.03,.28,.225])};
ax_marker={subplot('position',[.040,.975,.28,.010],'Parent',f);subplot('position',[.355,.975,.63,.010],'Parent',f)};
t0=datetime('2000-01-01 00:00:00','inputformat','yyyy-MM-dd HH:mm:ss');tc=t0+seconds(tt);time_stamps=datestr(t0+seconds(stime(1:120:601)),'HH:MM:ss');xticks=stime(1:120:601);nn=size(sdata,1)/4;reg_tag={'LL','RL','LP','RP'};colormap('jet');
for i=1:size(ax_spec,1)
    set(f,'CurrentAxes',ax_spec{i});cla(ax_spec{i})
    hold(ax_spec{i},'on')
        spec=pow2db(sdata((i-1)*nn+1:i*nn,:)+eps);
        imagesc(ax_spec{i},stime,sfreq,spec,[-10,25]);axis(ax_spec{i},'xy');plot(ax_spec{i},[stime(tt),stime(tt)],[sfreq(1),sfreq(end)+.1],'k--','linewidth',1.2);
        yticks=get(ax_spec{i},'ytick');yticklabels=get(ax_spec{i},'yticklabel');yticklabels{end}=reg_tag{i};ylabel(ax_spec{i},'Freq (Hz)');
        set(ax_spec{i},'ylim',[sfreq(1),sfreq(end)+.1],'xlim',[stime(1),stime(end)],'yticklabel',yticklabels,'ytick',yticks,'xtick',[],'box','on')        
        if i==4
            set(ax_spec{i},'xtick',xticks,'xticklabel',time_stamps,'fontsize',9);  
        end                   
    hold(ax_spec{i},'off')
end     

set(f,'CurrentAxes',ax_marker{1});cla(ax_marker{1});
hold(ax_marker{1},'on')
    plot(ax_marker{1},stime(tt),1,'gv','markerfacecolor','g');
    text(ax_marker{1},stime(tt)+10,1,['\color{red}',datestr(tc,'HH:MM:ss')],'horizontalalignment','left','fontsize',9);xlim([stime(1),stime(end)]);axis off;
hold(ax_marker{1},'off')

set(f,'CurrentAxes',ax_marker{2});cla(ax_marker{2});
hold(ax_marker{2},'on')
    plot(ax_marker{2},0,1,'gv','markerfacecolor','g');axis off;
hold(ax_marker{2},'off')

z_scale=1/150;
eeg=data(1:19,:);gap=NaN(1,size(eeg,2));
if size(data,1)<20
    ekg=gap;
else
    ekg=-data(20,:);ekg=(ekg-mean(ekg))/(std(ekg)+eps);
end
seg=fcn_bipolar(eeg);seg(seg>300)=300;seg(seg<-300)=-300;seg_disp=[seg(1:4,:);gap;seg(5:8,:);gap;seg(9:12,:);gap;seg(13:16,:);gap;seg(17:18,:);gap;ekg];
channels_disp={'Fp1-F7','F7-T3','T3-T5','T5-O1','','Fp2-F8','F8-T4','T4-T6','T6-O2','','Fp1-F3','F3-C3','C3-P3','P3-O1','','Fp2-F4','F4-C4','C4-P4','P4-O2','','Fz-Cz','Cz-Pz','','EKG'};
tto=1;tt1=size(data,2);tt=tto:tt1;M=size(seg_disp,1);dc_offset=repmat(flipud((1:M)'),1,size(seg_disp,2));timeStamps=datestr(tc-seconds(5)+seconds(0:1:10),'HH:MM:ss');

set(f,'CurrentAxes',ax_eeg);cla(ax_eeg);
hold(ax_eeg,'on')
    for k=1:11
        ta=Fs*(k-1);line([ta ta],[0 M+3],'linestyle','--','color',[.7,.7,.7])                
        text(ax_eeg,ta,-0.265,timeStamps(k,:),'horizontalalignment','center','fontsize',9)        
    end
    for k=1:length(channels_disp)
        text(ax_eeg,-.1*Fs,M-k+1,channels_disp{k},'horizontalalignment','right')
    end
    plot(ax_eeg,tt,z_scale*seg_disp(1:end-1,:)+dc_offset(1:end-1,:),'k','linewidth',1.0);plot(ax_eeg,tt,1/5*seg_disp(end,:)+dc_offset(end,:),'r','linewidth',1.0);box on;
    set(ax_eeg,'box','off','ylim',[0 M+1],'xlim',[tto-1,tt1+1]);       
    dt=tt1-tto+1;a=round(dt*4/5);xa1=tto+[a a+Fs-1];ya1=[3 3]-.7;xa2=tto+[a a];ya2=ya1+[0 100*z_scale];
    text(xa1(1)-.6*a/10,mean(ya2),'100\muV','Color','b','FontSize',11);text(mean(xa1),2,'1 sec','Color','b','FontSize',11,'horizontalalignment','center');line(xa1,ya1,'LineWidth',2,'Color','b');line(xa2,ya2,'LineWidth',2,'Color','b');axis off;
hold(ax_eeg,'off')
