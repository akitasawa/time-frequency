
%erp make picture by TIAN
%
% a = {'FCZ','FC1','FC2'}         %选中要画电极点
a = {'P5','P6'}
subject = [1,2,3,4,5,6]
cond = [1,2]
chanloc = zeros(length(EEG.chanlocs),1)

for i = 1:length(a)
    for n = 1: length(EEG.chanlocs)
        chan = strcmpi(a(i),EEG.chanlocs(n).labels)
        if chan == 1
            chanloc(n,1) = chan
        end
        
    end
end
dianji = find(chanloc==1)        %获取电极点编号

plot(EEG.times,mean(mean(ALLEEG(1).data(dianji,:,:)),3),'r','linewidth',1)   %画条件一，如果多个电极点，就用两次平均，或者NANMEAN
hold on
plot(EEG.times,mean(mean(ALLEEG(2).data(dianji,:,:)),3),'k','linewidth',1)   %画条件二
hold on
plot(EEG.times,mean(mean(ALLEEG(3).data(dianji,:,:)),3),'g','linewidth',1)   %画条件三
hold on
plot(EEG.times,mean(mean(ALLEEG(4).data(dianji,:,:)),3),'m','linewidth',1)   %画条件四
hold on
plot(EEG.times,mean(mean(ALLEEG(5).data(dianji,:,:)),3),'y','linewidth',1)   %画条件五
hold on
plot(EEG.times,mean(mean(ALLEEG(6).data(dianji,:,:)),3),'c','linewidth',1)   %画条件六
hold on
plot([-1000,2000],[0,0],'k')                %画横轴
hold on
plot([0,0],[-10 10],'k')                   %画数轴
xlim([-500 800])                           %确定范围
ylim([-10,10])
set(gca,'YDir','reverse');                 %坐标轴方向
% set(gca,"xticklabel",{-1000:100:2000})  %tick用于标记
% x = [500,500,700,700];                     %标记显著
% y = [-10,10,10,-10];
% legend('0°~60°','70°~180°','FontSize',12)
% annotation('textarrow',[0.43 0.53],[0.37 0.27],'String','p300','fontsize',22,'color','b');
% annotation('textarrow',[0.4 0.48],[0.55 0.45],'String','N2','fontsize',22,'color','k');
% annotation('textarrow',[0.6,0.7],[0.7 0.6],'String','spw','fontsize',22,'color','g');
% title(strcat(a{1},',',a{2}),'fontsize',20)
% % hold on;h=fill(x,y,[.5 .5 .5],'facealpha',0.4,'edgealpha',0);
legend('1back_中','1back_正','1back_负','2back_中','2back_正','2back_负','FontSize',12)
annotation('textarrow',[0.28 0.38],[0.76 0.66],'String','N1','fontsize',22,'color','g');    %标记成分
annotation('textarrow',[0.57 0.67],[0.3 0.],'String','spw','fontsize',22,'color','r');    %标记成分
annotation('textarrow',[0.32 0.42],[0.2 0.3],'String','vpp','fontsize',22,'color','b');
title('前额区所有条件的比较','fontsize',20)   %电极点

times = [230,280]        %选择要分析的时间范围
start = (abs(EEG.times(1,1))+times(1,1)+2)/2
ends  = (abs(EEG.times(1,1))+times(1,2)+2)/2

%统计
F = 0
for p = 1:2
    for m = 1:20     %几个被试
        stat(m,p) = squeeze(mean(mean(ALLEEG(F+m).data(dianji,start:ends),1)))
    end
    F = F+20
end


%计算峰值
F = 0
for p = 1:2
    for m = 1:3     %几个被试
        edge{m,p} = mean(squeeze(ALLEEG(F+m).data(dianji,start:ends),1))
    end
    F = F+3
end
edge_index = find(edge{1, 1}==max(edge{1, 1}))     %峰值，需要改进
edges = ALLEEG(1).times(start+edge_index)

%如果显著则标记显著，不显著则不标记
if  ttest(stat(:,1),stat(:,2),'Alpha',0.05) == 0
    plot(EEG.times,mean(ALLEEG(1).data(chanloc,:,:),3));
    hold on
    plot(EEG.times,mean(ALLEEG(3).data(chanloc,:,:),3));
    hold on
    plot([-1000,800],[0,0],'k');
    hold on
    plot([0,0],[-10 10],'k');
    xlim([-200,800]);
    ylim([-10,10]);
    x = [times(1,1),times(1,1),times(1,2),times(1,2)];
    y = [-10,10,10,-10];
    hold on;
    fill(x,y,[.5 .5 .5],'facealpha',0.3,'edgealpha',0);
else
    plot(EEG.times,mean(ALLEEG(1).data(chanloc,:,:),3))
    hold on
    plot(EEG.times,mean(ALLEEG(3).data(chanloc,:,:),3))
    hold on
    plot([-1000,800],[0,0],'k')
    hold on
    plot([0,0],[-10 10],'k')
    xlim([-200,800])
    ylim([-10,10])
    
end

