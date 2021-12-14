
%erp make picture by TIAN
%
% a = {'FCZ','FC1','FC2'}         %ѡ��Ҫ���缫��
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
dianji = find(chanloc==1)        %��ȡ�缫����

plot(EEG.times,mean(mean(ALLEEG(1).data(dianji,:,:)),3),'r','linewidth',1)   %������һ���������缫�㣬��������ƽ��������NANMEAN
hold on
plot(EEG.times,mean(mean(ALLEEG(2).data(dianji,:,:)),3),'k','linewidth',1)   %��������
hold on
plot(EEG.times,mean(mean(ALLEEG(3).data(dianji,:,:)),3),'g','linewidth',1)   %��������
hold on
plot(EEG.times,mean(mean(ALLEEG(4).data(dianji,:,:)),3),'m','linewidth',1)   %��������
hold on
plot(EEG.times,mean(mean(ALLEEG(5).data(dianji,:,:)),3),'y','linewidth',1)   %��������
hold on
plot(EEG.times,mean(mean(ALLEEG(6).data(dianji,:,:)),3),'c','linewidth',1)   %��������
hold on
plot([-1000,2000],[0,0],'k')                %������
hold on
plot([0,0],[-10 10],'k')                   %������
xlim([-500 800])                           %ȷ����Χ
ylim([-10,10])
set(gca,'YDir','reverse');                 %�����᷽��
% set(gca,"xticklabel",{-1000:100:2000})  %tick���ڱ��
% x = [500,500,700,700];                     %�������
% y = [-10,10,10,-10];
% legend('0��~60��','70��~180��','FontSize',12)
% annotation('textarrow',[0.43 0.53],[0.37 0.27],'String','p300','fontsize',22,'color','b');
% annotation('textarrow',[0.4 0.48],[0.55 0.45],'String','N2','fontsize',22,'color','k');
% annotation('textarrow',[0.6,0.7],[0.7 0.6],'String','spw','fontsize',22,'color','g');
% title(strcat(a{1},',',a{2}),'fontsize',20)
% % hold on;h=fill(x,y,[.5 .5 .5],'facealpha',0.4,'edgealpha',0);
legend('1back_��','1back_��','1back_��','2back_��','2back_��','2back_��','FontSize',12)
annotation('textarrow',[0.28 0.38],[0.76 0.66],'String','N1','fontsize',22,'color','g');    %��ǳɷ�
annotation('textarrow',[0.57 0.67],[0.3 0.],'String','spw','fontsize',22,'color','r');    %��ǳɷ�
annotation('textarrow',[0.32 0.42],[0.2 0.3],'String','vpp','fontsize',22,'color','b');
title('ǰ�������������ıȽ�','fontsize',20)   %�缫��

times = [230,280]        %ѡ��Ҫ������ʱ�䷶Χ
start = (abs(EEG.times(1,1))+times(1,1)+2)/2
ends  = (abs(EEG.times(1,1))+times(1,2)+2)/2

%ͳ��
F = 0
for p = 1:2
    for m = 1:20     %��������
        stat(m,p) = squeeze(mean(mean(ALLEEG(F+m).data(dianji,start:ends),1)))
    end
    F = F+20
end


%�����ֵ
F = 0
for p = 1:2
    for m = 1:3     %��������
        edge{m,p} = mean(squeeze(ALLEEG(F+m).data(dianji,start:ends),1))
    end
    F = F+3
end
edge_index = find(edge{1, 1}==max(edge{1, 1}))     %��ֵ����Ҫ�Ľ�
edges = ALLEEG(1).times(start+edge_index)

%������������������������򲻱��
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

