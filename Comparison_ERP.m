clear all; clc; close all
%定义被试编号
Subj = [1:10]; 
%定义条件的marker
Cond = {'L1','L2','L3','L4'};

%% compute averaged data   在特点通道下两个条件是否存在差异
%对于每一个被试
for i = 1:length(Subj)
    %拼接被试文件名
    setname = strcat(num2str(i),'_LH.set'); 
    %定义数据所在路径
    setpath = 'D:\ERPS\25EEG_day2\Example_data';
    %导入数据
    EEG = pop_loadset('filename',setname,'filepath',setpath); 
    %检查数据结构体
    EEG = eeg_checkset( EEG );
    %对于每个条件
    for j = 1:length(Cond)
        %以分段的方式提取第j个条件的所有数据
        EEG_new = pop_epoch( EEG, Cond(j), [-1  2], 'newname', 'Merged datasets pruned with ICA', 'epochinfo', 'yes'); 
        %EEG_new ch*times*epch？
        EEG_new = eeg_checkset( EEG_new );
        %基线校正
        EEG_new = pop_rmbase( EEG_new, [-1000     0]); 
        EEG_new = eeg_checkset( EEG_new );
        %将第i个被试 第j个条件 所有通道 所有时间点的数据进行汇总
        %EEG_avg sub*cond*ch*times
        EEG_avg(i,j,:,:)=squeeze(mean(EEG_new.data,3));  %% subj*cond*channel*timepoints
    end 
end

%%  point-by-point paried t-test  across multiple time points 沿着时间点做统计，在特定电极下，到底在什么时间下不同条件不同组别下存在差异
%EEG_avg sub*cond*times
%假设我们感兴趣的是第13号电极
%提取所有被试 所有条件 感兴趣的电极上 所有时间点的数据，并挤压多余维度
data_test = squeeze(EEG_avg(:,:,13,:)); %% select the data at Cz, data_test: subj*cond*time
%data_test sub*cond*times 10*4*300
%对于每个时间点
for i = 1:size(data_test,3)
    %提取被试L3条件，第i个时间点的数据，并挤压多余维度
    data_1 = squeeze(data_test(:,3,i)); %% select condition L3 for each time point
     %提取被试L4条件，第i个时间点的数据，并挤压多余维度
    data_2 = squeeze(data_test(:,4,i)); %% select condition L4 for each time point
    %由于我们是检验两个条件间的差异，所以要用配对样本t检验 ttset（=号后）
    %如果做的是两组人之间的差异 则需要做的是独立样本T检验 ttest2（=号后）
    %ttest 可以有四个输出[h,p,ci,stats];
    %h是假设,p为检验的p值，ci置信区间，stats为结构体变量，其中stats.tstat 储存统计量 即T值
    %stats.df 储存自由度信息 stats.sd 储存标准差
    %ttest如果只有一个输出变量 第一位输出的是h
    %h为0假设是否 被拒绝的情况， 1为拒绝0假设（显著） 0为不拒绝（不显著）
    [h p] = ttest(data_1,data_2); %% ttest comparison
    %建立变量P_ttest 储存每次统计量的P值
    P_ttest(i) = p; %% save the p value from ttest
end
figure; 
%data_test 3D sub*cond*times
%绘制2行1列的图
%2行第1列图中第一个图
%提取所有被试 第三个条件 所有时间点的数据，并沿着被试的维度进行平均
subplot(211); plot(EEG.times,squeeze(mean(data_test(:,3,:),1)),'b'); %% plot the average waveform for Condition L3
hold on; plot(EEG.times,squeeze(mean(data_test(:,4,:),1)),'r'); %% plot the average waveform for Condition L4
subplot(212); plot(EEG.times,P_ttest); ylim([0 0.05]); %%plot the p values from ttest
sig=find(P_ttest <0.05)%如果要找p值所在时间点则用此项

%% point-by-point paried t-test  across multiple channels （找成分，平均波幅的差异检验）
%定义感兴趣的时间范围---锁定成分
%找到感兴趣的时间范围内的采样点的位置
test_idx = find((EEG.times>=197)&(EEG.times<=217)); %% define the intervals
%EEG_avg 4D sub*cond*ch*times
%提取所有被试 所有条件 所有通道 感兴趣的范围内的数据
%沿着时间进行平均
%data_test sub*cond*ch
data_test = squeeze(mean(EEG_avg(:,:,:,test_idx),4)); %% select the data in [197 217]ms, subj*cond*channel
%对于每一个通道
for i = 1:size(data_test,3)
    %提取所有被试第三个条件第i个通道的数据 并挤压多余维度
    data_1 = squeeze(data_test(:,3,i)); %% select condition L3 for each channel
   %提取所有被试第四个条件第i个通道的数据 并挤压多余维度
    data_2 = squeeze(data_test(:,4,i)); %% select condition L4 for each channel
    %配对样本T检验（=号后面函数里的ttest、ttest2才代表统计方法）
    [h,p,ci,stats] = ttest(data_1,data_2); %% ttest comparison
    %汇总P值(=前面ttest是自己命名的，与统计无关）
    P_ttest2(i) = p; %% save the p value from ttest
    %汇总T值（包含自由度方差等）
    T_ttest2(i) = stats.tstat; 
end

figure; 
subplot(141); 
%data_test sub*con*ch
%提取所有被试 第三个条件 所有通道的数据 并沿着被试做平均
topoplot(squeeze(mean(data_test(:,3,:),1)),EEG.chanlocs,'maplimits',[-20 20]); 
subplot(142); 
%提取所有被试 第4个条件 所有通道的数据 并沿着被试做平均
topoplot(squeeze(mean(data_test(:,4,:),1)),EEG.chanlocs,'maplimits',[-20 20]); 
subplot(143); 
topoplot(T_ttest2,EEG.chanlocs); 
subplot(144); 
topoplot(P_ttest2,EEG.chanlocs,'maplimits',[0 0.05]); 

%% point-by-point repeated measures of ANOVA across time points 一个因素的重复测量方差分析（2*2不能用）
data_test = squeeze(EEG_avg(:,:,13,:)); %% select the data at Cz, data_test: subj*cond*time
for i = 1:size(data_test,3)
    data_anova = squeeze(data_test(:,:,i)); %% select the data at time point i
    %off 表示不显示计算窗口，可改为on呈现出来
    [p, table] = anova_rm(data_anova,'off');  %% perform repeated measures ANOVA
    P_anova(i) = p(1); %% save the data from ANOVA
end

mean_data = squeeze(mean(data_test,1)); %% dimension: cond*time
figure; 
subplot(211);plot(EEG.times, mean_data,'linewidth', 1.5); %% waveform for different condition 
set(gca,'YDir','reverse');
axis([-500 1000 -35 25]);
subplot(212);plot(EEG.times,P_anova); axis([-500 1000 0 0.05]); %% plot the p values from ANOVA

%% point-by-point repeated measures of ANOVA across channels

test_idx = find((EEG.times>=197)&(EEG.times<=217)); %% define the intervals
data_test = squeeze(mean(EEG_avg(:,:,:,test_idx),4)); %% select the data in [197 217]ms, subj*cond*channel
for i = 1:size(data_test,3)
    data_anova = squeeze(data_test(:,:,i)); %% select the data at channel i
    [p, table] = anova_rm(data_anova,'off');  %% perform repeated measures ANOVA
    P_anova2(i) = p(1); %% save the data from ANOVA
    F_anova2(i) = table{2,5};
end
figure; 
for i = 1:4
    subplot(1,5,i); 
    topoplot(squeeze(mean(data_test(:,i,:),1)),EEG.chanlocs,'maplimits',[-20 20]); 
end
% subplot(1,5,5); topoplot( P_anova2,EEG.chanlocs,'maplimits',[0 0.05]); 
subplot(1,5,5); topoplot( F_anova2,EEG.chanlocs); 



