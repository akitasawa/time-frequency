% clear all; clc;
% Sub={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'};
%  Cond = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{20},{30},{40},{50},{60},{70},{80},{90}}
% save('E:\数据\重建\预处理完的数据\Cond.mat','Cond')
clear all;
id = { '002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017',...
    '018'	,'019',	'020'	,'021',	'022',	'023'	,'024'	,'025',	'026',	'027'	,'028'	,'029'	,'030',	'031',	'032',...
    '033'	,'034',	'040'	,'041'	,'042',	'043'	,'044'	,'045'	,'046',	'047',	'048'	,'049',...
    '050',	'051'	,'052'	,'053'};
Cond= { {'S 11'}  {'S 22'}  {'S 33'}  {'S 36'} }
%%  time-frequency analysis for multiple conditions
% 对于所有被试
for i=1:length(id)
    tic
    setname=([ id{i} '.set']);
    setpath='H:\CSPCE\去伪迹后';
    EEG= pop_loadset('filename',setname,'filepath',setpath);
    EEG= eeg_checkset( EEG );
    % 对于所有条件
    for j=1:length(Cond)
        EEG_new = pop_epoch( EEG,Cond{1, j}, [-0.8 1.2], 'newname', 'Merged datasets pruned with ICA   epochs epochs', 'epochinfo', 'yes');
        EEG_new = eeg_checkset( EEG_new );
%         EEG_new = pop_rmbase( EEG_new, [-200    0]);
%         EEG_new = eeg_checkset( EEG_new );
        % 对于所有通道
        for nchan=1:size(EEG_new.data,1)
            x = squeeze(EEG_new.data(nchan,:,:));
            xtimes=EEG_new.times/1000;
            t=EEG_new.times/1000;
            f=1:1:30;
            Fs = EEG.srate;
            %             algm_opt = 'fft';
            winsize = 0.400;
            [S, P, F, U] = sub_stft(x, xtimes, t, f, Fs, winsize);
            % 被试 * 条件 * 通道 * 频率 * 时间点
            P_DATA(i,j,nchan,:,:)=squeeze(mean(P,3)); %%P_data (without baseline correciton):  subj*cond*chan*f*time
        end
    end
    toc
    fprintf('\n############ completed for subject %s ###########\n\n\n\n',id{i} )
    waitbar(i/length(id))
    
end
save('H:\CSPCE\wavelet.mat','P_DATA','xtimes','f','P_DB')
%% baseline correction
% 对于所有被试、条件、通道、频率，都做基线校正
t_pre_idx=find((xtimes>=-0.2)&(xtimes<=0));
for i=1:size(P_DATA,1)%所有被试
    for j=1:size(P_DATA,2)%所有条件
        for ii=1:size(P_DATA,3)%所有通道
            for jj=1:size(P_DATA,4)%所有频率
                temp_data=squeeze(P_DATA(i,j,ii,jj,:));
%                                 P_CC(i,j,ii,jj,:)=temp_data-mean(temp_data(t_pre_idx))  ;    %absolute法
                P_DB(i,j,ii,jj,:) = 10*log10( bsxfun(@rdivide, temp_data,mean(temp_data(t_pre_idx))));  %DB法
            end
        end
    end
end

%% ttest for each time-frequency point

% P_data: 被试 * 条件 * 通道 * 频率 * 时间点
% 提取所有被试、所有条件、13号通道（空间ROI、Cz）、所有频率、所有时间的power
%: data_test： 被试 *  条件 * 频率 *  时间点
data_test=squeeze(P_BC(:,:,13,:,:)); %% select the data at Cz, data_test: subj*cond*frequency*time
%对于每一个频率点
for i=1:size(data_test,3)
    %对于每一个时间点
    for j=1:size(data_test,4)
        % 挑选第一种条件的数据
        data_1=double(squeeze(data_test(:,1,i,j))); %% select condition L3 for each time-frequency point
        %挑选第二种条件的数据
        data_2=double(squeeze(data_test(:,2,i,j))); %% select condition L4 for each time-frequency point
        % 配对T检验
        [h,p,ci,stats]=ttest(data_1,data_2); %% ttest comparison
        % 存储第i个频率点、第j个时间点的p值
        P_ttest(i,j)=p; %% save the p value from ttest
        %  存储T值
        T_ttest(i,j)=stats.tstat; %% save the t value from ttest
    end
end


figure;
subplot(511); imagesc(t,f,squeeze(mean(P_BC(:,1,19,:,:),1))); title('MC_M_I');axis xy; colorbar();
subplot(512); imagesc(t,f,squeeze(mean(P_BC(:,2,19,:,:),1))); title('MC_M_C');axis xy; colorbar();
subplot(513); imagesc(t,f, T_ttest); title('T values'); axis xy; colorbar();
subplot(514); imagesc(t,f, P_ttest); title('P values'); axis xy; colorbar();
subplot(515); imagesc(t,f, P_ttest); title('P values'); axis xy; colorbar();caxis([0 0.05]);

%%

%对于每一个时频点
for i=1:size(data_test,3)
    %对于每个时间点
    for j=1:size(data_test,4)
        % 挑选所有条件的数据
        data_anova=squeeze(data_test(:,:,i,j)); %% select the data at time-frequency point%这里是每个被试每个条件的
        %重复测量方差分析
        [p, table] = anova_rm(data_anova,'off');  %% perform repeated measures ANOVA
        %存储各时频点的p值
        P_anova(i,j)=p(1); %% save the data from ANOVA
        % 存储F值
        F_anova(i,j)=table{2,5}; %% F value from ANOVA
    end
end

%%   原始图
figure
% contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,2,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3)),40,'linecolor','none')
contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,4,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3))-squeeze(mean(mean(mean(P_DB(:,3,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3)),40,'linecolor','none')
% contourf(xtimes, f,squeeze(mean(mean(mean(P_DATA(:,1:2:4,[9;10;11;18;19;20],:,:),1),2),3))-squeeze(mean(mean(mean(P_DATA(:,2:2:4,[9;10;11;18;19;20],:,:),1),2),3)),40,'linecolor','none')
% % contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,1:12,[7;8;9;10;11;12;16;17;19;20;21;22],:,:),1),2),3)),40,'linecolor','none')
set(gca,'xlim',[-0.4 1], 'clim',[-1 1])
title('前额叶:大多数不一致的冲突','fontsize',20)
xlabel('时间(S)','fontsize',15)
ylabel('频率(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','dB','fontsize',15);
colormap(jet)    % 选择配色方案
 rectangle('Position',[0.55 5 0.2 3],'Linestyle','--','EdgeColor','k')
rectangle('Position',[0.9 9 0.2 4],'Linestyle','--','EdgeColor','k')
% saveas(gcf,'C:\Users\TIAN\Desktop\数据\顶枕叶图\大多数不一致下不一致减去一致.jpg')

for i =1:2
subplot(1,2,i)
end

DIFFSS = squeeze(mean(mean(P_DB(:,1,[45:47,53:55,58:60],:,:),1),3)) - squeeze(mean(mean(P_DB(:,2,[45:47,53:55,58:60],:,:),1),3))

grand = {'MC_M_I ','MC_M_C','MC_F_I','MC_F_C','MC_S_I',...
    'MC_S_C','MI_M_I','MI_M_C','MI_F_I','MI_F_C','MI_S_I',...
    'MI_S_C'}

for n=1:12
    figure
    contourf(t, f,squeeze(mean(mean(P_DB(:,n,[42:51,53:55,57:60],:,:),1),3)),40,'linecolor','none')
%     contourf(t, f,squeeze(mean(mean(P_DB(:,n,[42:51,53:60],:,:),1),3)),40,'linecolor','none')
    set(gca,'xlim',[-0.4 1.1], 'clim',[-5 5])
%     title(grand(n))
    xlabel('Time (s)')
    ylabel('freq (Hz)')
%     colorbar
    colormap(jet)    % 选择配色方案
   saveas(gcf,['C:\Users\TIAN\Desktop\论文写作\顶枕区\' grand{n} '.jpg'])
end



for m=1:6
figure
k = squeeze(mean(mean(diff(m,:,8:13,651:751),3),4))
topoplot(k, channel);colorbar();
k=[]
set(gca, 'clim',[-3 3])
end

%%

%计算差异
x = 1
for m = 1:2:4
    
    DIFF= squeeze(mean(P_DB(:,m,:,:,:),1)) - squeeze(mean(P_DB(:,m+1,:,:,:),1));
    diff(x,:,:,:) = DIFF;
    
    x = x+1
end

save(['E:\数据\重建\eeglab\stft.mat'],'channel','diff','f','t','P_BC','P_DB','P_data')

%差异图（批量）
diff_name = {'MC_M_diff ','MC_F_diff','MC_S_diff','MI_M_diff','MI_F_diff','MI_S_diff'}
for d=1:6
    figure
    contourf(t, f,squeeze(mean(diff(d,[42:51,53:55,57:60],:,:),2)),40,'linecolor','none')
%     contourf(t, f,squeeze(mean(diff(d,[7:12,16:22],:,:),2)),40,'linecolor','none')
    set(gca,'xlim',[-0.4 1.1], 'clim',[-3 3])
%     title(diff_name(d))
    xlabel('Time (s)')
    ylabel('freq (Hz)')
%     colorbar
    colormap(jet)    % 选择配色方案
   saveas(gcf,['C:\Users\TIAN\Desktop\论文写作\前额区\' diff_name{d} '.jpg'])
end

%% 地形图
load('E:\数据\重建\channel.mat')
time = [0.6 0.8]
fre  =[9,13]
time_ =  dsearchn(xtimes',time')
fre_  =  dsearchn(f',fre')
topoplot(squeeze(mean(mean(mean(mean(P_DB(:,1:2:4,:,fre_(1):fre_(2),time_(1):time_(2)),1),4),5),2))-squeeze(mean(mean(mean(mean(P_DB(:,2:2:4,:,fre_(1):fre_(2),time_(1):time_(2)),1),4),5),2)), channel,  'electrodes'  ,'labels');
% topoplot(squeeze(mean(mean(mean(mean(P_DB(:,1:4,:,fre_(1):fre_(2),time_(1):time_(2)),1),4),5),2)), channel,  'electrodes'  ,'labels');
colorbar;
set(gca, 'clim',[-1.5 1.5])
h= colorbar
set(get(h,'Title'),'string','dB','fontsize',15);
title('θ波:不一致减去一致地形图','fontsize',20)
%  saveas(gcf,['C:\Users\TIAN\Desktop\11.jpg'])
%差异地形图
diff_name = {'MC_M_地形图 ','MC_F_地形图','MC_S_地形图','MI_M_地形图','MI_F_地形图','MI_S_地形图'}
for i = 1:6
figure
time = [0.5 ,0.7]
fre  =[4,7]
time_ =  dsearchn(xtimes',time')
fre_  =  dsearchn(f',fre')
topoplot(squeeze(mean(mean(diff(2,:,fre_(1):fre_(2),time_(1):time_(2)),3),4)), channel);
colorbar;
set(gca, 'clim',[-1 1])
 saveas(gcf,['C:\Users\TIAN\Desktop\论文写作\前额区\' diff_name{i} '_θ.jpg'])
end

%% 统计
freq  = ALL_MC_F_C{1, 1}.freq;
chan  = ALL_MC_F_C{1, 1}.label  ;
time = ALL_MC_F_C{1, 1}.time
% define time window
% timewin      = [0.7 0.9];
timewin      = [0.4 0.7];    %θ
timewin_idx  = dsearchn(time', timewin');
% define frequency window
freqwin      = [5 10];  % theta band
% freqwin      = [11 21];  % alpha band
freqwin_idx  = dsearchn(freq', freqwin');
% define ROI (channels)
% chan2use = {'Fz', 'F1', 'F2', 'FCZ', 'FC1','FC2'};
chan2use = {'P1', 'PZ', 'P2', 'POZ', 'PO3','PO4','O1','OZ','O2'};
chanloc = zeros(length(ALL_MC_F_C{1, 1}.label),1)
% for i=1:length(chan2use)       % find the index of channels to use
%     ch = strcmpi(chan2use(i), chan);
%     chan_idx(i) = find(ch);
% end %取时间和频率段
for i = 1:length(chan2use)
    for n = 1: 60
        chan = strcmpi(chan2use(i),ALL_MC_F_C{1, 1}.label(n))
        if chan == 1
            chanloc(n,1) = chan
        end
        
    end
end
chan_idx = find(chanloc==1)        %获取电极点编号



id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'}  % 被试编号
Ns = length(id);
power = zeros(Ns,12); % initialize variable， 此例中，有Ns个被试，2个条件
for subi=1:Ns
    pow1 = ALL_MC_F_C{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow2 = ALL_MC_F_I{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow3 = ALL_MC_M_C{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow4 =ALL_MC_M_I{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow5 = ALL_MC_S_C{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow6 = ALL_MC_S_I{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow7 = ALL_MI_F_C{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow8 = ALL_MI_F_I{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow9 = ALL_MI_S_I{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow10 = ALL_MI_M_C{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow11 = ALL_MI_M_I{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow12 =ALL_MI_S_C{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    
    
    power(subi,1) = squeeze(mean(mean(mean( pow1  ))));  % 提取第一个条件的数据
    power(subi,2) = squeeze(mean(mean(mean( pow2  ))));  % 提取第二个条件的数据
    power(subi,3) = squeeze(mean(mean(mean( pow3  ))));  % 提取第一个条件的数据
    power(subi,4) = squeeze(mean(mean(mean( pow4  ))));  % 提取第二个条件的数据
    power(subi,5) = squeeze(mean(mean(mean( pow5  ))));  % 提取第一个条件的数据
    power(subi,6) = squeeze(mean(mean(mean( pow6  ))));  % 提取第二个条件的数据
    power(subi,7) = squeeze(mean(mean(mean( pow7  ))));  % 提取第一个条件的数据
    power(subi,8) = squeeze(mean(mean(mean( pow8  ))));  % 提取第二个条件的数据
    power(subi,9) = squeeze(mean(mean(mean( pow9  ))));  % 提取第一个条件的数据
    power(subi,10) = squeeze(mean(mean(mean( pow10  ))));  % 提取第二个条件的数据
    power(subi,11) = squeeze(mean(mean(mean( pow11  ))));  % 提取第一个条件的数据
    power(subi,12) = squeeze(mean(mean(mean( pow12  ))));  % 提取第二个条件的数据
end
TF_condition = {'TF_MC_M_I','TF_MC_M_C','TF_MC_F_I','TF_MC_F_C','TF_MC_S_I','TF_MC_S_C','TF_MI_M_I','TF_MI_M_C',...
    'TF_MI_F_I','TF_MI_F_C','TF_MI_S_I','TF_MI_S_C'};
title = (['顶枕叶' '，' '时间' num2str(timewin(1)) '——' num2str(timewin(2))  's' '，' ...
    '频率' num2str(freqwin(1)) '——' num2str(freqwin(2)) 'hz'])
xlswrite('C:\Users\TIAN\Desktop\顶枕叶_α.xlsx',title,1,"a1:a26");
xlswrite('C:\Users\TIAN\Desktop\顶枕叶_α.xlsx',TF_condition,1,"a2");    %将matlab数据写入excel，S为写入数据，1为sheet，c1为单元格位置）
xlswrite('C:\Users\TIAN\Desktop\顶枕叶_α.xlsx',power,1,"a3");

dlmwrite('F:\power.txt',power,'\t')  % 保存到txt文件中(用excel打开)，用于进一步分析


%% stats across channels

%感兴趣的时频区域
t_ROI = [0.4 0.6]; f_ROI=[10 15];
t_idx = find((t> t_ROI(1)) & (t < t_ROI(2)));
f_idx = find((f> f_ROI(1)) & (f < f_ROI(2)));
%提取感兴趣时频区域的能量（基线校正后）
data_test_ch = squeeze(mean(mean(P_BC(:,:,:,f_idx, t_idx),4),5));

% t test (L3 vs L4)
for i=1:EEG.nbchan   %对于每一个通道
    data1 = double(squeeze(data_test_ch(:,1,i)));  %提取所有被试第三个条件在第i个电极的数据
    data2 = double(squeeze(data_test_ch(:,2,i)));  %提取所有被试第四个条件在第i个电极的数据
    [~,p,~,stat] = ttest(data1,data2);  %配对T
    P_ttest_ch(i) = p;  %存储P值、T值
    T_ttest_ch(i) = stat.tstat;
end

%绘制T值的地形图
figure;
subplot(121);topoplot(T_ttest_ch, EEG.chanlocs); title('T values'); colorbar();
subplot(122);topoplot(P_ttest_ch, EEG.chanlocs); title('P values'); colorbar();


%% 四种条件做单因素重复测量的方差分析
% F test ( 4 conditions)
for d = 1:20
    for i=1:60
        data_temp = squeeze(ppp(d,i,:,:));
        [p, table] = anova_rm(data_temp, 'off');
        P_anova_ch(i) = p(1);
        F_anova_ch(i) = table{2,5};
    end
    pp(d,:)  =P_anova_ch;
    
    ff(d,:)  = F_anova_ch;
end
%绘制F值的地形图
figure;
subplot(121);topoplot(F_anova_ch, EEG.chanlocs); title('F values'); colorbar();
subplot(122);topoplot(P_anova_ch, EEG.chanlocs); title('P values'); colorbar(); caxis([0 0.05]);
%% fdr correction to account for multiple comparisons
%对多重比较做矫正，避免假阳性的问题
%对配对T检验的结果 做FDR校正（p_fdr1 ： fdr显著性阈值； p_masked，表示p值是否通过校正
[p_fdr1, p_masked] = fdr(P_ttest, 0.05); %% fdr correction for p values from ttest %0.05是要求fdr矫正将犯假阳性错误的概率控制在0.05以下
% 绘制p值图（时间横轴、频率为纵轴）； 只显示通过校正的结果
figure; imagesc(t,f,P_ttest); axis xy; caxis([0 0.05]);
%p_corrected=mafdr(ps, 'BHFDR', 1);%-->madfdr是matlab的fdr的矫正，返回的是矫正后的p值

% 对方差分析的结果做FDR 校正
[p_fdr2, p_masked] = fdr(P_anova, 0.05);%% fdr correction for p values from ANOVA
figure; imagesc(t,f,P_anova); axis xy; caxis([0 p_fdr2]); %如果没有数值能通过fdr矫正，则会显示原始的p值

%% correlation with behavioral measures

Rating=[1:10];
data_test=squeeze(mean(P_data(:,:,13,:,:),2)); %% select the data at Cz, data_test: subj*frequency*time
for i=1:size(data_test,2)
    for j=1:size(data_test,3)
        data_anova=squeeze(data_test(:,i,j)); %% select the data at time-frequency point (i,j), Subj*1
        [r p]=corrcoef(data_anova,Rating); %% correlation (pearson)
        R_corr(i,j)=r(1,2);  %% save the r values
        P_corr(i,j)=p(1,2); %% save the p values
    end
end

[p_fdr3, p_masked] = fdr(P_corr, 0.05);%% fdr correction for p values from ANOVA
figure; imagesc(t,f,P_corr); axis xy; caxis([0 p_fdr3]);

%%  时频统计
load('E:\数据\重建\channel.mat')
timewin      = [0.9 1.1];    %α
% timewin      = [0.55 0.75];    %θ
timewin_idx  = dsearchn(xtimes', timewin');
freqwin      = [9 13];  % α band
% freqwin      = [5 8];  % theta band
freqwin_idx  = dsearchn(f', freqwin');
% chan2use = { 'F5';'F3';'F1';'FZ';'F2';'F4';'FCZ';'FC2';'FC4';'FC6';'FC5';'FC3'};
chan2use = { 'F1';'FZ';'F2';'FCZ';'FC2';'FC1'};
% chan2use = {'p1','p3','Pz','p2','p4','po3','po7','poz','po4','po8','o1','oz','o2'};
% chan2use ={'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2'}
chanloc = zeros(60,1)
for i = 1:length(chan2use)
    for n = 1: 60
        chan = strcmpi(chan2use(i),channel(n).labels)
        if chan == 1
            chanloc(n,1) = chan
        end
        
    end
end
chan_idx = find(chanloc==1)        %获取电极点编号
id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'}  % 被试编号
for cond = 1 :12
    for s = 1:length(id)
        power(s,cond) = squeeze(mean(mean(mean(P_DB(s,cond,chan_idx,freqwin_idx(1):freqwin_idx(2),timewin_idx(1):timewin_idx(2)),3),4),5))
        
              
    end
end





