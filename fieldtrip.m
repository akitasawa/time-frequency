cd 'E:\数据\重建\全部数据'
close all
clear
% id = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'};  % 被试编号
id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'};  % 被试编号

Ns = length(id);
for subi=1:Ns
    load(['MI_S_I' id{subi} '_CleanData.mat']);     %赋值后读取数据
    
    %     %%%%%%%%%%%%%%%%%%% time frequency analysis %%%%%%%%%%%%%%%%
    %     %%% use hanning and dpss taper
    % hanning taper method, fixed window length，适用于低频段 < 30 Hz
    cfg              = [];            %清空cfg,防止其它的数值进行干扰
    cfg.output       = 'pow';         %输出类型为功率谱
    cfg.pad          = 'nextpow2';        % 让fft计算更快
    cfg.channel      = 'all';         %选择所有电极点
    cfg.method       = 'mtmconvol';
    cfg.taper        = 'hanning';       % one hanning taper
    cfg.foi          = 2:1:40;           %感兴趣的频段,以2hz为开始,30hz为终点,以2hz为步长
    cfg.t_ftimwin    = 0.4 .* ones(size(cfg.foi));   % fixed length of taper (time window) = 0.5 sec      窗口的长度
    %     cfg.t_ftimwin    = 5./cfg.foi;                 % 6 cycles per time window
    cfg.toi          = -0.5:0.02:1.2;  % time window "slides" from -0.2 to 0.5 sec in steps of 0.02 sec (20 ms)    感兴趣的时间点
    TF_MI_S_I = ft_freqanalysis(cfg, TF_MI_S_I);    % for A condition
    %     power_B  = ft_freqanalysis(cfg, condB);    % for B condition
    %
    % % % % %     %  multi-taper method，适用于高频段 > 30 Hz
    %     cfg = [];
    %     cfg.output       = 'pow';
    %     cfg.pad          = 'nextpow2';        % 让fft计算更快
    %     cfg.channel      = 'all';
    %     cfg.method       = 'mtmconvol';
    %     cfg.taper        = 'dpss';          % multitaper
    %     cfg.foi          = 2:1:40;
    %     cfg.t_ftimwin    = 0.4 * ones(size(cfg.foi));   % fixed length of taper (time window) = 0.5 sec
    %     cfg.tapsmofrq    = 5 * ones(size(cfg.foi));     % width of frequency smoothing
    % %     cfg.t_ftimwin     = 5./cfg.foi;             % 6 cycles per time window
    % %     cfg.tapsmofrq     = 0.5 .*cfg.foi;           % width of frequency smoothing
    %     % freq smoothing is decided by the number of tapers, which = 2* cfg.t_ftimwin * cfg.tapsmofrq -1
    %     cfg.toi          = -0.5:0.02:1.2;
    % % %     cfg.keeptrials   = 'yes';
    %     TF_MC_M_C  = ft_freqanalysis(cfg, TF_MC_M_C);   % for A condition
    % %     power_B       = ft_freqanalysis(cfg, MC_F_I);   % for B condition
    
    %     % wavelet analysis, also preferably for < 30 Hz
    %     cfg = [];
    %     cfg.output     = 'pow';	            % cond输出power
    %     cfg.pad        = 'nextpow2';        % 让fft计算更快
    %     cfg.channel    = 'all';
    %     cfg.method     = 'wavelet';         % 使用小波分析
    %      cfg.width      = linspace(3,10,30); % number of cycles       12-20hz            1/10/pi - 1/6/pi
    % % cfg.width      =  6
    %     cfg.foi        = linspace(2,30,30); % frequencies	           bandwidth = f/cycle*2     durtion = cycle/f/pi
    %     cfg.toi        = -0.5:0.002:1.2;	    % time window "slides" from -0.2 to 0.5 sec in steps of 0.02 sec (20 ms)
    %     TF_MC_M_C = ft_freqanalysis(cfg, TF_MC_M_C);   % for A condition
    %     power_B   = ft_freqanalysis(cfg, condA);   % for B condition
    %
    %
    %     %%% plot subject-TFR
    %     % 查看每个被试的subject-TFR是有必要的，这样可以检查一下每个被试的数据质量；如果太差，需要剔除该被试
    %     cfg = [];
    %     cfg.baseline     = [-0.2 0];
    %     cfg.baselinetype = 'db'; 	    % decibe    基线校正的方法
    %      cfg.xlim         = [-0.2 0.9];  % time axis 时间范围
    %     cfg.zlim         = [-3 3];      %颜色条范围
    %     cfg.showlabels   = 'yes';
    %     cfg.layout       =  'F:\学习视频\时频\zky时频\wang\ft_template\NeuroScan_quickcap64_layout.lay';
    %     figure;
    %     ft_multiplotTFR(cfg, TF_MC_F_C);
    %     查找各类型的帽子所对应的layout，见： http://www.fieldtriptoolbox.org/template
    
    %     save(['E:\数据\分析文件\MC_F_C\MCFC' id{subi} '_CleanData.mat'],'condD')
    
    save(['E:\数据\重建\时频全部-stft\TF_MI_S_I' id{subi} '.mat'], 'TF_MI_S_I')
    fprintf('\n############ completed for subject %s ###########\n\n\n\n', id{subi} )
end
load train
sound(y,Fs)
%% prepare data for grand-avg and statistical test
% close all
clear
cd('C:\Users\TIAN\Desktop\新建文件夹 (2)')
% id = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'};  % 被试编号
id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'};  % 被试编号
Ns = length(id);
% prepare the data structure (cell structure)
allsub_A = cell(1,Ns);
allsub_B = cell(1,Ns);
%
for subi=1:Ns
    load(['TF_MC_M_C' id{subi} '.mat']);
    
    % baseline normalization
    cfg = [];
    cfg.baseline     = [-0.2 0];
    cfg.baselinetype = 'db';
    % baselinetype option include 'absolute', 'relative', 'relchange', 'normchange' or 'db'
    %     power_A = ft_freqbaseline(cfg,power_A);
    power_A = ft_freqbaseline(cfg,TF_MC_M_C);
    %     power_C= ft_freqbaseline(cfg,power_C);
    %     power_D = ft_freqbaseline(cfg,power_D);
    % 将所有被试的数据组织成 cell array
    allsub_A{1,subi} = power_A;
    %     allsub_B{1,subi} = power_B;
    %     allsub_C{1,subi} = power_C;
    %     allsub_D{1,subi} = power_D;
end
for subi=1:Ns
    load(['TF_MC_M_I' id{subi} '.mat']);
    
    % baseline normalization
    cfg = [];
    cfg.baseline     = [-0.2 0];
    cfg.baselinetype = 'db';
    % baselinetype option include 'absolute', 'relative', 'relchange', 'normchange' or 'db'
    %     power_A = ft_freqbaseline(cfg,power_A);
    power_B = ft_freqbaseline(cfg,TF_MC_M_I);
    %     power_C= ft_freqbaseline(cfg,power_C);
    %     power_D = ft_freqbaseline(cfg,power_D);
    % 将所有被试的数据组织成 cell array
    allsub_B{1,subi} = power_B;
    %     allsub_B{1,subi} = power_B;
    %     allsub_C{1,subi} = power_C;
    %     allsub_D{1,subi} = power_D;
end
allsub = cat(2,allsub_A,allsub_B)
% cfg           = [];
% cfg.channel   = 'ALL';
% cfg.method    = 'triangulation';
% %cfg.grad      = TFR_diff.grad;
% cfg.feedback  = 'yes';
% neighbours    = ft_prepare_neighbours(cfg);
% cfg = [];
% cfg.parameter = 'powspctrm';
% cfg.operation = 'x1-x2';
% TFR_diff      = ft_math(cfg, power_A,power_B);
% cfg           = [];
% cfg.channel   = 'all';
% cfg.statistic = 'indepsamplesT';
% cfg.parameter   = 'powspctrm'
% cfg.method    = 'stats';
% cfg.correctm  = 'no';
% TFR_stat1     = ft_freqstatistics(cfg, TFR_diff);
%
% cfg = [];
% cfg.marker  = 'on';
% cfg.layout  = 'F:\学习视频\时频\wang\ft_template\NeuroScan_quickcap64_layout.lay';
% cfg.xlim   = [0.2 0.7];
% cfg.channel = 'all';
% cfg.showlabels   = 'yes';
% figure; ft_multiplotTFR(cfg, TFR_diff);
% colormap(jet);
% ft_topoplotER(cfg,TFR_diff);

%% calculate grand average for each condition
cfg = [];
cfg.channel   = 'all';
cfg.toilim    = 'all';
cfg.foilim    = 'all'
% cfg.keepindividual = 'yes'
pow_A_grand = ft_freqgrandaverage(cfg, allsub_A{:});
pow_B_grand = ft_freqgrandaverage(cfg, allsub_B{:});
pow_ALL_grand = ft_freqgrandaverage(cfg, allsub{:});
AC_vs_BC = pow_A_grand;
% % AD_vs_BD = pow_A_grand;
% % AC_vs_AD = pow_A_grand;
% % BC_vs_BD = pow_A_grand;
AC_vs_BC.powspctrm = pow_A_grand.powspctrm - pow_B_grand.powspctrm;
% pow_C_grand = ft_freqgrandaverage(cfg, allsub_C{:});
% pow_D_grand = ft_freqgrandaverage(cfg, allsub_D{:});
%%% plot grand average of TFR
cfg = [];
cfg.xlim         = [-0.25 0.95];  % time axis 时间范围
cfg.zlim         = [-3 3];
cfg.ylim         = [1 40];

cfg.showlabels   = 'yes';
cfg.showoutline   = 'yes';
cfg.layout       = 'F:\时频\zky视频\wang\ft_template\NeuroScan_quickcap64_layout.lay';
figure
ft_multiplotTFR(cfg,AC_vs_BC);
colormap(jet);
colorbar;
colormap(jet);

figure
ft_multiplotTFR(cfg,pow_B_grand);
colormap(jet);
colorbar;
colormap(jet);

figure
ft_multiplotTFR(cfg,AC_vs_BC);
colormap(jet);
colorbar;
colormap(jet);




figure
contourf(AC_vs_BC.time,AC_vs_BC.freq,squeeze(mean(AC_vs_BC.powspctrm(19,:,:),1)),40,'linecolor','none')
set(gca,'xlim',[-0.2 0.9], 'clim',[-3 3],'ylim',[2 40])
title('PO8')
xlabel('Time (s)')
ylabel('freq (Hz)')
colorbar
colormap(jet)    % 选择配色方案
% figure
% ft_multiplotTFR(cfg, pow_A_grand );
% colorbar;
% colormap(jet);
% 在界面上选择多个电极，点击后，画出的TFR是这几个电极的平均

%%% plot topography of TFR of WF effect
% difference TFR(两条件相减)
AC_vs_BC = pow_A_grand;
% % AD_vs_BD = pow_A_grand;
% % AC_vs_AD = pow_A_grand;
% % BC_vs_BD = pow_A_grand;
AC_vs_BC.powspctrm = pow_A_grand.powspctrm - pow_B_grand.powspctrm;
% % AD_vs_BD.powspctrm = pow_C_grand.powspctrm - pow_D_grand.powspctrm;
% % AC_vs_AD.powspctrm = pow_A_grand.powspctrm - pow_C_grand.powspctrm;
% % BC_vs_BD.powspctrm = pow_B_grand.powspctrm - pow_D_grand.powspctrm;
% % [h,p,ci,stats] = ttest2(pow_A_grand.powspctrm,pow_B_grand.powspctrm);
% 画出WF效应(差异TFR)的地形图
cfg = [];
cfg.parameter = 'powspctrm';
cfg.xlim   = [0 0.8];   % time limit, in second    时间范围
cfg.ylim   = [4 50];       % freq limit, in Hz    频率范围
cfg.layout = 'F:\学习视频\时频\zky时频\wang\ft_template\NeuroScan_quickcap64_layout.lay';
cfg.zlim   = [-1 1];
cfg.showlabels   = 'yes';
figure
ft_multiplotTFR(cfg,MC_F_diff);
colorbar;
colormap(jet);
figure
ft_multiplotTFR(cfg,AD_vs_BD);
colorbar;
colormap(jet);
figure
ft_multiplotTFR(cfg,AC_vs_AD);
colorbar;
colormap(jet);
figure
ft_multiplotTFR(cfg,BC_vs_BD);
colorbar;
colormap(jet);
% figure
% cfg.gridscale = 300;
% cfg.style = 'straight';
% cfg.marker = 'labels';
ft_topoplotER(cfg,pow_A_grand );
% colorbar;
% colormap(jet);
%% Compute the neighbours
cfg           = [];
cfg.channel   = 'all';
cfg.method    = 'triangulation';
cfg.grad      = power_B.elec;
cfg.feedback  = 'yes';
neighbours    = ft_prepare_neighbours(cfg);

%% Compute the statistics
cfg           = [];
cfg.channel   = 'all';
cfg.statistic = 'indepsamplesT';
cfg.ivar      = 1;
cfg.design    = zeros(1, size(allsub,2));

cfg.design(TFR_all.trialinfo(:,1)== 256) = 1;
cfg.design(TFR_all.trialinfo(:,1)==4096) = 2;

cfg.method    = 'analytic';
cfg.correctm  = 'no';
TFR_stat1     = ft_freqstatistics(cfg, pow_A_grand);


%%  proceed with three methods that do correct for the MCP.
cfg.method    = 'analytic';
cfg.correctm  = 'bonferoni';
TFR_stat2     = ft_freqstatistics(cfg, power);

cfg.method    = 'analytic';
cfg.correctm  = 'fdr';
TFR_stat3     = ft_freqstatistics(cfg, power);

cfg.method            = 'montecarlo';
cfg.correctm          = 'cluster';
cfg.numrandomization  = 1000; % 1000 is recommended, but takes longer
cfg.neighbours        = neighbours;
TFR_stat4     = ft_freqstatistics(cfg, pow_A_grand);
ft_multiplotTFR(cfg,TFR_stat4 );
%% Visualise the results
cfg               = [];
cfg.marker        = 'on';
cfg.layout        = 'F:\学习视频\时频\zky时频\wang\ft_template\NeuroScan_quickcap64_layout.lay';
%cfg.channel       = 'MEG*1';
cfg.parameter     = 'stat';  % plot the t-value
cfg.maskparameter = 'mask';  % use the thresholded probability to mask the data
% cfg.maskstyle     = 'outline';
cfg.maskalpha     = 0.05
cfg.xlim   = [-0.25 0.95];   % time limit, in second    时间范围
cfg.ylim   = [1 30];       % freq limit, in Hz    频率范围
cfg.zlim   = [-3 3];
cfg.showlabels   = 'yes';
figure; ft_multiplotTFR(cfg, TFR_stat1);

figure; ft_multiplotTFR(cfg, TFR_stat2);
figure; ft_multiplotTFR(cfg, TFR_stat3);
figure; ft_multiplotTFR(cfg, TFR_stat4);
%% 导出数据用于进一步统计分析
% 根据已有假设或前一步得到的总平均图，选择时间窗口、频率窗口、电极点（ROI）
time  = pow_A_grand.time;
freq  = pow_A_grand.freq;
chan  = pow_A_grand.label;

% define time window
timewin      = [0.5 0.6];
timewin_idx  = dsearchn(time', timewin');
% define frequency window
freqwin      = [10 15];  % theta band
freqwin_idx  = dsearchn(freq', freqwin');
% define ROI (channels)
chan2use = {'POZ'};
chan_idx = zeros(1,length(chan2use));
for i=1:length(chan2use)       % find the index of channels to use
    ch = strcmpi(chan2use(i), chan);
    chan_idx(i) = find(ch);
end

% extract mean TFR over these time window, freq window, and ROI, for each condition and each subject
id = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'};  % 被试编号
Ns = length(id);
power = zeros(Ns,2); % initialize variable， 此例中，有Ns个被试，2个条件
for subi=1:Ns
    pow1 = allsub_A{1,subi}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    %     pow2 = allsub_B{1,subi}.powspctrm( chan_idx, ...
    %         freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    %     pow3 = allsub_C{1,subi}.powspctrm( chan_idx, ...
    %         freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    %     pow4 = allsub_D{1,subi}.powspctrm( chan_idx, ...
    %         freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    power(subi,1) = squeeze(mean(mean(mean( pow1  ))));  % 提取第一个条件的数据
    %     power(subi,2) = squeeze(mean(mean(mean( pow2  ))));  % 提取第二个条件的数据
    %     power(subi,3) = squeeze(mean(mean(mean( pow3  ))));  % 提取第一个条件的数据
    %     power(subi,4) = squeeze(mean(mean(mean( pow4  ))));  % 提取第二个条件的数据
end
dlmwrite('F:\power.txt',power,'\t')  % 保存到txt文件中(用excel打开)，用于进一步分析
% 如有其他时频窗口或ROI，也类似操作
% 当选择了多个时频窗口或ROI时，进行了多次比较，此时需要对p值进行校正（eg, FDR）

%%ft_definetrial

