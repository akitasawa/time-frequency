% Using the code without proper understanding the code and relevant background 
% of EEG may lead to confusion, incorrect data analyses,or misinterpretations of results.
% The author assumes NO responsibility for inappropriate or incorrect use of this code.
%
% Author: Dr.Cheng Wang. E-mail: wangchengpsych@aliyun.com
%
% Cite as:
% Wang, C., & Zhang, Q.* (2021). Word frequency effect in written production: 
%           Evidence from ERPs and neural oscillations. Psychophysiology, 58: e13775. https://doi.org/10.1111/psyp.13775
% 请引用上述研究，本脚本中的的示范数据和示范代码都来自于上述研究。
% Please cite the above research. All demo codes and demo data used here are from this research.



%% =============================================================== 
%%% functional connectivity, ie, power correlation or phase coherence
%=======================================================================
cd('C:\wang\ft_data')
close all
clear
id = {'1','2'};  % 被试编号
Ns = length(id);

for subi=1:Ns    
    load(['sub' id{subi} '_CleanData.mat']);
    
    % get analytical signal via TF decomposition，and keep trials
    cfg = [];
    cfg.method       = 'mtmconvol';
    cfg.taper        = 'hanning';
    cfg.output       = 'fourier';
    cfg.pad          = 'nextpow2';
    cfg.keeptrials   = 'yes';  
    cfg.foi          = 2:3:30;                     % 可以选择更密的频率点，eg, 1:1:30     
    cfg.t_ftimwin    = ones(size(cfg.foi)).*0.5;  
    cfg.toi          = -0.3:0.02:0.5;              % 可以选择更密的时间点, eg, -0.2:0.01:0.5
    freqA = ft_freqanalysis(cfg, condA); 
    freqB = ft_freqanalysis(cfg, condB); 

    % compute connectivity
    cfg         = [];
    cfg.method  = 'wpli_debiased';  % or coh,wpli,wpli_debiased
    cohA        = ft_connectivityanalysis(cfg, freqA); 
    cohB        = ft_connectivityanalysis(cfg, freqB); 
    
%     % plot TFR of coh
%     cfg           = [];
%     cfg.channel   = {'fz','fpz','pz','p1','p2'}; 
%     cfg.parameter = 'wpli_debiasedspctrm';
% %     cfg.zlim      = [0 0.5];
%     ft_connectivityplot(cfg, cohA);    
       
%     % topoplot of coh
%     cfg                  = [];
%     cfg.parameter        = 'wpli_debiasedspctrm';
%     cfg.xlim             = [0.2 0.4];   % time limit
%     cfg.ylim             = [8 13];      % freq limit
% %     cfg.zlim             = [-1 1];      % coloarbar limit
%     cfg.refchannel       = 'fz';        % seed channel
%     cfg.layout           = 'C:\wang\ft_template\NeuroScan_quickcap64_layout.lay';
%     ft_topoplotER(cfg, cohA)

    save(['C:\wang\COH\coh_sub' id{subi} '.mat'], 'cohA', 'cohB')     
end




%%% prepare data for grand-avg and statistical test
close all
clear
cd('C:\wang\COH')
id = {'1','2'};
Ns = length(id);

chan = {'TP7','PO7'};   % define seed channel
load('coh_sub1.mat');
chan_idx  = match_str(cohA.label, chan);   % index of seed channels

% prepare the data structure (cell structure)
allsub_A = cell(1,Ns);
allsub_B = cell(1,Ns);

for subi=1:Ns    
    load(['coh_sub' id{subi} '.mat']);
    
    % select data for this seed 
    cohA.wpli_debiasedspctrm = squeeze(mean(abs(cohA.wpli_debiasedspctrm(chan_idx,:,:,:)),1));
    cohB.wpli_debiasedspctrm = squeeze(mean(abs(cohB.wpli_debiasedspctrm(chan_idx,:,:,:)),1));
    cohA.dimord = 'chan_freq_time';
    cohB.dimord = 'chan_freq_time';
    
    % baseline normalization
    cfg = [];
    cfg.parameter    = 'wpli_debiasedspctrm';
    cfg.baseline     = [-0.3 -0.1];
    cfg.baselinetype = 'absolute';   
    cohA = ft_freqbaseline(cfg,cohA);
    cohB = ft_freqbaseline(cfg,cohB);  
    
    cohA.wpli_debiasedspctrm = atanh(cohA.wpli_debiasedspctrm);     % Fisher-Z transform
    cohB.wpli_debiasedspctrm = atanh(cohB.wpli_debiasedspctrm);    
    
    % 将所有被试的数据组织成 cell array
    allsub_A{1,subi} = cohA;        
    allsub_B{1,subi} = cohB;
end 


%%% grand avg
cfg = [];
cfg.parameter = 'wpli_debiasedspctrm';
cfg.channel   = 'all';
cfg.toilim    = 'all';
cfg.foilim    = 'all';
A_grand = ft_freqgrandaverage(cfg, allsub_A{:});
B_grand = ft_freqgrandaverage(cfg, allsub_B{:});

%%% plot grand average of TFR
cfg = [];
cfg.parameter = 'wpli_debiasedspctrm';
% cfg.zlim         = [-0.4 0.4];	        
cfg.showlabels   = 'yes';	
cfg.layout       = 'C:\wang\ft_template\NeuroScan_quickcap64_layout.lay';
figure; ft_multiplotTFR(cfg, A_grand);
figure; ft_multiplotTFR(cfg, B_grand);


% difference TFR(两条件相减)
A_vs_B = A_grand;
A_vs_B.wpli_debiasedspctrm = A_grand.wpli_debiasedspctrm - B_grand.wpli_debiasedspctrm; 

cfg = [];
cfg.parameter = 'wpli_debiasedspctrm';
cfg.layout    = 'C:\wang\ft_template\NeuroScan_quickcap64_layout.lay';
% cfg.zlim      = [-0.2 0.2];	        
figure 
ft_multiplotTFR(cfg, A_vs_B);

% 经过以上处理，此数据的结构与时频分析结果的结构完全一样。
% 统计时，根据数据结构，从中选择电极对，选择时间窗口，选择频率窗口，平均后作为该被试该条件的值。
% 也可做cluster-based permutation test
% 分析时，注意不要把seed channel考虑在内。



%% =================================================== 
%%% effective connectivity, ie, granger causality
%===========================================================
cd('C:\wang\ft_data')
close all
clear
id = {'1','2'};  % 被试编号
Ns = length(id);


for subi=1:Ns    
    load(['sub' id{subi} '_CleanData.mat']);

    cfg = [];
    cfg.channel      = {'f1','f2','f3','f4','fz','p1','p2','p3','p4','pz'};   % select channels that are highlighted in COH analysis 
    cfg.method       = 'mtmconvol';
    cfg.output       = 'fourier';
    cfg.taper        = 'hanning';
    cfg.pad          = 'nextpow2';
    cfg.foi          = 2:2:30;                          
    cfg.t_ftimwin    = ones(size(cfg.foi)).*0.5;  
    cfg.toi          = -0.3:0.02:0.5;
    freqA = ft_freqanalysis(cfg, condA); 
    freqB = ft_freqanalysis(cfg, condB);  
    freqA.foi = cfg.foi;
    freqB.foi = cfg.foi;    

    % compute granger causality
    cfg            = [];
    cfg.method     = 'granger';
    gcA            = ft_connectivityanalysis(cfg, freqA);
    gcB            = ft_connectivityanalysis(cfg, freqB);
 
    % plot TFR of GC
    cfg           = [];
    cfg.parameter = 'grangerspctrm';
%     cfg.zlim      = [0 0.5];
    ft_connectivityplot(cfg, gcA);    
%     
%     % plot TFR of GC
    cfg = [];
    cfg.parameter    = 'grangerspctrm';
    cfg.refchannel   = 'fz';        % seed channel, which is the effect channel
%     cfg.zlim         = [0 4];	        
    cfg.showlabels   = 'yes';	
    cfg.layout       = 'F:\Matlab\toolbox\fieldtrip-20200605\NeuroScan_quickcap64_layout.lay';
    ft_multiplotTFR(cfg, gcA);  

%     save(['C:\wang\COH\gc_sub' id{subi} '.mat'], 'gcA', 'gcB')     
end



%%% prepare data for grand-avg and statistical test
close all
clear
cd('C:\Users\TIAN\Desktop')
id={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'};
Ns = length(id);
load('E:\格兰杰\MC_M_C21.mat')

% select seed channel (from), note the direction of GC
seed     = {'FZ'};  % 区分大小写
seed_idx = match_str(gcA.label, seed);

% select aim channel (to)
aim      = {'POZ'};
aim_idx  = match_str(gcA.label, aim);

% prepare the data structure (cell structure)
allsub_A = cell(1,Ns);
allsub_B = cell(1,Ns);

for subi=1:Ns    
    load(['MC_M_C' id{subi} '.mat']);
    
    gcA.grangerspctrm = shiftdim(mean(mean(gcA.grangerspctrm(seed_idx,aim_idx,:,:),1),2),1);
%     gcB.grangerspctrm = shiftdim(mean(mean(gcB.grangerspctrm(seed_idx,aim_idx,:,:),1),2),1);
    gcA.label  = {'from Front'};
%     gcB.label  = {'from Front'};
    gcA.dimord = 'chan_freq_time';
%     gcB.dimord = 'chan_freq_time';
    
    % baseline normalization
    cfg = [];
    cfg.parameter    = 'grangerspctrm';
    cfg.baseline     = [-0.2 0];
    cfg.baselinetype = 'absolute';   
    gcA = ft_freqbaseline(cfg,gcA);
%     gcB = ft_freqbaseline(cfg,gcB);  
    
    % 将所有被试的数据组织成 cell array
    allsub_A{1,subi} = gcA;        
%     allsub_B{1,subi} = gcB;
end 


%%% grand avg
cfg = [];
cfg.parameter = 'grangerspctrm';
cfg.channel   = 'all';
cfg.toilim    = 'all';
cfg.foilim    = 'all';
A_grand = ft_freqgrandaverage(cfg, allsub_A{:});
% B_grand = ft_freqgrandaverage(cfg, allsub_B{:});

%%% plot grand average of TFR
cfg = [];
cfg.parameter = 'grangerspctrm';
% cfg.zlim         = [-3 3];	        
cfg.showlabels   = 'yes';	
cfg.layout       = 'F:\Matlab\toolbox\fieldtrip-20200605\NeuroScan_quickcap64_layout.lay';
figure; ft_singleplotTFR(cfg, A_grand);
figure; ft_singleplotTFR(cfg, B_grand);


% difference TFR(两条件相减)
A_vs_B = A_grand;
A_vs_B.wpli_spctrm = A_grand.wpli_spctrm - B_grand.wpli_spctrm; 


% 统计时，根据数据结构，选择时间窗口，选择频率窗口，平均后作为该被试该条件的值。
% 也可做cluster-based permutation test
% 分析时，注意不要把seed channel考虑在内。


%=============  cluster-based permutation test  ==============
%%% cluster-based permutation
cfg = [];
cfg.parameter        = 'grangerspctrm';
cfg.channel          = {'all'};
cfg.avgoverchan      = 'yes';
cfg.latency          = [0 0.5];     % in second
cfg.avgovertime      = 'no'; 
cfg.frequency        = 'all';
cfg.avgoverfreq      = 'no'; 
cfg.method           = 'montecarlo';
cfg.statistic        = 'depsamplesT';
cfg.correctm         = 'cluster';
cfg.clustertail      = 0;           % alpha level of the sample-specific test statistic
cfg.clusteralpha     = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.tail             = 0;
cfg.alpha            = 0.05;        % alpha level of the permutation test
cfg.correcttail      = 'alpha';     % correct alpha level for a two tailed test
cfg.numrandomization = 1000;

cfg.design(1,1:2*Ns)  = [ones(1,Ns) 2*ones(1,Ns)];
cfg.design(2,1:2*Ns)  = [1:Ns 1:Ns];
cfg.ivar                = 1; % the 1st row in cfg.design contains the independent variable
cfg.uvar                = 2; % the 2nd row in cfg.design contains the subject number
 
stat = ft_freqstatistics(cfg,allsub_A{:},allsub_B{:});


