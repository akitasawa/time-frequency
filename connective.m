% find the interesting epochs of data
cfg = [];
cfg.trialfun                  = 'trialfun_left';
cfg.dataset                   = 'SubjectCMC.ds';
cfg = ft_definetrial(cfg);

% detect EOG artifacts in the MEG data
cfg.continuous                = 'yes';
cfg.artfctdef.eog.padding     = 0;
cfg.artfctdef.eog.bpfilter    = 'no';
cfg.artfctdef.eog.detrend     = 'yes';
cfg.artfctdef.eog.hilbert     = 'no';
cfg.artfctdef.eog.rectify     = 'yes';
cfg.artfctdef.eog.cutoff      = 2.5;
cfg.artfctdef.eog.interactive = 'no';
cfg = ft_artifact_eog(cfg);

% detect jump artifacts in the MEG data
cfg.artfctdef.jump.interactive = 'no';
cfg.padding                    = 5;
cfg = ft_artifact_jump(cfg);

% detect muscle artifacts in the MEG data
cfg.artfctdef.muscle.cutoff      = 8;
cfg.artfctdef.muscle.interactive = 'no';
cfg = ft_artifact_muscle(cfg);

% reject the epochs that contain artifacts
cfg.artfctdef.reject          = 'complete';
cfg = ft_rejectartifact(cfg);

% preprocess the MEG data
cfg.demean                    = 'yes';
cfg.dftfilter                 = 'yes';
cfg.channel                   = {'MEG'};
cfg.continuous                = 'yes';
meg = ft_preprocessing(cfg);


cfg              = [];
cfg.dataset      = meg.cfg.dataset;
cfg.trl          = meg.cfg.trl;
cfg.continuous   = 'yes';
cfg.demean       = 'yes';
cfg.dftfilter    = 'yes';
cfg.channel      = {'EMGlft' 'EMGrgt'};
cfg.hpfilter     = 'yes';
cfg.hpfreq       = 10;
cfg.rectify      = 'yes';
emg = ft_preprocessing(cfg);

data = ft_appenddata([], meg, emg);
save data data

%%
load data
figure
subplot(2,1,1);
plot(data.time{1},data.trial{1}(77,:));
axis tight;
legend(data.label(77));

subplot(2,1,2);
plot(data.time{1},data.trial{1}(152:153,:));
axis tight;
legend(data.label(152:153));

%%
%Method 1
cfg            = [];
cfg.output     = 'fourier';
cfg.method     = 'mtmfft';
cfg.foilim     = [5 100];
cfg.tapsmofrq  = 5;
cfg.keeptrials = 'yes';
cfg.channel    = {'MEG' 'EMGlft' 'EMGrgt'};
freqfourier    = ft_freqanalysis(cfg, data);


%Method 2
cfg            = [];
cfg.output     = 'powandcsd';
cfg.method     = 'mtmfft';
cfg.foilim     = [5 100];
cfg.tapsmofrq  = 5;
cfg.keeptrials = 'yes';
cfg.channel    = {'MEG' 'EMGlft' 'EMGrgt'};
cfg.channelcmb = {'MEG' 'EMGlft'; 'MEG' 'EMGrgt'};
freq           = ft_freqanalysis(cfg, data);

%connectivityanalysis
cfg            = [];
cfg.method     = 'coh';
cfg.channel   = {'fz', 'pz', 'cz'};
fd             = ft_connectivityanalysis(cfg, freq);
% fdfourier      = ft_connectivityanalysis(cfg, freqfourier);


%% visual  The coherence between the left EMG and all the MEG sensors
...calculated using ft_freqanalysis and ft_connectivityanalysis.
    ...Plotting was done with ft_multiplotER
    cfg                  = [];
cfg.parameter        = 'cohspctrm';
cfg.xlim             = [5 30];
cfg.refchannel       = 'gui';
cfg.layout           = 'F:\时频\zky视频\wang\ft_template\NeuroScan_quickcap64_layout.lay';
cfg.showlabels       = 'yes';
figure;ft_multiplotER(cfg, fdfourier);

cfg.channel = 'fz';
figure; ft_singleplotER(cfg, fdfourier);


cfg                  = [];
cfg.parameter        = 'cohspctrm';
cfg.xlim             = [10 15];
cfg.zlim             = [0 0.1];
cfg.refchannel       = 'gui';
cfg.layout           = 'F:\时频\zky视频\wang\ft_template\NeuroScan_quickcap64_layout.lay';
figure; ft_topoplotER(cfg, fdfourier);


cfg            = [];
cfg.output     = 'powandcsd';
cfg.method     = 'mtmfft';
cfg.foilim     = [5 100];
cfg.tapsmofrq  = 2;
cfg.keeptrials = 'yes';
cfg.channel    = {'MEG' 'EMGlft'};
cfg.channelcmb = {'MEG' 'EMGlft'};
freq2          = ft_freqanalysis(cfg,data);

cfg            = [];
cfg.method     = 'coh';
cfg.channelcmb = {'MEG' 'EMG'};
fd2            = ft_connectivityanalysis(cfg,freq2);

cfg               = [];
cfg.parameter     = 'cohspctrm';
cfg.refchannel    = 'EMGlft';
cfg.xlim          = [5 80];
cfg.channel       = 'MRC21';
figure; ft_singleplotER(cfg, fd, fd2);
