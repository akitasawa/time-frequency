%%
clear all
cfg = [];
cfg.dataset     = '001EDMSall.cnt';
cfg.reref       = 'yes';
cfg.channel     = 'all';
%cfg.implicitref = 'M1';         % the implicit (non-recorded) reference channel is added to the data representation
cfg.refchannel  = {'M1', 'M2'}; 
cfg.lpfilter        = 'yes';
cfg.lpfreq          = 30;
cfg.hpfilter        = 'yes';
cfg.hpfreq          = 1;
data_eeg        = ft_preprocessing(cfg);
cfg.trialdef.eventtype = 'trigger';
cfg.trialdef.eventvalue = {101};
cfg         = ft_definetrial(cfg);
A          = ft_preprocessing(cfg,data_eeg);
cfg.trialdef.eventvalue = {102};
cfg         = ft_definetrial(cfg);
B          = ft_preprocessing(cfg,data_eeg);
