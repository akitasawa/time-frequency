clear
cd 'E:\数据\重建\预处理完的数据'
% id = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'};
id = {'19','21'};  % 被试编号
% id = {'18'};  % 被试编号
Ns = length(id);
%% read EEG data, separately for each condition
for subi=1:Ns
    InFile =['sub' id{subi}  '_pre.set']               % 定义要读入数据的名字
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {4,5,6,7,8,9};
    %     cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % 另一个条件的marker
    cfg = ft_definetrial(cfg);
    mat = ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MC_I' id{subi}  '_CleanData.mat'],'mat')
    
    
    InFile =['sub' id{subi}  '_pre.set']                  % 定义要读入数据的名字
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {1,2,3};
    %    cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % 另一个条件的marker
    cfg = ft_definetrial(cfg);
    mat= ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MC_C' id{subi}  '_CleanData.mat'],'mat')
    %     保存数据到另一个文件夹,使用绝对路径
  
        InFile =['sub' id{subi}  '_pre.set']                  % 定义要读入数据的名字
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {40,50,60,70,80,90};
    %    cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % 另一个条件的marker
    cfg = ft_definetrial(cfg);
    mat= ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MI_C' id{subi}  '_CleanData.mat'],'mat')
    %     保存数据到另一个文件夹,使用绝对路径
    
    
        InFile =['sub' id{subi}  '_pre.set']                  % 定义要读入数据的名字
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {10,20,30};
    %    cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % 另一个条件的marker
    cfg = ft_definetrial(cfg);
    mat= ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MI_C' id{subi}  '_CleanData.mat'],'mat')
    %     保存数据到另一个文件夹,使用绝对路径
 
    
    
    % 保存数据到另一个文件夹,使用绝对路径
    fprintf('\n############ completed for subject %s ###########\n\n\n\n', id{subi} )
    
end
