clear
cd 'E:\����\�ؽ�\Ԥ�����������'
% id = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'};
id = {'19','21'};  % ���Ա��
% id = {'18'};  % ���Ա��
Ns = length(id);
%% read EEG data, separately for each condition
for subi=1:Ns
    InFile =['sub' id{subi}  '_pre.set']               % ����Ҫ�������ݵ�����
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {4,5,6,7,8,9};
    %     cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % ��һ��������marker
    cfg = ft_definetrial(cfg);
    mat = ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MC_I' id{subi}  '_CleanData.mat'],'mat')
    
    
    InFile =['sub' id{subi}  '_pre.set']                  % ����Ҫ�������ݵ�����
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {1,2,3};
    %    cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % ��һ��������marker
    cfg = ft_definetrial(cfg);
    mat= ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MC_C' id{subi}  '_CleanData.mat'],'mat')
    %     �������ݵ���һ���ļ���,ʹ�þ���·��
  
        InFile =['sub' id{subi}  '_pre.set']                  % ����Ҫ�������ݵ�����
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {40,50,60,70,80,90};
    %    cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % ��һ��������marker
    cfg = ft_definetrial(cfg);
    mat= ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MI_C' id{subi}  '_CleanData.mat'],'mat')
    %     �������ݵ���һ���ļ���,ʹ�þ���·��
    
    
        InFile =['sub' id{subi}  '_pre.set']                  % ����Ҫ�������ݵ�����
    cfg = [];
    cfg.dataset                 = InFile;
    cfg.channel                 = 'all';
    cfg.trialdef.eventtype      = 'trigger';
    cfg.trialdef.prestim        = 0.5;
    cfg.trialdef.poststim       = 1.198;
    cfg.trialdef.eventvalue     = {10,20,30};
    %    cfg.trialdef.eventvalue     = {1	2	3	4	5	6	7	8	9	10	20	30	40	50	60	70	80	90};
    % ��һ��������marker
    cfg = ft_definetrial(cfg);
    mat= ft_preprocessing(cfg);
    save([ 'H:\feildtrip(four)\MI_C' id{subi}  '_CleanData.mat'],'mat')
    %     �������ݵ���һ���ļ���,ʹ�þ���·��
 
    
    
    % �������ݵ���һ���ļ���,ʹ�þ���·��
    fprintf('\n############ completed for subject %s ###########\n\n\n\n', id{subi} )
    
end
