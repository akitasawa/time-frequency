%%
clear all ;
% cd H:\����\ʱƵ\flanker\ʱƵ\ʱƵȫ������
cd('E:\����\�ؽ�\ʱƵȫ��-stft')
%����ÿ�������ı��ԣ�׼��ƽ��
% condition = {'STFT_MC_M_I','STFT_MC_M_C','STFT_MC_F_I','STFT_MC_F_C','STFT_MC_S_I','STFT_MC_S_C','STFT_MI_M_I','STFT_MI_M_C',...
%     'STFT_MI_F_I','STFT_MI_F_C','STFT_MI_S_I','STFT_MI_S_C'};
% TF_condition = {'TF_MC_M_I','TF_MC_M_C','TF_MC_F_I','TF_MC_F_C','TF_MC_S_I','TF_MC_S_C','TF_MI_M_I','TF_MI_M_C',...
%     'TF_MI_F_I','TF_MI_F_C','TF_MI_S_I','TF_MI_S_C'};
condition = {'TF_MC_M_I','TF_MC_M_C','TF_MC_F_I','TF_MC_F_C','TF_MC_S_I','TF_MC_S_C','TF_MI_M_I','TF_MI_M_C',...
    'TF_MI_F_I','TF_MI_F_C','TF_MI_S_I','TF_MI_S_C'};
TF_condition = {'TF_MC_M_I','TF_MC_M_C','TF_MC_F_I','TF_MC_F_C','TF_MC_S_I','TF_MC_S_C','TF_MI_M_I','TF_MI_M_C',...
    'TF_MI_F_I','TF_MI_F_C','TF_MI_S_I','TF_MI_S_C'};

nu = length(condition);
for cond = 1:nu
    id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'};
%     id = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'};  % ���Ա��
    Ns = length(id);
    for subi=1:Ns
        load([condition{cond} id{subi} '.mat']);
        pow = eval(TF_condition{cond})
        cfg = [];
        cfg.baseline     = [-0.2 0];
        cfg.baselinetype = 'db';
        power = ft_freqbaseline(cfg,pow);
        all_power{cond,subi} = power;
    end
end

f = {'X','Y'}

s  = ones(2,30)
for i = 1:2
    s(i,:) = eval([f{i} '_data'])

end

for i =1:length(id)
    ALL_MC_M_I{1,i} =  all_power{1,i}
    ALL_MC_M_C{1,i} =   all_power{2,i}
    ALL_MC_F_I{1,i} =  all_power{3,i}
    ALL_MC_F_C{ 1,i}=  all_power{4,i}
    ALL_MC_S_I{1,i}=  all_power{5,i}
    ALL_MC_S_C{1,i} =  all_power{6,i}
    ALL_MI_M_I{1,i} =  all_power{7,i}
    ALL_MI_M_C{1,i} =  all_power{8,i}
    ALL_MI_F_I{1,i} =  all_power{9,i}
    ALL_MI_F_C{1,i} =  all_power{10,i}
    ALL_MI_S_I{1,i} =  all_power{11,i}
    ALL_MI_S_C{1,i} =  all_power{12,i}
    
end

save(['E:\����\�ؽ�\ʱƵȫ��-stft\all_stft.mat'],'ALL_MC_M_I',...
    'ALL_MC_M_C','ALL_MC_F_I','ALL_MC_F_C','ALL_MC_S_I','ALL_MC_S_C',...
    'ALL_MI_M_I','ALL_MI_M_C','ALL_MI_F_I','ALL_MI_F_C','ALL_MI_S_I','ALL_MI_S_C')



%ƽ��
cfg = [];
cfg.keepindividual = 'yes'    %ͳ����Ҫ����ͳ����no
cfg.channel   = 'all';
cfg.toilim    = 'all';
cfg.foilim    = 'all'


all_F = {ALL_MC_M_I{:},ALL_MC_M_C{:}}   % �Ա���������
all_MC_M_Cluster =  ft_freqgrandaverage(cfg,all_F{:});  %������ƽ��

% ������ƽ��
MC_F_C_grand =  ft_freqgrandaverage(cfg, ALL_MC_F_C{:});
MC_F_I_grand =  ft_freqgrandaverage(cfg, ALL_MC_F_I{:});
MC_S_C_grand =  ft_freqgrandaverage(cfg, ALL_MC_S_C{:});
MC_S_I_grand =  ft_freqgrandaverage(cfg, ALL_MC_S_I{:});
MC_M_C_grand =  ft_freqgrandaverage(cfg, ALL_MC_M_C{:});
MC_M_I_grand =  ft_freqgrandaverage(cfg, ALL_MC_M_I{:});
MI_F_C_grand =  ft_freqgrandaverage(cfg, ALL_MI_F_C{:});
MI_F_I_grand =  ft_freqgrandaverage(cfg, ALL_MI_F_I{:});
MI_S_C_grand =  ft_freqgrandaverage(cfg, ALL_MI_S_C{:});
MI_S_I_grand =  ft_freqgrandaverage(cfg, ALL_MI_S_I{:});
MI_M_C_grand =  ft_freqgrandaverage(cfg, ALL_MI_M_C{:});
MI_M_I_grand =  ft_freqgrandaverage(cfg, ALL_MI_M_I{:});

save(['C:\Users\TIAN\Desktop\all_average.mat'],'MI_M_I_grand',...
    'MI_M_C_grand','MI_S_I_grand','MI_S_C_grand','MI_F_I_grand','MI_F_C_grand',...
    'MC_M_I_grand','MC_M_C_grand','MC_S_I_grand','MC_S_C_grand','MC_F_I_grand','MC_F_C_grand')
%%  ����
MC_F_diff = MC_F_C_grand;
MC_F_diff.powspctrm = MC_F_I_grand .powspctrm - MC_F_C_grand.powspctrm;

MC_S_diff = MC_F_C_grand;
MC_S_diff.powspctrm = MC_S_I_grand .powspctrm - MC_S_C_grand.powspctrm;

MC_M_diff = MC_F_C_grand;
MC_M_diff.powspctrm = MC_M_I_grand .powspctrm - MC_M_C_grand.powspctrm;

MI_F_diff = MC_F_C_grand;
MI_F_diff.powspctrm = MI_F_I_grand .powspctrm - MI_F_C_grand.powspctrm;

MI_S_diff = MC_F_C_grand;
MI_S_diff.powspctrm = MI_S_I_grand .powspctrm - MI_S_C_grand.powspctrm;

MI_M_diff = MC_F_C_grand;
MI_M_diff.powspctrm = MI_M_I_grand .powspctrm - MI_M_C_grand.powspctrm;

save(['E:\����\�ؽ�\ʱƵȫ��-stft\all_stft.mat'])

%% ͼ��

%������ͼ��
load 'E:\����\�ؽ�\ALL1.mat'
cfg = [];
cfg.xlim         = [-0.25 0.95];
cfg.zlim         = [-5 5];
cfg.showlabels   = 'yes';
cfg.showoutline   = 'yes';
cfg.layout       = 'F:\Matlab\toolbox\fieldtrip-20200605\NeuroScan_quickcap64_layout.lay';

grand = {'MC_M_I_grand ','MC_M_C_grand','MC_F_I_grand','MC_F_C_grand','MC_S_I_grand',...
    'MC_S_C_grand','MI_M_I_grand ','MI_M_C_grand','MI_F_I_grand','MI_F_C_grand','MI_S_I_grand',...
    'MI_S_C_grand'}
for g = 1:12
    pingjun = eval(grand{g})
    figure
    ft_multiplotTFR(cfg,pingjun);
    colormap(jet);
    colorbar;
    colormap(jet);
    title(grand{g})
    
end


%����ͼ
cfg = [];
cfg.xlim         = [-0.25 0.95];
cfg.zlim         = [-3 3];
cfg.showlabels   = 'yes';
cfg.showoutline   = 'yes';
cfg.layout       = 'F:\Matlab\toolbox\fieldtrip-20200605\NeuroScan_quickcap64_layout.lay';
ft_topoplotER(cfg,MC_M_diff);
diff = {'MC_M_diff ','MC_F_diff','MC_S_diff','MI_M_diff','MI_F_diff','MI_S_diff'}
for g = 1:6
    diff_pingjun = eval(diff{g})
    figure
    ft_multiplotTFR(cfg,diff_pingjun);
    colormap(jet);
    colorbar;
    colormap(jet);
    title(diff{g})
end


%% contourf��ͼ
figure
%  contourf(MC_M_C_grand.time,MC_M_I_grand.freq,squeeze(mean(MI_S_C_grand.powspctrm([9:11,18:20],:,:),1)),40,'linecolor','none')
contourf(MC_M_diff.time,MC_M_diff.freq,squeeze(mean(MI_S_diff.powspctrm([45:47,53:55,58:60],:,:),1)),40,'linecolor','none')
set(gca,'xlim',[-0.3 0.95], 'clim',[-3 3],'ylim',[2,40])
title('������')
xlabel('Time (s)')
ylabel('freq (Hz)')
%     colorbar
colormap(jet)    % ѡ����ɫ����
saveas(gcf,"C:\Users\TIAN\Desktop\����д��\������\MI_S_diff.jpg")


cfg=[];
cfg.toilim     = [0.6 0.8];
cfg.foilim     = [9 15];            % frequencies
cfg.layout  = 'F:\Matlab\toolbox\fieldtrip-20200605\layout\NeuroScan_quickcap64_layout.lay';
ft_topoplotER(cfg,MC_F_C_grand);

%%  ��time  = pow_A_grand.time;
freq  = ALL_MC_F_C{1, 1}.freq;
chan  = ALL_MC_F_C{1, 1}.label  ;
time = ALL_MC_F_C{1, 1}.time
% define time window
% timewin      = [0.7 0.9];
timewin      = [0.5 0.8];    %��
timewin_idx  = dsearchn(time', timewin');
% define frequency window
freqwin      = [4 7];  % theta band
% freqwin      = [11 21];  % alpha band
freqwin_idx  = dsearchn(freq', freqwin');
% define ROI (channels)
% chan2use = {'Fz', 'F1', 'F2', 'FCZ', 'FC1','FC2'};
chan2use = {'P1', 'PZ', 'P2', 'POZ', 'PO3','PO4','O1','OZ','O2'};
chanloc = zeros(length(ALL_MC_F_C{1, 1}.label),1)
% for i=1:length(chan2use)       % find the index of channels to use
%     ch = strcmpi(chan2use(i), chan);
%     chan_idx(i) = find(ch);
% end %ȡʱ���Ƶ�ʶ�
for i = 1:length(chan2use)
    for n = 1: 60
        chan = strcmpi(chan2use(i),ALL_MC_F_C{1, 1}.label(n))
        if chan == 1
            chanloc(n,1) = chan
        end
        
    end
end
chan_idx = find(chanloc==1)        %��ȡ�缫����



id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'}  % ���Ա��
Ns = length(id);
power = zeros(Ns,12); % initialize variable�� �����У���Ns�����ԣ�2������
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
    
    
    power(subi,1) = squeeze(mean(mean(mean( pow1  ))));  % ��ȡ��һ������������
    power(subi,2) = squeeze(mean(mean(mean( pow2  ))));  % ��ȡ�ڶ�������������
    power(subi,3) = squeeze(mean(mean(mean( pow3  ))));  % ��ȡ��һ������������
    power(subi,4) = squeeze(mean(mean(mean( pow4  ))));  % ��ȡ�ڶ�������������
    power(subi,5) = squeeze(mean(mean(mean( pow5  ))));  % ��ȡ��һ������������
    power(subi,6) = squeeze(mean(mean(mean( pow6  ))));  % ��ȡ�ڶ�������������
    power(subi,7) = squeeze(mean(mean(mean( pow7  ))));  % ��ȡ��һ������������
    power(subi,8) = squeeze(mean(mean(mean( pow8  ))));  % ��ȡ�ڶ�������������
    power(subi,9) = squeeze(mean(mean(mean( pow9  ))));  % ��ȡ��һ������������
    power(subi,10) = squeeze(mean(mean(mean( pow10  ))));  % ��ȡ�ڶ�������������
    power(subi,11) = squeeze(mean(mean(mean( pow11  ))));  % ��ȡ��һ������������
    power(subi,12) = squeeze(mean(mean(mean( pow12  ))));  % ��ȡ�ڶ�������������
end
TF_condition = {'TF_MC_M_I','TF_MC_M_C','TF_MC_F_I','TF_MC_F_C','TF_MC_S_I','TF_MC_S_C','TF_MI_M_I','TF_MI_M_C',...
    'TF_MI_F_I','TF_MI_F_C','TF_MI_S_I','TF_MI_S_C'};
title = (['����Ҷ' '��' 'ʱ��' num2str(timewin(1)) '����' num2str(timewin(2))  's' '��' ...
    'Ƶ��' num2str(freqwin(1)) '����' num2str(freqwin(2)) 'hz'])
xlswrite('C:\Users\TIAN\Desktop\����Ҷ_��.xlsx',title,1,"a1:a26");
xlswrite('C:\Users\TIAN\Desktop\����Ҷ_��.xlsx',TF_condition,1,"a2");    %��matlab����д��excel��SΪд�����ݣ�1Ϊsheet��c1Ϊ��Ԫ��λ�ã�
xlswrite('C:\Users\TIAN\Desktop\����Ҷ_��.xlsx',power,1,"a3");

dlmwrite('F:\power.txt',power,'\t')  % ���浽txt�ļ���(��excel��)�����ڽ�һ������




for n = 1:19
    pow = ALL_MI_M_I{1, n}.powspctrm(19,2:5,56:66)   %poz,        Time:0.6-0.8,        Fre:12-26
    pow2= ALL_MI_M_C{1, n}.powspctrm(19,2:5,56:66)
    pS(n,1) = squeeze(mean(mean(mean(pow))))
    pS(n,2) = squeeze(mean(mean(mean(pow2))))
end




%% ͳ��
%�������ڵ�
%ͳ�����
%ѡ��ͳ�ƺ�У������
%��ͼ
%% cluster-based permutation
%�����ٽ�ֵ
cfg = []
cfg.method      = 'template'; % try 'distance' as well
cfg.template    = 'F:\ʱƵ\zky��Ƶ\wang\ft_template\NeuroScan_quickcap64_neighbours.mat';    % specify type of template
cfg.layout      = 'F:\ʱƵ\zky��Ƶ\wang\ft_template\NeuroScan_quickcap64_layout.lay';
cfg.feedback    = 'yes';                             % show a neighbour plot
neighbours      = ft_prepare_neighbours(cfg,MC_F_diff); % define neighbouring channels
%  ͳ�����
cfg           = [];
% cfg.channel   = 'EEG*1';
cfg.statistic = 'depsamplesT';   %ѡ��ͳ�Ʒ���
cfg.ivar      = 1;
cfg.uvar      = 2;
% cfg.design    = zeros(1, size(all_F,2));
design = [ones(1,20),ones(1,20)*2;1:20,1:20]   %����ͳ�Ʒ�ʽ
cfg.design = design
cfg.channel          = {'all'};
% cfg.latency          = [0.1 0.5];     % in second
cfg.frequency        = 'all';
cfg.avgoverfreq      = 'no';
cfg.method           = 'montecarlo';
cfg.statistic        = 'depsamplesT';
cfg.correctm         = 'cluster';
cfg.clusteralpha     = 0.025;
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan        = 2;
cfg.tail             = 0;
cfg.clustertail      = 0;
cfg.alpha            = 0.05; % for cfg.alpha, which control the false alarm rate of the permutation test
cfg.numrandomization = 500;
cfg.neighbours       = neighbours;
Cluster_MC_M   = ft_freqstatistics(cfg, ALL_MC_M_I{:},ALL_MC_M_C{:});
% Cluster_MC_F   = ft_freqstatistics(cfg, ALL_MC_F_I{:},ALL_MC_F_C{:});
% Cluster_MC_S   = ft_freqstatistics(cfg, ALL_MC_S_I{:},ALL_MC_S_C{:});
% Cluster_MI_M   = ft_freqstatistics(cfg, ALL_MI_M_I{:},ALL_MI_M_C{:});
% Cluster_MI_F   = ft_freqstatistics(cfg, ALL_MI_F_I{:},ALL_MI_F_C{:});
% Cluster_MI_S   = ft_freqstatistics(cfg, ALL_MI_S_I{:},ALL_MI_S_C{:});
%% T����
cfg.method    = 'analytic';
cfg.correctm  = 'no';
cfg.alpha = 0.05
TFR_stat1     = ft_freqstatistics(cfg, Cluster_MC_M)
%% ѡ��У������
% bonferonУ��
cfg.method    = 'analytic';
cfg.correctm  = 'bonferoni';
TFR_stat2     = ft_freqstatistics(cfg, all_MC_M_Cluster);

% fdrУ��   �Ƚ��ϸ�
cfg.method    = 'analytic';
cfg.correctm  = 'fdr';
TFR_stat3     = ft_freqstatistics(cfg,  all_MC_M_Cluster);


%% ͳ�ƻ�ͼ
cfg               = [];
cfg.marker        = 'on';
cfg.layout        = 'F:\ʱƵ\zky��Ƶ\wang\ft_template\NeuroScan_quickcap64_layout.lay';
%cfg.channel       = 'EEG*1';
cfg.parameter     = 'stat';  % �ó���tֵ ע��������stat��prob��mask����ֵ��statΪtֵ��probΪpֵ��mask���Ƿ�����p�Ĵ�С
cfg.maskparameter = 'mask';  % �Ƿ��ڱ�
cfg.maskstyle     = 'outline';
% cfg.maskalpha     = 0.05    %ѡ���ù����ֵ��Ҫ�����ֵ
cfg.xlim   = [-0.25 0.95];   %    ʱ�䷶Χ
cfg.ylim   = [1 30];       %     Ƶ�ʷ�Χ
cfg.zlim   = [-10 10];
cfg.showlabels   = 'yes';

figure; ft_multiplotTFR(cfg, Cluster_MC_M);
figure; ft_multiplotTFR(cfg, Cluster_MC_F);
figure; ft_multiplotTFR(cfg, Cluster_MC_S);
figure; ft_multiplotTFR(cfg, Cluster_MI_M);
figure; ft_multiplotTFR(cfg, Cluster_MI_F);
figure; ft_multiplotTFR(cfg, Cluster_MI_S);



%%


load chirp
sound(y,Fs)

% ��һ��
sound(sin(2*pi*25*(1:4000)/100));

% ������
sound(sin(2*pi*25*(1:4000)/100));
sleep(1);
sound(sin(2*pi*25*(1:4000)/100));

% ����
load chirp
sound(y,Fs)

% ����
load gong
sound(y,Fs)

% ����·��
load handel
sound(y,Fs)

% Ц��
load laughter
sound(y,Fs)

% ž����
load splat
sound(y,Fs)

% ��
load train
sound(y,Fs)