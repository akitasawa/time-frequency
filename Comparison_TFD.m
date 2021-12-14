% clear all; clc;
% Sub={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'};
%  Cond = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{20},{30},{40},{50},{60},{70},{80},{90}}
% save('E:\����\�ؽ�\Ԥ�����������\Cond.mat','Cond')
clear all;
id = { '002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017',...
    '018'	,'019',	'020'	,'021',	'022',	'023'	,'024'	,'025',	'026',	'027'	,'028'	,'029'	,'030',	'031',	'032',...
    '033'	,'034',	'040'	,'041'	,'042',	'043'	,'044'	,'045'	,'046',	'047',	'048'	,'049',...
    '050',	'051'	,'052'	,'053'};
Cond= { {'S 11'}  {'S 22'}  {'S 33'}  {'S 36'} }
%%  time-frequency analysis for multiple conditions
% �������б���
for i=1:length(id)
    tic
    setname=([ id{i} '.set']);
    setpath='H:\CSPCE\ȥα����';
    EEG= pop_loadset('filename',setname,'filepath',setpath);
    EEG= eeg_checkset( EEG );
    % ������������
    for j=1:length(Cond)
        EEG_new = pop_epoch( EEG,Cond{1, j}, [-0.8 1.2], 'newname', 'Merged datasets pruned with ICA   epochs epochs', 'epochinfo', 'yes');
        EEG_new = eeg_checkset( EEG_new );
%         EEG_new = pop_rmbase( EEG_new, [-200    0]);
%         EEG_new = eeg_checkset( EEG_new );
        % ��������ͨ��
        for nchan=1:size(EEG_new.data,1)
            x = squeeze(EEG_new.data(nchan,:,:));
            xtimes=EEG_new.times/1000;
            t=EEG_new.times/1000;
            f=1:1:30;
            Fs = EEG.srate;
            %             algm_opt = 'fft';
            winsize = 0.400;
            [S, P, F, U] = sub_stft(x, xtimes, t, f, Fs, winsize);
            % ���� * ���� * ͨ�� * Ƶ�� * ʱ���
            P_DATA(i,j,nchan,:,:)=squeeze(mean(P,3)); %%P_data (without baseline correciton):  subj*cond*chan*f*time
        end
    end
    toc
    fprintf('\n############ completed for subject %s ###########\n\n\n\n',id{i} )
    waitbar(i/length(id))
    
end
save('H:\CSPCE\wavelet.mat','P_DATA','xtimes','f','P_DB')
%% baseline correction
% �������б��ԡ�������ͨ����Ƶ�ʣ���������У��
t_pre_idx=find((xtimes>=-0.2)&(xtimes<=0));
for i=1:size(P_DATA,1)%���б���
    for j=1:size(P_DATA,2)%��������
        for ii=1:size(P_DATA,3)%����ͨ��
            for jj=1:size(P_DATA,4)%����Ƶ��
                temp_data=squeeze(P_DATA(i,j,ii,jj,:));
%                                 P_CC(i,j,ii,jj,:)=temp_data-mean(temp_data(t_pre_idx))  ;    %absolute��
                P_DB(i,j,ii,jj,:) = 10*log10( bsxfun(@rdivide, temp_data,mean(temp_data(t_pre_idx))));  %DB��
            end
        end
    end
end

%% ttest for each time-frequency point

% P_data: ���� * ���� * ͨ�� * Ƶ�� * ʱ���
% ��ȡ���б��ԡ�����������13��ͨ�����ռ�ROI��Cz��������Ƶ�ʡ�����ʱ���power
%: data_test�� ���� *  ���� * Ƶ�� *  ʱ���
data_test=squeeze(P_BC(:,:,13,:,:)); %% select the data at Cz, data_test: subj*cond*frequency*time
%����ÿһ��Ƶ�ʵ�
for i=1:size(data_test,3)
    %����ÿһ��ʱ���
    for j=1:size(data_test,4)
        % ��ѡ��һ������������
        data_1=double(squeeze(data_test(:,1,i,j))); %% select condition L3 for each time-frequency point
        %��ѡ�ڶ�������������
        data_2=double(squeeze(data_test(:,2,i,j))); %% select condition L4 for each time-frequency point
        % ���T����
        [h,p,ci,stats]=ttest(data_1,data_2); %% ttest comparison
        % �洢��i��Ƶ�ʵ㡢��j��ʱ����pֵ
        P_ttest(i,j)=p; %% save the p value from ttest
        %  �洢Tֵ
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

%����ÿһ��ʱƵ��
for i=1:size(data_test,3)
    %����ÿ��ʱ���
    for j=1:size(data_test,4)
        % ��ѡ��������������
        data_anova=squeeze(data_test(:,:,i,j)); %% select the data at time-frequency point%������ÿ������ÿ��������
        %�ظ������������
        [p, table] = anova_rm(data_anova,'off');  %% perform repeated measures ANOVA
        %�洢��ʱƵ���pֵ
        P_anova(i,j)=p(1); %% save the data from ANOVA
        % �洢Fֵ
        F_anova(i,j)=table{2,5}; %% F value from ANOVA
    end
end

%%   ԭʼͼ
figure
% contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,2,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3)),40,'linecolor','none')
contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,4,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3))-squeeze(mean(mean(mean(P_DB(:,3,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3)),40,'linecolor','none')
% contourf(xtimes, f,squeeze(mean(mean(mean(P_DATA(:,1:2:4,[9;10;11;18;19;20],:,:),1),2),3))-squeeze(mean(mean(mean(P_DATA(:,2:2:4,[9;10;11;18;19;20],:,:),1),2),3)),40,'linecolor','none')
% % contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,1:12,[7;8;9;10;11;12;16;17;19;20;21;22],:,:),1),2),3)),40,'linecolor','none')
set(gca,'xlim',[-0.4 1], 'clim',[-1 1])
title('ǰ��Ҷ:�������һ�µĳ�ͻ','fontsize',20)
xlabel('ʱ��(S)','fontsize',15)
ylabel('Ƶ��(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','dB','fontsize',15);
colormap(jet)    % ѡ����ɫ����
 rectangle('Position',[0.55 5 0.2 3],'Linestyle','--','EdgeColor','k')
rectangle('Position',[0.9 9 0.2 4],'Linestyle','--','EdgeColor','k')
% saveas(gcf,'C:\Users\TIAN\Desktop\����\����Ҷͼ\�������һ���²�һ�¼�ȥһ��.jpg')

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
    colormap(jet)    % ѡ����ɫ����
   saveas(gcf,['C:\Users\TIAN\Desktop\����д��\������\' grand{n} '.jpg'])
end



for m=1:6
figure
k = squeeze(mean(mean(diff(m,:,8:13,651:751),3),4))
topoplot(k, channel);colorbar();
k=[]
set(gca, 'clim',[-3 3])
end

%%

%�������
x = 1
for m = 1:2:4
    
    DIFF= squeeze(mean(P_DB(:,m,:,:,:),1)) - squeeze(mean(P_DB(:,m+1,:,:,:),1));
    diff(x,:,:,:) = DIFF;
    
    x = x+1
end

save(['E:\����\�ؽ�\eeglab\stft.mat'],'channel','diff','f','t','P_BC','P_DB','P_data')

%����ͼ��������
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
    colormap(jet)    % ѡ����ɫ����
   saveas(gcf,['C:\Users\TIAN\Desktop\����д��\ǰ����\' diff_name{d} '.jpg'])
end

%% ����ͼ
load('E:\����\�ؽ�\channel.mat')
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
title('�Ȳ�:��һ�¼�ȥһ�µ���ͼ','fontsize',20)
%  saveas(gcf,['C:\Users\TIAN\Desktop\11.jpg'])
%�������ͼ
diff_name = {'MC_M_����ͼ ','MC_F_����ͼ','MC_S_����ͼ','MI_M_����ͼ','MI_F_����ͼ','MI_S_����ͼ'}
for i = 1:6
figure
time = [0.5 ,0.7]
fre  =[4,7]
time_ =  dsearchn(xtimes',time')
fre_  =  dsearchn(f',fre')
topoplot(squeeze(mean(mean(diff(2,:,fre_(1):fre_(2),time_(1):time_(2)),3),4)), channel);
colorbar;
set(gca, 'clim',[-1 1])
 saveas(gcf,['C:\Users\TIAN\Desktop\����д��\ǰ����\' diff_name{i} '_��.jpg'])
end

%% ͳ��
freq  = ALL_MC_F_C{1, 1}.freq;
chan  = ALL_MC_F_C{1, 1}.label  ;
time = ALL_MC_F_C{1, 1}.time
% define time window
% timewin      = [0.7 0.9];
timewin      = [0.4 0.7];    %��
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


%% stats across channels

%����Ȥ��ʱƵ����
t_ROI = [0.4 0.6]; f_ROI=[10 15];
t_idx = find((t> t_ROI(1)) & (t < t_ROI(2)));
f_idx = find((f> f_ROI(1)) & (f < f_ROI(2)));
%��ȡ����ȤʱƵ���������������У����
data_test_ch = squeeze(mean(mean(P_BC(:,:,:,f_idx, t_idx),4),5));

% t test (L3 vs L4)
for i=1:EEG.nbchan   %����ÿһ��ͨ��
    data1 = double(squeeze(data_test_ch(:,1,i)));  %��ȡ���б��Ե����������ڵ�i���缫������
    data2 = double(squeeze(data_test_ch(:,2,i)));  %��ȡ���б��Ե��ĸ������ڵ�i���缫������
    [~,p,~,stat] = ttest(data1,data2);  %���T
    P_ttest_ch(i) = p;  %�洢Pֵ��Tֵ
    T_ttest_ch(i) = stat.tstat;
end

%����Tֵ�ĵ���ͼ
figure;
subplot(121);topoplot(T_ttest_ch, EEG.chanlocs); title('T values'); colorbar();
subplot(122);topoplot(P_ttest_ch, EEG.chanlocs); title('P values'); colorbar();


%% �����������������ظ������ķ������
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
%����Fֵ�ĵ���ͼ
figure;
subplot(121);topoplot(F_anova_ch, EEG.chanlocs); title('F values'); colorbar();
subplot(122);topoplot(P_anova_ch, EEG.chanlocs); title('P values'); colorbar(); caxis([0 0.05]);
%% fdr correction to account for multiple comparisons
%�Զ��رȽ�����������������Ե�����
%�����T����Ľ�� ��FDRУ����p_fdr1 �� fdr��������ֵ�� p_masked����ʾpֵ�Ƿ�ͨ��У��
[p_fdr1, p_masked] = fdr(P_ttest, 0.05); %% fdr correction for p values from ttest %0.05��Ҫ��fdr�������������Դ���ĸ��ʿ�����0.05����
% ����pֵͼ��ʱ����ᡢƵ��Ϊ���ᣩ�� ֻ��ʾͨ��У���Ľ��
figure; imagesc(t,f,P_ttest); axis xy; caxis([0 0.05]);
%p_corrected=mafdr(ps, 'BHFDR', 1);%-->madfdr��matlab��fdr�Ľ��������ص��ǽ������pֵ

% �Է�������Ľ����FDR У��
[p_fdr2, p_masked] = fdr(P_anova, 0.05);%% fdr correction for p values from ANOVA
figure; imagesc(t,f,P_anova); axis xy; caxis([0 p_fdr2]); %���û����ֵ��ͨ��fdr�����������ʾԭʼ��pֵ

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

%%  ʱƵͳ��
load('E:\����\�ؽ�\channel.mat')
timewin      = [0.9 1.1];    %��
% timewin      = [0.55 0.75];    %��
timewin_idx  = dsearchn(xtimes', timewin');
freqwin      = [9 13];  % �� band
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
chan_idx = find(chanloc==1)        %��ȡ�缫����
id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'}  % ���Ա��
for cond = 1 :12
    for s = 1:length(id)
        power(s,cond) = squeeze(mean(mean(mean(P_DB(s,cond,chan_idx,freqwin_idx(1):freqwin_idx(2),timewin_idx(1):timewin_idx(2)),3),4),5))
        
              
    end
end





