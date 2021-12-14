% %%
clear all;
names = dir(['C:\Users\TIAN\Desktop\ZTT_EEG_DATA1\','*.','vhdr']);
for subi = 1:length(names)
%     EEG = loadcurry(['G:\����\�½��ļ���\Acquisition ' name{subi} '.cdt'],...
%         'CurryLocations', 'False');   %����scan
EEG = pop_loadbv('C:\Users\TIAN\Desktop\ZTT_EEG_DATA1', names(subi).name , [ ] ,[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64]);
    
    EEG=pop_chanedit(EEG, 'lookup',...     % �缫��λ
      'F:\\Matlab\\toolbox\\eeglab\\eeglab2020_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp');
    
    EEG = pop_saveset( EEG, 'filename',[names(subi).name(10) '.set'],'filepath','F:\ZTT_EEG_DATA');   %����·��
    %  STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    fprintf('\n############ completed for subject %s ###########\n\n\n\n', names(subi).name )
end


%% ɾ���缫���缫�嵼���زο����˲�����ȡ����
clear all;
% eeglab;     % ��eeglab
names = dir(['C:\Users\TIAN\Desktop\ZTT_EEG_DATA1\','*.','set']);
Ns = length(names);  % number of subjects
for subi=1:Ns
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = [names(subi).name(1) '_clean.set'];
    
    % load data
    EEG = pop_loadset('filename',LoadName);
    EEG=pop_chanedit(EEG, 'lookup','F:\\Matlab\\toolbox\\eeglab\\eeglab2020_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp');
    %     ɾ����Ч�缫
%     EEG = pop_select( EEG,'nochannel',{'HEO', 'VEO','CB1','CB2','M2'});  % ������Ҫ���Ҫɾ���ĵ缫
     EEG = pop_select( EEG,'nochannel',{'EOG'});  % ������Ҫ���Ҫɾ���ĵ缫
    %     % �缫�嵼
    %     originalEEG = EEG;
    %     EEG = clean_rawdata(EEG, 5, -1, 0.80, -1, 20, 0.35);  % ���Բ����޸ģ���Ҳ���Ը��ݽ���޸�һЩ����
    %     EEG = pop_interp(EEG, originalEEG.chanlocs, 'spherical');   % ����Ҫ�޸�
     
%EEG = pop_rejchan(EEG, 'elec',[1:66] ,'threshold',5,'norm','on','measure','kurt'); %�Զ��ܾ����缫
%pop_rejchan( EEG ) ���ھܾ�
 %�زο�
    %        EEG.data(42,:)=EEG.data(42,:)/2   %M2��һ��
    %        ALLEEG.data(42,:)=EEG.data(42,:)
    %        EEG = pop_reref( EEG, 42);
   EEG = pop_reref( EEG, [30 31] );
    
    % �˲�
    EEG = pop_eegfiltnew(EEG, 'locutoff',1);
    EEG = pop_eegfiltnew(EEG, 'hicutoff',30);
    %       EEG = pop_eegfiltnew(EEG, 'locutoff',2,'hicutoff',48);
    %
    %      % �ز���
%     EEG = eeg_checkset( EEG );
%     EEG = pop_resample( EEG, 500);
    %
    %     % ��ȡȫ������    �ĳ���ʵ����Ҫ�õ�marker���� epoch����ʼʱ��

    EEG = pop_epoch( EEG, {  'S111'  'S112'  'S113'  'S114'  'S115'  'S116'  'S117'  'S118'  'S121'  'S122'  'S123'  'S124'  'S125'  'S126'  'S127'  'S128'  'S131'  'S132'  'S133'  'S134'  'S135'  'S136'  'S137'  'S138'  'S141'  'S142'  'S143'  'S144'  'S145'  'S146'  'S147'  'S148'  'S151'  'S152'  'S153'  'S154'  'S155'  'S156'  'S157'  'S158'  'S161'  'S162'  'S163'  'S164'  'S165'  'S166'  'S167'  'S168'  }, [-0.5 1.5],  'epochinfo', 'yes');
    EEG = pop_selectevent( EEG, 'type',['S 66'] ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_rmbase( EEG, [-300 0] ,[]);
    
    % save
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\�½��ļ���\');
    
    % delete the all of the current datasets from memory
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
%     fprintf('\n############ completed for subject %s ###########\n\n\n\n', id{subi} )
end



%%  ICA
clear all;
eeglab;     % ��eeglab
names = dir(['C:\Users\TIAN\Desktop\ZTT_EEG_DATA1\','*.','set']);
for subi = 2:3:length(names)
    LoadName  = [names(subi).name(1) '_clean.set'];
    SaveName  = [names(subi).name(1) '_ICA.set'];
    
    % load data
    EEG = pop_loadset('filename',LoadName);
    %
    % % %     % run ICA       ���ø�
%         if isfield(EEG.etc, 'clean_channel_mask')  % compute dataRank
%             dataRank = min([rank(double(EEG.data(:,:,1)')) sum(EEG.etc.clean_channel_mask)]);
%         else
%             dataRank = rank(double(EEG.data(:,:,1)'));
%         end
        EEG = pop_runica(EEG, 'extended',1,'interupt','on', 'pca',40);   % run ICA
    
        % save
        EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\ICA\');
    %
    %     %�Զ��ܾ�ica�ɷ�α��
%     EEG = eeg_checkset( EEG );
%     EEG = pop_iclabel(EEG, 'default');
%     EEG = eeg_checkset( EEG );
%     EEG = pop_icflag(EEG, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;NaN NaN]);   %��0.9Ϊ��׼ȥ���۵�ͼ���
%     EEG = pop_subcomp( EEG, [find(EEG.reject.gcompreject==1)], 0);
%     EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','E:\����\flanker\ʱƵ\ica��\');
    % delete the all of the current datasets from memory
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
%     
%     fprintf('\n############ completed for subject %s ###########\n\n\n\n', id{subi})
end



%% �ܾ�����ֵ
clear all;
eeglab;     % ��eeglab
cd('E:\����\�ؽ�\ICA��')
id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21'};

Ns = length(id);

for subi=1:Ns
    LoadName = ['sub' id{subi} '_clear.set'];
    SaveName = ['sub' id{subi} '_pre.set'];
    EEG = pop_loadset('filename',LoadName);
        EEG = eeg_checkset( EEG );
        EEG = pop_iclabel(EEG, 'default');
        EEG = eeg_checkset( EEG );
        EEG = pop_icflag(EEG, [NaN NaN;0.5 1;0.5 1;0.5 1;0.5 1;0.5 1;NaN NaN]);   %��0.9Ϊ��׼ȥ���۵�ͼ���
        EEG = pop_subcomp( EEG, [find(EEG.reject.gcompreject==1)], 0);
    EEG = pop_eegthresh(EEG,1,[1:60] ,-80,80,-0.5,1.498,0,1);
    EEG = pop_saveset( EEG, 'filename','111','filepath','C:\Users\TIAN\Desktop\');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','E:\����\�ؽ�\Ԥ�����������\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    fprintf('\n############ completed for subject %s ###########\n\n\n\n', id{subi})
end

%% ��ȡ�Ƚϵ������������Լ���Ҫ���Ӻͼ��٣�
clear all;
eeglab;     % ��eeglab
names = dir(['C:\Users\TIAN\Desktop\ZTT_EEG_DATA1\','*.','set']);
for subi=2:2:length(names)
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['20��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S113','S114'},'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['20��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S117','S118'},'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['40��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S123','S124' } ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
     LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['40��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{,'S127','S128' } ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['60��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{ 'S133','S134' } ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['60��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{ 'S137','S138' } ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];   
    
    
     
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['80��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S143','S144' } ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['80��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S147','S148' } ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];   
    
    
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['100��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S153','S154'},'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['100��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S157','S158' },'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['180��_��' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S163','S164'} ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    
    LoadName  = [names(subi).name(1) '.set'];
    SaveName  = ['180��_' names(subi).name(1) '_.set'];
    EEG = eeg_checkset( EEG );
    EEG = pop_loadset('filename',LoadName);
    EEG = pop_selectevent( EEG, 'type',{'S167','S168'} ,'deleteevents','off','deleteepochs','on','invertepochs','off');
    EEG = pop_saveset( EEG, 'filename',SaveName,'filepath','F:\ZTT_EEG_DATA\����\');
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
   
%     fprintf('\n############ completed for subject %s ###########\n\n\n\n', id{subi} )
end
%% ��erpͼ
% clear all;
% eeglab
% cd('E:\����\�����ļ�\ICA')
% %�������б���
% EEG = pop_loadset('filename',{'sub1_1_filter.set','sub2_1_filter.set','sub3_1_filter.set','sub4_1_filter.set','sub5_1_filter.set','sub6_1_filter.set'},'filepath','E:\\����\\�����ļ�\\');
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
%
% %��erp
[erp1 erp2 erpsub time sig] = pop_comperp( ALLEEG,1, [1:4],[5:8],'addavg','off','addstd','off','addall','off','subavg','off','suball','off',...
    'diffavg','on','diffstd','off','diffall','off','tplotopt',{'ydir',-1,'legend',{ '20��-��','20��-��','���첨'}},'tlim',[-500 1000],'ylim',[-10 10], ...
    'allerps' ,'off ', 'diffonly','off')
set(axes,'FontSize',14,'XTick',...
    [-500 -400 -300 -200 -100 0 100 200 300 400 500 600 700 800 900 1000]);
saveas(gcf,'E:\����\�����ļ�\MC_M.fig')

% for n = 1:20
%   a(n,1)=mean(mean(ALLEEG(n).data(8,401:501,:)))
% end
% for n = 1:20
%   a(n,2)=mean(mean(ALLEEG(n+20).data(8,401:501,:)))
% end
% dlmwrite('C:\Users\TIAN\Desktop\22.txt',a,'\t')