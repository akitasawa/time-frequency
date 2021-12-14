%%
clear all;
id = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','19','21'};
Cond = [6,8,2,3,4,9,1,3,5,7,1,2,60,80,20,30,40,90,10,30,50,70,10,20]
%% 短时傅里叶变换
tic  
for s=1:length(id)
    setname=(['sub' id{s} '_pre.set']);
    setpath='E:\数据\重建\预处理完的数据';
    EEG= pop_loadset('filename',setname,'filepath',setpath);
    EEG= eeg_checkset( EEG );
    m=1
    for j=1:length(Cond)/2
        EEG_new = pop_epoch( EEG, {Cond(m),Cond(m+1)}, [-0.5  1.2], 'newname', 'Merged datasets pruned with ICA   epochs epochs', 'epochinfo', 'yes');
        EEG_new = eeg_checkset( EEG_new );
        EEG_new     = pop_rmbase( EEG_new, [-200    0]);
        m = m+2
        for chan = 1:length( EEG_new .chanlocs)
            chan2use =  EEG_new .chanlocs(chan).labels;
            
            % EEG= pop_loadset('MC_M_C1_.set');
            timewin        = 400; % 窗的大小
            times2save     = -500:20:1000; % 时间范围
            channel2plot   = chan2use;  %电极点
            % frequency2plot = 15;  % in Hz   %
            % timepoint2plot = 200; % ms
            
            %搜索参数的位置
            times2saveidx = zeros(size(times2save));
            for i=1:length(times2save)
                [junk,times2saveidx(i)]=min(abs(EEG_new.times-times2save(i)));
            end
            timewinidx = round(timewin/(1000/EEG_new.srate));
            chan2useidx = strcmpi(channel2plot,{ EEG_new .chanlocs.labels});
            
            
            % 创建汉宁窗
            hann_win = .5*(1-cos(2*pi*(0:timewinidx-1)/(timewinidx-1)));
            
            % 设置频率大小
            frex = linspace(0, EEG_new .srate/2,floor(timewinidx/2)+1);
            
            % 时频矩阵
            tf = zeros(length(frex),length(times2save));
            %基线
            % baseidx = dsearchn( EEG_new .times',[-200 0]');
            % 执行短时傅里叶变换
            for timepointi=1:length(times2save)
                
                % extract time series data for this center time point
                tempdat = squeeze( EEG_new .data(chan2useidx,times2saveidx(timepointi)-floor(timewinidx/2):times2saveidx(timepointi)+floor(timewinidx/2)-mod(timewinidx+1,2),:)); % note: the 'mod' function here corrects for even or odd number of points
                
                % taper data (using bsxfun instead of repmat... note sizes of tempdat
                % and hann_win)
                taperdat = bsxfun(@times,tempdat,hann_win');
                
                fdat = fft(taperdat,[],1)/timewinidx; % 3rd input is to make sure fft is over time
                tf(:,timepointi) = mean(abs(fdat(1:floor(timewinidx/2)+1,:)).^2,2); % average over trials'
                
                baselinetime = [ -200 0]; % in ms
                
                % convert baseline window time to indices
                [~,baselineidx(1)]=min(abs(times2save -baselinetime(1)));
                [~,baselineidx(2)]=min(abs(times2save -baselinetime(2)));
                
                % dB-correct
                baseline_power = mean(tf(:,baselineidx(1):baselineidx(2)),2);
                dbconverted = 10*log10( bsxfun(@rdivide,tf,baseline_power));
                
            end
            chanpower(chan,:,:) = dbconverted(:,:);
        end
        cond(j,:,:,:) = chanpower(:,:,:)
    end
    TF(s,:,:,:,:) = cond(:,:,:,:)
end
toc
%%
% plot
figure
contourf(times2save,frex,squeeze(mean(all,1)),40,'linecolor','none')
set(gca,'ylim',[1 30],'clim',[-5 5])
set(gca,'ylim',[1 30])
title([ 'Sensor ' channel2plot ', power plot ( baseline correction)' ])
disp([ 'Overlap of ' num2str(100*(1-mean(diff(times2save))/timewin)) '%' ])