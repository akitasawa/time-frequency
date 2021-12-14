clear all;
id = { '002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017',...
    '018'	,'019',	'020'	,'021',	'022',	'023'	,'024'	,'025',	'026',	'027'	,'028'	,'029'	,'030',	'031',	'032',...
    '033'	,'034',	'040'	,'041'	,'042',	'043'	,'044'	,'045'	,'046',	'047',	'048'	,'049',...
    '050',	'051'	,'052'	,'053'};
Cond= { {'S 11'}  {'S 22'}  {'S 33'}  {'S 36'} }
%%
cd 'H:\CSPCE\去伪迹后'
for i=1:length(id)
    tic
    setname=([ id{i} '.set']);
    setpath='H:\CSPCE\去伪迹后';
    EEG_new= pop_loadset('filename',setname,'filepath',setpath);
    EEG_new= eeg_checkset(EEG_new );
    for j=1:length(Cond)
        EEG= pop_epoch( EEG_new,Cond{1, j}, [-0.8  1.2], 'newname', 'Merged datasets pruned with ICA   epochs epochs', 'epochinfo', 'yes');
        EEG= eeg_checkset( EEG);
        for chan = 1:length(EEG.chanlocs)
            chan2use = EEG.chanlocs(chan).labels;
            
            min_freq =  2;
            max_freq = 40;
            num_frex = 30;
            
            % define wavelet parameters
            time = -1:1/EEG.srate:1;
            frex = logspace(log10(min_freq),log10(max_freq),num_frex);
            s    = logspace(log10(3),log10(10),num_frex)./(2*pi*frex);
            % s    =  3./(2*pi*frex); % this line is for figure 13.14
            % s    = 10./(2*pi*frex); % this line is for figure 13.14
            
            % definte convolution parameters
            n_wavelet            = length(time);
            n_data               = EEG.pnts*EEG.trials;
            n_convolution        = n_wavelet+n_data-1;
            n_conv_pow2          = pow2(nextpow2(n_convolution));
            half_of_wavelet_size = (n_wavelet-1)/2;
            
            % get FFT of data
            eegfft = fft(reshape(EEG.data(strcmpi(chan2use,{EEG.chanlocs.labels}),:,:),1,EEG.pnts*EEG.trials),n_conv_pow2);
            
            % initialize
            eegpower = zeros(num_frex,EEG.pnts); % frequencies X time X trials
            
            baseidx = dsearchn(EEG.times',[-200 0]');
            
            % loop through frequencies and compute synchronization
            for fi=1:num_frex
                
                wavelet = fft( sqrt(1/(s(fi)*sqrt(pi))) * exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*(s(fi)^2))) , n_conv_pow2 );
                
                % convolution
                eegconv = ifft(wavelet.*eegfft);
                eegconv = eegconv(1:n_convolution);
                eegconv = eegconv(half_of_wavelet_size+1:end-half_of_wavelet_size);
                
                % Average power over trials (this code performs baseline transform,
                % which you will learn about in chapter 18)
                temppower = mean(abs(reshape(eegconv,EEG.pnts,EEG.trials)).^2,2);
%                 eegpower(fi,:) = 10*log10(temppower./mean(temppower(baseidx(1):baseidx(2))));
                p_data(i,j,chan,fi,:) = temppower;
            end
        end
    end
    toc
    fprintf('\n############ completed for subject %s ###########\n\n\n\n',id{i} )
    waitbar(i/length(id))
end
save('H:\CSPCE\all.mat','times','frex','p_data','P_DB')


%% baseline correction
% 对于所有被试、条件、通道、频率，都做基线校正
t_pre_idx=find((times>=-0.2)&(times<=0))
for i=1:size(p_data,1)%所有被试
    for j=1:size(p_data,2)%所有条件
        for ii=1:size(p_data,3)%所有通道
            for jj=1:size(p_data,4)%所有频率
                temp_data=squeeze(p_data(i,j,ii,jj,:));
%                                 P_CC(i,j,ii,jj,:)=temp_data-mean(temp_data(t_pre_idx))  ;    %absolute法
                P_DB(i,j,ii,jj,:) = 10*log10( bsxfun(@rdivide, temp_data,mean(temp_data(t_pre_idx))));  %DB法
            end
        end
    end
end
%%
figure
subplot(121)
contourf(EEG.times,frex,squeeze(mean(all,1)),40,'linecolor','none')
set(gca,'clim',[-3 3],'xlim',[-200 1000],'yscale','log','ytick',logspace(log10(min_freq),log10(max_freq),6),'yticklabel',round(logspace(log10(min_freq),log10(max_freq),6)*10)/10)
title('Logarithmic frequency scaling')

figure
contourf(times,frex,squeeze(mean(all(:,54,:,:),1)),40,'linecolor','none')
set(gca,'clim',[-5 5],'xlim',[-400 1000],'ylim',[2,40])
title('POZ')

imagesc(EEG.times,frex,squeeze(mean(all,1)));
set(gca,'clim',[-3 3],'xlim',[-200 1000],'ylim',[2,45],'ydir','normal')
title('Linear frequency scaling')



%%
figure
% % contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,1:4,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3)),40,'linecolor','none')
% contourf(xtimes, f,squeeze(mean(mean(mean(P_DB(:,3,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3))-squeeze(mean(mean(mean(P_DB(:,4,[44;45;46;47;48;51;53;54;55;57;58;59;60],:,:),1),2),3)),40,'linecolor','none')
contourf(times, frex,squeeze(mean(mean(mean(P_DB(:,[2,4],[19],:,:),1),2),3))-squeeze(mean(mean(mean(P_DB(:,[1,3],[19],:,:),1),2),3)),40,'linecolor','none')
% contourf(times, frex,squeeze(mean(mean(mean(P_DB(:,[1,3],[19],:,:),1),2),3)),40,'linecolor','none')
set(gca,'xlim',[-600 1000], 'clim',[-1 1])
title('前额叶:大多数不一致的冲突','fontsize',20)
xlabel('时间(S)','fontsize',15)
ylabel('频率(Hz)','fontsize',15)
colorbar




