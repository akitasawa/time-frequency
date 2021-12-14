%%  %�׷���
cd( "F:\ѧϰ��Ƶ\ʱƵ\����")
eeglab
LoadName  = ['1.set'];
ALLEEG = pop_loadset('filename',LoadName);
OUTEEG = pop_select(ALLEEG(1),'time',[0 1]);
[spectra,fres,speccomp,contrib,specstd ] = ...        
                                spectopo(OUTEEG.data,OUTEEG.pnts,OUTEEG.srate, ... 
                                'nfft',250,'winsize',250,'overlap',125,...
                                'plot','off','freqrange',[2 60]);       %���㹦��

%����ͼ                          
figure;
topoplot(mean(spectra(:,16:30),2),OUTEEG.chanlocs,'maplimits','maxmin');     % 16:30ΪƵ�ʵ�λ�Σ���fres�д򿪡�
h = colorbar;    %��ɫ��
set(get(h,'title'),'string','10*log10(\muV^2/HZ)');

%��ͨ��Ƶ��ͼ
figure
[spectra,fres,speccomp,contrib,spcstd ] = ...        
                                spectopo(OUTEEG.data,OUTEEG.pnts,OUTEEG.srate, ... 
                                'nfft',250,'winsize',250,'overlap',125,...
                                'plot','on','chanlocs',EEG.chanlocs,'plotchans',12,'freqrange',[2 60]);  %plotchans Ϊͨ��
%%    ��������Ҷ�任
Pyy = Y.*conj(Y)/251;
f = 1000/251*(0:127);
plot(f,Pyy(1:128))
title('Power spectral density')
xlabel('Frequency (Hz)')
%%  ��ɢ����Ҷ�任
clear
t = 0:1/100:10-1/100;                     % Time vector
x = sin(2*pi*15*t) + sin(2*pi*40*t);      % Signal
y = fft(x);                               % Compute DFT of x
m = abs(y);                               % Magnitude
y(m<1e-6) = 0;
p = unwrap(angle(y));                     % Phase
f = (0:length(y)-1)*100/length(y);        % Frequency vector

subplot(2,1,1)
plot(f,m)
title('Magnitude')
ax = gca;
ax.XTick = [15 40 60 85];

subplot(2,1,2)
plot(f,p*180/pi)
title('Phase')
ax = gca;
ax.XTick = [15 40 60 85];
 %%  %ʱƵ����  
 %����С��
 clear
 srate = 500;%������
 f = 1
 while f<=4
%  f = 60; %С��Ƶ��
 time = -1:1/ srate:1;  %ʱ����Բ����ʣ������ʱ��β�����
 sine_wave = exp(2*pi*1i*f.*time); %���Ҳ�
 %�����˹����
 s = 3/(2*pi*f) ;  %��˹���ߵı�׼��,3Ϊcycle��
 guassian_win= exp(-time.^2./(2*s^2));
 
 %����С��
 wavelet = sine_wave .* guassian_win;
h = num2str(f) 
%  %��ͼ
 figure
 subplot(311)
 plot(time,real(sine_wave));    %ֻ��ʾʵ��
 title(['sine wave',h])    %���Ҳ�
 subplot(312)
 plot(time,real(guassian_win));    %ֻ��ʾʵ��
 title(['guassian win',h])   %��˹����
 subplot(313)
 plot(time,real(wavelet));   %ֻ��ʾʵ��
 title(['wavelet',h])        %С��
 xlabel('time(ms)')  
 f=f+1
 end
 %% ���
clear   %���Workspace�еı���
clc     %���Command Window�е�����
uls=ones(1,10); %����һ��1*10�ľ���
Length_u = length(uls); %������uls�ĳ��ȸ���Length_u
hls = exp(-0.1*(1:15)); %����һ������Ϊ15������hls
Length_h = length(hls); %������hls�ĳ��ȸ���Length_h
lmax = max(Length_u,Length_h); %������u�ĳ���������h�ĳ����е����ֵ����lmax
%if end ���ȷ����nh��nu��ֵ���������������u������h�У���֤���߳������
if Length_u>Length_h 
    nu=0; nh = Length_u - Length_h;
elseif Length_u<Length_h 
    nh=0; nu = Length_h - Length_u;
else
    nu=0; nh=0;
end         %nh=0 nu=5
dt = 0.5;
lt = lmax;%������u�ĳ���������h�ĳ����е����ֵ����lt
u = [zeros(1,lt),uls,zeros(1,nu),zeros(1,lt)];% ����һ������Ϊ45��������uls��ֵ���м䣬���ھ��
t1 = (-lt+1:2*lt)*dt;%������һ������Ϊ45������Ϊ0.5����������-7��15
h = [zeros(1,2*lt),hls,zeros(1,nh)];% ����һ������Ϊ45��������hls��ֵ��ĩβһ��
hf = fliplr(h);%��h���з�������
y = zeros(1,3*lt);%����һ��1*45�������
for k = 0:2*lt%����ѭ��31��
    p = [zeros(1,k),hf(1:end-k)];%p�ǳ���Ϊ45��������������hfƽ��k����λ����
    y1 = u.*p*dt;%����е����
    yk = sum(y1);%����еĻ��֣���ͣ�
    y(k+lt+1) = yk;%��y�е�Ԫ�ظ�ֵ
    subplot(4,1,1);stairs(t1,u)%�ָ�ͼ�δ���Ϊ4*1���ڵ�һ���ֻ���Ҫ���о���ĺ���u
    axis([-lt*dt,2*lt*dt,min(u),max(u)]),hold on%�������������ֵ������ͼ�εȴ�
    ylabel('u(t)')%��y������
    subplot(4,1,2);stairs(t1,p)%��ͼ�δ��ڵĵڶ����ֻ���Ҫ���о���ĺ���h(k-t)
    axis([-lt*dt,2*lt*dt,min(p),max(p)])%�������������ֵ
    ylabel('h(k-t)')%��y������
    subplot(4,1,3);stairs(t1,y1)%��ͼ�δ��ڵĵ������ֻ���u(t)*h(k-t)�Ľ���״ͼ��
    axis([-lt*dt,2*lt*dt,min(y1),max(y1)+eps])%�������������ֵ
    ylabel('s=u*h(k-t)')%��y������
    subplot(4,1,4);stem(k*dt,yk)%��ͼ�δ��ڵĵ��Ĳ��ֻ����������ĵ�״ͼ
    axis([-lt*dt,2*lt*dt,floor(min(y)+eps),ceil(max(y+eps))])%�������������ֵ
    hold on,ylabel('y(k)=sum(s)*dt')%��y������
    pause(1);%ÿ��ѭ����ͣһ�룬���㿴�����ͼ�εı仯
end

 %%
t = linspace(-10,10,100); %����һ�ٸ�Ԫ��
y = (square(t) + 1)./2;  %y�ĺ���
subplot(211);         %�ָ�����飬�����һ�����
plot(t./(2*pi*10),y,'r-');grid on    %����10HZ �ķ���
axis([0,0.3,-1.2,1.2]);
xlabel('t'),ylabel('y1'),title('10Hz');
subplot(212);        %����ڶ������
plot(t./(2*pi*16),y,'c-');grid on     %����16HZ �ķ���
axis([0,0.3,-1.2,1.2]);
xlabel('t'),ylabel('y2'),title('16Hz');

%%clear;%��������ռ�ı���
clf;%���ͼ��
clc;%���������е�����
t0=-1;%��t0��ֵ1����Ϊ�������Сֵ
tf=5;%��tf��ֵ5����Ϊ��������ֵ
dt=0.05;%��Ϊð�ű��ʽ�Ĳ���
t1=0;
t=t0:dt:tf; %��������
Len_t = length(t);%������t�ĳ��ȸ�ֵ��Len_t
n1 = floor((t1-t0)/dt);%ѡ��t=0������t�ж�Ӧ��Ԫ�����
 
x1 = zeros(1,Len_t);%����һ����t�ȳ���һά�����
x1(n1) = 1/dt;%ѡ��t=0������t�ж�Ӧ��Ԫ��
subplot(2,2,1),stairs(t,x1),grid on %��ͼ�δ��ڷָ��2*2���ĸ����֣���һ��������stairs����������λ�������
axis([-1,5,0,22])%���������ᣬ������-1��5֮�䣬������0��22֮��
title('1.����ź�');%����һ��ͼ������
 
% x2 = [zeros(1,n1-1),ones(1,Len_t-n1+1)];
% x2 = (t>0);
 
% x2 = 1/2*(sign(t-0)+1);%���÷��ź���ʵ�ֵ�λ��Ծ����
 
x2 = stepfun(t,t1);%����һ������x2����t<t1ʱ��Ԫ�ض�Ϊ0����t>=t1ʱ��Ԫ�ض�Ϊ1
 
subplot(2,2,3),stairs(t,x2),grid on %��ͼ�δ��ڵĵ��������ֻ�����λ��Ծ����
axis([-1,5,0,1.1])  %������ʾ��������������Сֵ
title('2.��λ��Ծ�ź�'); %���ڶ���ͼ������
 
alpha = -0.5;%Ϊx3�е�alpha��ֵ
omega = 10;%Ϊx3�е�omega��ֵ
x3 = exp((alpha+j*omega)*t);%������һ����ָ���ź�
subplot(2,2,2),plot(t,real(x3)),grid on %��ͼ�δ��ڵĵڶ����ֻ�����ָ���źŵ�ʵ��
title('3.��ָ��Ծ�źţ�ʵ����'); %����
subplot(2,2,4),plot(t,imag(x3)),grid on %��ͼ�δ����еĵ��Ĳ��ֻ�����ָ���źŵ��鲿
title('4.��ָ��Ծ�źţ��鲿��'); %����

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% ѡ��ʱ�䴰�ڣ�Ƶ�Σ�ROI��Ȼ��ƽ�� %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;
cd('F:\ѧϰ��Ƶ\ʱƵ\wang\matlab_codes')
load('sampleTFR.mat');  % order of dimension: chan*freq*time
% ѡ��缫�㣨ROI��
chan2use = {'P1','P3','P5','PO3','PO5','PO7','O1'};
chan_idx = zeros(1,length(chan2use));
for i=1:length(chan2use)       % find the index of channels to use
    ch = strcmpi(chan2use(i), data.chan);
    chan_idx(i) = find(ch);  
end

% �����ROI��ʱƵͼ
figure
contourf(data.time,data.freq,squeeze(mean(data.pow(chan_idx,:,:),1)),40,'linecolor','none') 
set(gca,'xlim',[-0.2 0.5], 'clim',[-5 5])
title('TFR')
xlabel('Time (s)')
ylabel('freq (Hz)')
colorbar
colormap(jet)    % ѡ����ɫ����


 
 
 %%  imagesc()
clear
[x y] = meshgrid(-3:.2:3,-3:.2:3);
z = x.^2 + x.*y+y.^2;
surf(x,y,z);   %����ͼ
box on ;
set(gca,"fontsize",16);
zlabel('z');
xlim([-4 4]);
xlabel("x");
ylim([-4 4]);
ylabel("y");
colorbar;
colormap(jet);
figure;
imagesc(z);
axis square on;
xlabel("x");
ylabel("y");
colorbar;
colormap(jet);
 
 
 
 
 