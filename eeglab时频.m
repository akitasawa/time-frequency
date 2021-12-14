%%  %谱分析
cd( "F:\学习视频\时频\数据")
eeglab
LoadName  = ['1.set'];
ALLEEG = pop_loadset('filename',LoadName);
OUTEEG = pop_select(ALLEEG(1),'time',[0 1]);
[spectra,fres,speccomp,contrib,specstd ] = ...        
                                spectopo(OUTEEG.data,OUTEEG.pnts,OUTEEG.srate, ... 
                                'nfft',250,'winsize',250,'overlap',125,...
                                'plot','off','freqrange',[2 60]);       %计算功率

%地形图                          
figure;
topoplot(mean(spectra(:,16:30),2),OUTEEG.chanlocs,'maplimits','maxmin');     % 16:30为频率的位次，在fres中打开。
h = colorbar;    %颜色轴
set(get(h,'title'),'string','10*log10(\muV^2/HZ)');

%单通道频谱图
figure
[spectra,fres,speccomp,contrib,spcstd ] = ...        
                                spectopo(OUTEEG.data,OUTEEG.pnts,OUTEEG.srate, ... 
                                'nfft',250,'winsize',250,'overlap',125,...
                                'plot','on','chanlocs',EEG.chanlocs,'plotchans',12,'freqrange',[2 60]);  %plotchans 为通道
%%    连续傅里叶变换
Pyy = Y.*conj(Y)/251;
f = 1000/251*(0:127);
plot(f,Pyy(1:128))
title('Power spectral density')
xlabel('Frequency (Hz)')
%%  离散傅里叶变换
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
 %%  %时频分析  
 %创造小波
 clear
 srate = 500;%采样率
 f = 1
 while f<=4
%  f = 60; %小波频率
 time = -1:1/ srate:1;  %时间除以采样率，即这个时间段采样率
 sine_wave = exp(2*pi*1i*f.*time); %正弦波
 %创造高斯曲线
 s = 3/(2*pi*f) ;  %高斯曲线的标准差,3为cycle数
 guassian_win= exp(-time.^2./(2*s^2));
 
 %创造小波
 wavelet = sine_wave .* guassian_win;
h = num2str(f) 
%  %作图
 figure
 subplot(311)
 plot(time,real(sine_wave));    %只显示实部
 title(['sine wave',h])    %正弦波
 subplot(312)
 plot(time,real(guassian_win));    %只显示实部
 title(['guassian win',h])   %高斯曲线
 subplot(313)
 plot(time,real(wavelet));   %只显示实部
 title(['wavelet',h])        %小波
 xlabel('time(ms)')  
 f=f+1
 end
 %% 卷积
clear   %清除Workspace中的变量
clc     %清除Command Window中的命令
uls=ones(1,10); %建立一个1*10的矩阵
Length_u = length(uls); %把向量uls的长度赋给Length_u
hls = exp(-0.1*(1:15)); %建立一个长度为15的向量hls
Length_h = length(hls); %把向量hls的长度赋给Length_h
lmax = max(Length_u,Length_h); %把向量u的长度与向量h的长度中的最大值赋给lmax
%if end 语句确定了nh与nu的值，用于下面的向量u与向量h中，保证两者长度相等
if Length_u>Length_h 
    nu=0; nh = Length_u - Length_h;
elseif Length_u<Length_h 
    nh=0; nu = Length_h - Length_u;
else
    nu=0; nh=0;
end         %nh=0 nu=5
dt = 0.5;
lt = lmax;%把向量u的长度与向量h的长度中的最大值赋给lt
u = [zeros(1,lt),uls,zeros(1,nu),zeros(1,lt)];% 建立一个长度为45的向量，uls的值在中间，易于卷积
t1 = (-lt+1:2*lt)*dt;%建立了一个长度为45，步长为0.5的向量，从-7到15
h = [zeros(1,2*lt),hls,zeros(1,nh)];% 建立一个长度为45的向量，hls的值在末尾一段
hf = fliplr(h);%将h进行反褶运算
y = zeros(1,3*lt);%建立一个1*45的零矩阵
for k = 0:2*lt%设置循环31次
    p = [zeros(1,k),hf(1:end-k)];%p是长度为45的向量，由向量hf平移k个单位而来
    y1 = u.*p*dt;%卷积中的相乘
    yk = sum(y1);%卷积中的积分（求和）
    y(k+lt+1) = yk;%给y中的元素赋值
    subplot(4,1,1);stairs(t1,u)%分割图形窗口为4*1，在第一部分画出要进行卷积的函数u
    axis([-lt*dt,2*lt*dt,min(u),max(u)]),hold on%设置坐标轴的最值，并让图形等待
    ylabel('u(t)')%给y轴命名
    subplot(4,1,2);stairs(t1,p)%在图形窗口的第二部分画出要进行卷积的函数h(k-t)
    axis([-lt*dt,2*lt*dt,min(p),max(p)])%控制坐标轴的最值
    ylabel('h(k-t)')%给y轴命名
    subplot(4,1,3);stairs(t1,y1)%在图形窗口的第三部分画出u(t)*h(k-t)的阶梯状图形
    axis([-lt*dt,2*lt*dt,min(y1),max(y1)+eps])%控制坐标轴的最值
    ylabel('s=u*h(k-t)')%给y轴命名
    subplot(4,1,4);stem(k*dt,yk)%在图形窗口的第四部分画出卷积结果的点状图
    axis([-lt*dt,2*lt*dt,floor(min(y)+eps),ceil(max(y+eps))])%控制坐标轴的最值
    hold on,ylabel('y(k)=sum(s)*dt')%给y轴命名
    pause(1);%每次循环暂停一秒，方便看清各个图形的变化
end

 %%
t = linspace(-10,10,100); %产生一百个元素
y = (square(t) + 1)./2;  %y的函数
subplot(211);         %分割成两块，进入第一块界面
plot(t./(2*pi*10),y,'r-');grid on    %产生10HZ 的方波
axis([0,0.3,-1.2,1.2]);
xlabel('t'),ylabel('y1'),title('10Hz');
subplot(212);        %进入第二块界面
plot(t./(2*pi*16),y,'c-');grid on     %产生16HZ 的方波
axis([0,0.3,-1.2,1.2]);
xlabel('t'),ylabel('y2'),title('16Hz');

%%clear;%清除工作空间的变量
clf;%清除图形
clc;%清除命令窗口中的命令
t0=-1;%给t0赋值1，作为横轴的最小值
tf=5;%给tf赋值5，作为横轴的最大值
dt=0.05;%作为冒号表达式的步长
t1=0;
t=t0:dt:tf; %建立向量
Len_t = length(t);%把向量t的长度赋值给Len_t
n1 = floor((t1-t0)/dt);%选出t=0在向量t中对应的元素序号
 
x1 = zeros(1,Len_t);%建立一个与t等长的一维零矩阵
x1(n1) = 1/dt;%选出t=0在向量t中对应的元素
subplot(2,2,1),stairs(t,x1),grid on %把图形窗口分割成2*2的四个部分，第一个部分用stairs函数画出单位冲击函数
axis([-1,5,0,22])%控制坐标轴，横轴在-1到5之间，纵轴在0到22之间
title('1.冲击信号');%给第一个图形命名
 
% x2 = [zeros(1,n1-1),ones(1,Len_t-n1+1)];
% x2 = (t>0);
 
% x2 = 1/2*(sign(t-0)+1);%利用符号函数实现单位阶跃函数
 
x2 = stepfun(t,t1);%建立一个向量x2，当t<t1时，元素都为0，当t>=t1时，元素都为1
 
subplot(2,2,3),stairs(t,x2),grid on %在图形窗口的第三个部分画出单位阶跃函数
axis([-1,5,0,1.1])  %设置显示的坐标轴的最大最小值
title('2.单位阶跃信号'); %给第二个图形命名
 
alpha = -0.5;%为x3中的alpha赋值
omega = 10;%为x3中的omega赋值
x3 = exp((alpha+j*omega)*t);%产生了一个复指数信号
subplot(2,2,2),plot(t,real(x3)),grid on %在图形窗口的第二部分画出复指数信号的实部
title('3.复指数跃信号（实部）'); %命名
subplot(2,2,4),plot(t,imag(x3)),grid on %在图形窗口中的第四部分画出复指数信号的虚部
title('4.复指数跃信号（虚部）'); %命名

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% 选择时间窗口，频段，ROI，然后平均 %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;
cd('F:\学习视频\时频\wang\matlab_codes')
load('sampleTFR.mat');  % order of dimension: chan*freq*time
% 选择电极点（ROI）
chan2use = {'P1','P3','P5','PO3','PO5','PO7','O1'};
chan_idx = zeros(1,length(chan2use));
for i=1:length(chan2use)       % find the index of channels to use
    ch = strcmpi(chan2use(i), data.chan);
    chan_idx(i) = find(ch);  
end

% 画这个ROI的时频图
figure
contourf(data.time,data.freq,squeeze(mean(data.pow(chan_idx,:,:),1)),40,'linecolor','none') 
set(gca,'xlim',[-0.2 0.5], 'clim',[-5 5])
title('TFR')
xlabel('Time (s)')
ylabel('freq (Hz)')
colorbar
colormap(jet)    % 选择配色方案


 
 
 %%  imagesc()
clear
[x y] = meshgrid(-3:.2:3,-3:.2:3);
z = x.^2 + x.*y+y.^2;
surf(x,y,z);   %曲面图
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
 
 
 
 
 