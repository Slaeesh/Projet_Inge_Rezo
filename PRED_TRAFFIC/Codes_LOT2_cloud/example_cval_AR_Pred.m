% script_SVM_Pred.m

clear all;
BATCHMODE=0;

%--- signals
DATASEL=1;
AGGR=5;
DSA=0;
DDIFF=0;
switch DATASEL
    case 0;
        N=27998;dt=100; % signal length
        NF=20; % NF=20;
        rng(1234,'twister'); b = fir1(NF-1, .5); [d,p0] = lpc(b,NF); u = sqrt(p0)*randn(1,N);
        y = filter(1,d,u);
    case 1
        [A data]=textread('/Users/hwendt/Documents/IRIT/PROJETS/2018/RT-CNES-Prediction/DATA/20180308/instant_flow_500_3000s.txt',['%n %n']); SigStr='flow_500_3000s'; SVMP=1;
        y=data(:)'; dt=100; clear data A
    case 2
        [A data]=textread('/Users/hwendt/Documents/IRIT/PROJETS/2018/RT-CNES-Prediction/DATA/20180308/instant_flow_750_3000s.txt',['%n %n']); SigStr='flow_750_3000s'; SVMP=2;
        y=data(:)'; dt=100; clear data A
end
y0=y;
if DDIFF; y=diff(y); end
[y,ta] = agregate_data(y,AGGR,DSA,dt);
yy=y; y=y-min(y);y=y/max(y); y=y-mean(y); y0=y(:);
N=length(y);

% --- set training and test parameters
Ntrain=500; % training examples
Ntest=N;
NtrainCV=250;
NtestCV=250;
D10=1;       % embedding distance
D20=5;       % D2 - step ahead prediction

D1=D10;D2=D20;
%if DSA==0; D1=AGGR*D10; D2=AGGR*D20; end



%--- AR model order
PP=2:20;


% for plots
iZOOM=2000; nZOOM=100; % for plots



% --- cross validation for model order selection
clear SD* CC*
for pid=1:length(PP)
    [~, SDR,CX] = ARprediction(y,NtrainCV,NtestCV,D1,D2,PP(pid));
    SDtr_cv(pid)=SDR.train;
    SD_cv(pid)  =SDR.test;
    CCtr_cv(pid)=CX.train;
    CC_cv(pid)  =CX.test;
end
% select best model order
[a,optid]=max(SD_cv);
Pstar_cv=PP(optid);optid_cv=optid;
%[a,Pstar]=max(CC);


% --- predict training and test data
for pid=1:length(PP)
    [tmp1, SDR,CX, tmp2] = ARprediction(y,Ntrain,Ntest,D1,D2,PP(pid));
    if pid==optid;
        py=tmp1;ID=tmp2;
    end
    SDtr(pid)=SDR.train;
    SD(pid)  =SDR.test;
    CCtr(pid)=CX.train;
    CC(pid)  =CX.test;
end
% select best model order
[a,optid]=max(SD);
Pstar=PP(optid);
%[a,Pstar]=max(CC);

% --- baseline prediction
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,Pstar_cv,Ntrain,Ntest,D1,D2);
baseline = inputsT(:,1);
tmp=corrcoef(targetsT,baseline); CXbaseline=tmp(2);
tmp=corrcoef(sign(targetsT),sign(baseline)); CXSbaseline=tmp(2);
SDRbase=10*log10(mean(targetsT.^2)/mean((targetsT-baseline).^2));

    % --- plots
    figure(1); clf;
    subplot(211);plot(ta,y,'k-');  grid on;  title('Signal');axis tight;
    [R2] = ACF_fft(y);
    %subplot(212);loglog(ta(1:floor(N/20)),abs(R2(1:floor(N/20))),'k-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
    subplot(212);plot(ta(1:50),(R2(1:50))/R2(1),'k-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
    %
    figure(2);clf;
    subplot(211);
    hold on;
    plot(ta(id{3}),targetsall,'k');
    plot(ta(id{2}),py.test,'b');
    plot(ta(id{1}),py.train,'r');
    xlim(ta([1 N]));
    hold off; grid on;
    title(['AR predict- Order=',num2str(Pstar),', Train=',num2str(Ntrain),', D1=',num2str(D1),', D2=',num2str(D2),'  - SDR=',num2str(SDR.test),'dB - Rho=',num2str(CX.test),'  - SDR base=',num2str(SDRbase),'dB']);
    subplot(212);
    hold on;
    plot(ta(id{3}),targetsall,'k.-');
    plot(ta(id{2}),py.test,'b.-');
    plot(ta(id{1}),py.train,'r');
    xlim(ta(iZOOM+[0 nZOOM-1]));
    hold off; grid on;
    
    SigRho_6_25=mean(abs(R2(6:25)/R2(1)))
    
    if 0;
        figure(3);clf;hold on;
        plot(PP,SD_cv,'b',PP,SD,'r');legend('cross validation SDR','test SDR','location','south');
        plot(PP(optid_cv),SD_cv(optid_cv),'b*',PP(optid),SD(optid),'r*','markersize',12); yl=get(gca,'ylim');ylim([0 yl(2)]);xlim([PP(1) PP(end)]);
        hold off; grid on;ylabel('SDR');xlabel('AR model order');
        title(['AR CV, D1=',num2str(D1),', D2=',num2str(D2),'  - SDR cv=',num2str(SD(optid_cv)),'dB - SDR best=',num2str(SD(optid)),'dB']);
        FPTH = '/Applications/MATLAB/work/IRIT/CONTRACTS/2018/CNES_Prediction/Codes_LOT2/Figures/';
        BASESTR = ['DAT',num2str(DATASEL),'_A',num2str(AGGR),'_DS',num2str(DSA),'_DD',num2str(DDIFF),'_Del',num2str(D1),'_Pred',num2str(D2)];
        hh1=figure(3);FIGNAME=[BASESTR,'_crossval']; set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
    end



