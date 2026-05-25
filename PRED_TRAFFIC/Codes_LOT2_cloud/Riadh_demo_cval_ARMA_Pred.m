% script_SVM_Pred.m

try BATCHMODE=BATCHMODE; catch BATCHMODE=0; end

if ~BATCHMODE
    clear all;
    BATCHMODE=0;
    
    PPATH = 'D:\prediction_trafic_reseau\data-07-2018\';
    PPATH_aller_500 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-500\default\';
    PPATH_aller_750 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-750\default\';
    PPATH_aller_5000 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-5000\default\';
    PPATH_aller_10000 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-10000\default\';
    PPATH_Users_variable = 'D:\prediction_trafic_reseau\data-07-2018\nb_variable_utilisateurs\scenario1\';
    

    addpath(genpath([PPATH_aller_500,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_aller_750,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_aller_5000,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_aller_10000,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_Users_variable,'Codes_LOT2_cloud']))
    


    %--- signals
    DATASEL=11;  % select data set
    %AGGR=3;     % aggregate over windows of size AGGR (0 or 1 = no aggretation)
    AGGR=1200;
    DSA=0;      % downsample aggregate by factor DSA  (0 or 1 = no downsampling)
    DDIFF=0;    % work with difference signal
    switch DATASEL
        case 0
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
        case 22
            [data]=textread([PPATH,'retour.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=2;
            y=data(:)'; dt=100; clear data A
          
        case 3
            [data]=textread([PPATH_aller_500,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=3;
            y=data(:)'; dt=100; u = length(y);clear data A
           
        case 4
            [data]=textread([PPATH_aller_500,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=4;
            y=data(:)'; dt=100; clear data A
            
        case 5
            [data]=textread([PPATH_aller_750,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=5;
            y=data(:)'; dt=100; clear data A
           
        case 6
            [data]=textread([PPATH_aller_750,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=6;
            y=data(:)'; dt=100; clear data A
        case 7
            [data]=textread([PPATH_aller_5000,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=7;
            y=data(:)'; dt=100; clear data A
        case 8
            [data]=textread([PPATH_aller_5000,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=8;
            y=data(:)'; dt=100; clear data A
        case 9
            [data]=textread([PPATH_aller_10000,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=9;
            y=data(:)'; dt=100; clear data A
        case 10
            [data]=textread([PPATH_aller_10000,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=10;
            y=data(:)'; dt=100; clear data A
        case 11
            [data]=textread([PPATH_Users_variable,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=6;
            y=data(:)'; dt=100; clear data A
            
    end
    y0=y;
    if DDIFF; y=diff(y); end
    [y,ta] = agregate_data(y,AGGR,DSA,dt);
    if ~iscell(y)
        ymin=min(y); y=y-min(y); ymax=max(y);y=y/max(y);ymean=mean(y); y=y-mean(y); y0=y(:);
        N=length(y);
    else
        for i=1:length(y)
            %y{i}=y{i}-min(y{i});y{i}=y{i}/max(y{i}); y{i}=y{i}-mean(y{i}); y0{i}=y{i}(:);
            ymin(i)=min(y{i});y{i}=y{i}-min(y{i});ymax(i)=max(y{i});y{i}=y{i}/max(y{i}); ymean(i)=mean(y{i});y{i}=y{i}-mean(y{i}); y0{i}=y{i}(:);
            N=length(y{i});
        end
    end     
%     ymean = mean(y);ymax = max(y);ymin = min(y);
%     yy=y; y=y-min(y);y=y/max(y); y=y-mean(y); y0=y(:);
%     N=length(y);
    
    
    % --- set training and test parameters
    Ntrain=10000; % training examples
    Ntest=N;
    NtrainCV=5000;
    NtestCV=5000;
    D10=1;       % embedding distance
    D20=1200;       % D2 - step ahead prediction
    %D20=1200;
    %D20 = 12;
    D1=D10;D2=D20;
    %if DSA==0; D1=AGGR*D10; D2=AGGR*D20; end
end

%--- AR and MA model orders
%PP=2:20;
PP=10:20;
QQ=2:10;


% for plots
iZOOM=2000; nZOOM=100; % for plots


% --- cross validation for model order selection: AR
clear SD* CC* arSD* arCC* maSD* maCC*
for pid=1:length(PP)
    [~, SDR,CX] = ARprediction(y,NtrainCV,NtestCV,D1,D2,PP(pid));
    arSDtr(pid)=SDR.train;
    arSD(pid)  =SDR.test;
    arCCtr(pid)=CX.train;
    arCC(pid)  =CX.test;
end
% select best model order
[a,optid]=max(arSD);
Pstar=PP(optid);
%[a,Pstar]=max(arCC);

% --- cross validation for model order selection: MA
for qid=1:length(QQ)
    [~, SDR,CX] = ARMAprediction(y,NtrainCV,NtestCV,D1,D2,Pstar,QQ(qid));
    maSDtr(qid)=SDR.train;
    maSD(qid)  =SDR.test;
    maCCtr(qid)=CX.train;
    maCC(qid)  =CX.test;
end
% select best model order
[a,optid]=max(maSD);
Qstar=QQ(optid);
%[a,Qstar]=max(maCC);


% --- predict training and test data
[py,SDR,CX,ID] = ARMAprediction(y,Ntrain,Ntest,D1,D2,Pstar,Qstar);

% --- baseline prediction
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,Pstar,Ntrain,Ntest,D1,D2);
baseline = inputsT(:,1);
tmp=corrcoef(targetsT,baseline); CXbaseline=tmp(2);
tmp=corrcoef(sign(targetsT),sign(baseline)); CXSbaseline=tmp(2);
SDRbase=10*log10(mean(targetsT.^2)/mean((targetsT-baseline).^2));

% --- compute average absolute errors and MAD.

%  targetsTr_prediction=(y(id{2})+ymean)*ymax+ymin; 
%  xtmp_prediction=(py.test+ymean)*ymax+ymin; 
targetsTr_prediction = y(id{2});
xtmp_prediction = py.test;
AbsError=mean(abs(targetsTr_prediction-xtmp_prediction)); 
xerr=targetsTr_prediction-xtmp_prediction;
MAD_prediction = 100*(sum(abs(targetsTr_prediction-xtmp_prediction))/sum(abs(targetsTr_prediction)));

% targetsTr_baseline=(targetsT+ymean)*ymax+ymin; 
% xtmp_baseline=(baseline+ymean)*ymax+ymin; 
targetsTr_baseline=targetsT; 
xtmp_baseline= baseline;
AbsErrorbase=mean(abs(targetsTr_baseline-xtmp_baseline)); 
xerrbase=targetsTr_baseline-xtmp_baseline;
MAD_baseline = 100*(sum(abs(targetsTr_baseline-xtmp_baseline))/sum(abs(targetsTr_baseline)));
%MAD_baseline = mad(xtmp);
if ~BATCHMODE
    disp('Prediction:');
    SDRtest=SDR.test,AbsError,MAD_prediction
    disp('Baseline:');
    SDRbase,AbsErrorbase,MAD_baseline 
    % --- plots
    if ~iscell(y)
    figure(1); clf;
    subplot(211);plot(ta,y,'k-');  grid on;  title('Signal');axis tight;
    [R2] = ACF_fft(y);
    %subplot(212);loglog(ta(1:floor(N/20)),abs(R2(1:floor(N/20))),'k-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
    subplot(212);plot(ta(1:50),(R2(1:50))/R2(1),'k-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
    end
    %
    figure(2);clf;
%     subplot(211);
    hold on;
    plot(ta(id{3}),targetsall,'k');
    plot(ta(id{2}),py.test,'b');
    plot(ta(id{1}),py.train,'r');
    legend('input data','test data','training data');
    xlim(ta([1 N]));
    xlabel('time (ms)');
    ylabel('normalized and mean centering data');
    hold off; grid on;
    title(['ARMA predict- Order=',num2str(Pstar),'/',num2str(Qstar),', Train=',num2str(Ntrain),', D1=',num2str(D1),', D2=',num2str(D2),'  - SDR=',num2str(SDR.test),' ,AGGR = ',num2str(AGGR),'dB - Rho=',num2str(CX.test),'  - SDR base=',num2str(SDRbase),'dB']);
%     subplot(212);
%     hold on;
%     plot(ta(id{3}),targetsall,'k');
%     plot(ta(id{2}),py.test,'b');
%     plot(ta(id{1}),py.train,'r');
%     xlim(ta(iZOOM+[0 nZOOM-1]));
%     hold off; grid on;
else
    % --- save
    save([SPTH,BASESTR,'_ARMA.mat'],'py', 'SDR', 'CX', 'ID', 'baseline', 'CXbaseline', 'SDRbase', 'id', 'targetsall', ...
        'arSDtr', 'arSD', 'arCCtr', 'arCC', 'maSDtr', 'maSD', 'maCCtr', 'maCC', 'PP', 'Pstar', 'QQ', 'Qstar');
end

