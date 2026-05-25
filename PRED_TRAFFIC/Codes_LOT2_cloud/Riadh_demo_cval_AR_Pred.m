% script_SVM_Pred.m

try BATCHMODE=BATCHMODE; catch BATCHMODE=0; end

if ~BATCHMODE
    clear all;
    BATCHMODE=0;
    
    %PPATH='C:\Users\uayesta\Dropbox\RT-CNES-HW-JYT\data-06-2018
    PPATH = 'C:\Riadh\PREDICTION\cnes\RT-CNES-HW-JYT\Codes_LOT2_cloud\';
    PPATH_aller_500 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-500\default\';
    PPATH_aller_750 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-750\default\';
    PPATH_aller_5000 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-5000\default\';
    PPATH_aller_10000 = 'D:\prediction_trafic_reseau\data-07-2018\simulation_aller\multi-app-simulator-10000\default\';
    PPATH_Users_variable = 'C:\Riadh\PREDICTION\cnes\nb_variable_utilisateurs\scenario2\';
    %PPATH='/Users/hwendt/Dropbox/RT-CNES-HW-JYT/data-06-2018/';
    
    %addpath(genpath('C:\Users\uayesta\Dropbox\RT-CNES-HW-JYT\data-06-2018\Codes_LOT2_cloud'))
    addpath(genpath([PPATH,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_aller_500,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_aller_750,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_aller_5000,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_aller_10000,'Codes_LOT2_cloud']))
    addpath(genpath([PPATH_Users_variable,'Codes_LOT2_cloud']))
    
    
    CROSS_PRED=1; %% PREDICT y{2} from y{1}
    
    %--- signals
    DATASEL=13;  % select data set   
    AGGR=12000;     % aggregate over windows of size AGGR (0 or 1 = no aggretation)
   % AGGR = 10;
    %AGGR = 1200;
     
    DSA=0;      % downsample aggregate by factor DSA  (0 = no downsampling)
    
    DDIFF=0;    % work with difference signal
    switch DATASEL
        case 0;
            N=27998;dt=100; % signal length
            NF=20; % NF=20;
            rng(1234,'twister'); b = fir1(NF-1, .5); [d,p0] = lpc(b,NF); u = sqrt(p0)*randn(1,N);
            y = filter(1,d,u);
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt); 
        case 1
            [data]=textread([PPATH,'aller.txt'],['%n']); SigStr='flow_500_3000s'; SVMP=1;
            y=data(:)'; dt=100; clear data A
            u = y;
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 2
            [data]=textread([PPATH,'retour.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=2;
            y=data(:)';dt=100; clear data A
            u = y;
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt); 
            v = y;
        case 3
            [data]=textread([PPATH_aller_500,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=3;
            y=data(:)'; dt=100;  clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 4
            [data]=textread([PPATH_aller_500,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=4;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 5
            [data]=textread([PPATH_aller_750,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=5;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
            v = y;
        case 6
            [data]=textread([PPATH_aller_750,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=6;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
%         case 6
%             [data]=textread([PPATH_aller_750,'copie.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=6;
%             y=data(:)'; dt=100; clear data A
%             if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 7
            [data]=textread([PPATH_aller_5000,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=7;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 8
            [data]=textread([PPATH_aller_5000,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=8;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 9
            [data]=textread([PPATH_aller_10000,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=9;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 10
            [data]=textread([PPATH_aller_10000,'tx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=10;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 11
            [data]=textread([PPATH_Users_variable,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=6;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);
        case 13
            [data]=textread([PPATH_Users_variable,'rx_throughput.txt'],['%n']); SigStr='flow_750_3000s'; SVMP=11;
            y=data(:)'; dt=100; clear data A
            if DDIFF; y=diff(y); end;[y,ta] = agregate_data(y,AGGR,DSA,dt);

        
    end
    y0=y;
    yy=y; 
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
   
    % --- set training and test parameters
    Ntrain=10000; % training examples
    Ntest=N;
    NtrainCV=5000;
    NtestCV=5000;
    D10=1;       % embedding distance
    
    %D20=12;       % D2 - step ahead prediction
    D20 = 1200;
    %D20 = 1200;
    D1=D10;D2=D20;
    %if DSA==0; D1=AGGR*D10; D2=AGGR*D20; end
    
%     %% PREDICT y{2} from y{1}
%     if CROSS_PRED
%         y0=y;clear y; y{1}=y0;y{2}=y0.^2;
%     end
    
end


%--- AR model order
PP=10:20;
%PP = [100 200];
%PP=4:20;
%PP=;
%PP=24;


% for plots
iZOOM=10000; nZOOM=1000; % for plots


%y0=y;clear y; y{1}=y0;y{2}=y0*0.05;%y{2}=y{2}-mean(y{2});y{2}=y{2}/std(y{2})*std(y{1});

% --- cross validation for model order selection
clear SD* CC*

for pid=1:length(PP)
    [~, SDR,CX] = ARprediction(y,NtrainCV,NtestCV,D1,D2,PP(pid));
    SDtr(pid)=SDR.train;
    SD(pid)  =SDR.test;
    CCtr(pid)=CX.train;
    CC(pid)  =CX.test;
end




% select best model order
[a,optid]=max(SD);
Pstar=PP(optid);
Pstar;
%[a,Pstar]=max(CC);


% --- predict training and test data
[py,SDR,CX,ID] = ARprediction(y,Ntrain,Ntest,D1,D2,Pstar);
ww = py.train;

% --- baseline prediction
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,Pstar,Ntrain,Ntest,D1,D2);
baseline = inputsT(:,1);
tmp=corrcoef(targetsT,baseline); CXbaseline=tmp(2);
tmp=corrcoef(sign(targetsT),sign(baseline)); CXSbaseline=tmp(2);
SDRbase=10*log10(mean(targetsT.^2)/mean((targetsT-baseline).^2));
 
%coef_corr_baseline = sum((targetsT-mean(targetsT)).*(baseline-mean(baseline)))/length(targetsT)*sqrt(var(targetsT))*sqrt(var(baseline));

% --- compute average absolute errors and MAD.
targetsTr_prediction=(y(id{2})+ymean)*ymax+ymin; 
%targetsTr_prediction = y(id{2});
xtmp_prediction = py.test;
%xtmp_prediction=(py.test+ymean)*ymax+ymin; 
AbsError=mean(abs(targetsTr_prediction-xtmp_prediction)); 
xerr=targetsTr_prediction-xtmp_prediction;
MAD_prediction = 100*(mean(abs(targetsTr_prediction-xtmp_prediction))/mean(abs(targetsTr_prediction)));


 %targetsTr_baseline=(targetsT+ymean)*ymax+ymin; 
 %xtmp_baseline=(baseline+ymean)*ymax+ymin;
 targetsTr_baseline=targetsT; 
  xtmp_baseline= baseline; 

AbsErrorbase=mean(abs(targetsTr_baseline-xtmp_baseline)); xerrbase=targetsTr_baseline-xtmp_baseline;
AbsErrorbase_relative=mean(abs(targetsTr_baseline-xtmp_baseline))/mean(abs(targetsTr_baseline));
MAD_baseline = 100*(mean(abs(targetsTr_baseline-xtmp_baseline))/mean(abs(targetsTr_baseline)));
%MAD_baseline_matlab = mad(xtmp);
%coef_corr_prediction = sum((targetsTr-mean(targetsTr)).*(xtmp-mean(xtmp)))/length(targetsTr)*sqrt(var(targetsTr))*sqrt(var(xtmp));
if ~BATCHMODE
%         figure(4)
%         plot(PP,SD,'k-','Linewidth',2);
%         hold on;
%         plot(PP,CC,'r-','Linewidth',2);
%         title('La validation croisée pour la méthode AR.');
%         legend('cross validation','SDR test');
%         ylabel('SDR');
%         xlabel('Ordre P');
%         grid on;
        
        
    
    disp('Prediction:');
    SDRtest=SDR.test,AbsError,MAD_prediction%coef_corr_prediction
    disp('Baseline:');
    SDRbase,AbsErrorbase,MAD_baseline %coef_corr_baseline
    
    figure(5);
    plot(targetsTr_prediction,'k');
    hold on;
    plot(xtmp_prediction,'b')
    
    figure(6)
    plot(targetsTr_prediction-xtmp_prediction,'r');
     
     figure(111);clf;
     subplot(221); plot(ta(id{2}),xerr);axis tight;ylabel('absolute error');grid on;
     subplot(222); hist(xerr,50);grid on;
     subplot(223); plot(ta(id{2}),xerrbase);axis tight;ylabel('absolute error baseling');grid on;
     subplot(224); hist(xerrbase,50);grid on;

    

    % --- plots
    if ~iscell(y)
        figure(1); clf;
        subplot(211);plot(ta,y,'ko-');  grid on;  title('Signal');axis tight;
        [R2] = ACF_fft(y);
        %subplot(212);loglog(ta(1:floor(N/20)),abs(R2(1:floor(N/20))),'k-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
        subplot(212);plot(ta(1:50),(R2(1:50))/R2(1),'ko-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
    end
    %
    figure(2);clf;
    %subplot(211);
    hold on;
    plot(ta(id{3}),targetsall,'k');
    plot(ta(id{2}),py.test,'b');
    plot(ta(id{1}),py.train,'r');
    %legend('cross validation','SDR test');
    legend('input data','test data','training data');
    xlim(ta([1 N]));
    xlabel('time (ms)');
    ylabel('normalized and mean centering data');
    hold off; grid on;
    
    %title(['AR predict- Order=',num2str(Pstar),', Train=',num2str(Ntrain),', D1=',num2str(D1),', D2=',num2str(D2),'  - SDR=',num2str(SDR.test),'dB - Rho=',num2str(CX.test),'  - SDR base=',num2str(SDRbase),'dB']);
    title(['AR predict- Order=',num2str(Pstar),', Train=',num2str(Ntrain),', D1=',num2str(D1),', AGGR=',num2str(AGGR),', D2=',num2str(D2),'  - SDR=',num2str(SDR.test),'dB - Rho=',num2str(CX.test),'  - SDR base=',num2str(SDRbase),'dB']);
    
%    subplot(212);
%    hold on;
%     plot(ta(id{3}),targetsall,'k.-');
%     plot(ta(id{2}),py.test,'b.-');
%     plot(ta(id{1}),py.train,'r');
%     xlim(ta(iZOOM+[0 nZOOM-1]));
%     hold off; 
  grid on;
    
    %SigRho_6_25=mean(abs(R2(6:25)/R2(1)))
    
    if 0;
        figure(3);clf;hold on;
        plot(ta(id{3})/AGGR,targetsall,'k.-');
        plot(ta(id{2})/AGGR,py.test,'b.-');
        plot(ta(id{1})/AGGR,py.train,'r');
        xlim(ta(iZOOM+[0 nZOOM-1])/AGGR);
        hold off; grid on;
        title(['AR ',num2str(Pstar),', D1=',num2str(D1),', D2=',num2str(D2),'  - SDR=',num2str(SDR.test),'dB - SDR base=',num2str(SDRbase),'dB']);
        FPTH = '/Applications/MATLAB/work/IRIT/CONTRACTS/2018/CNES_Prediction/Codes_LOT2/Figures/';
        BASESTR = ['DAT',num2str(DATASEL),'_A',num2str(AGGR),'_DS',num2str(DSA),'_DD',num2str(DDIFF),'_Del',num2str(D1),'_Pred',num2str(D2)];
        hh1=figure(3);xlabel('t (ms)');FIGNAME=[BASESTR,'_data']; set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
    end
else
    % --- save
    save([SPTH,BASESTR,'_AR.mat'],'py', 'SDR', 'CX', 'ID', 'baseline', 'CXbaseline', 'SDRbase', 'id', 'targetsall', ...
        'SDtr', 'SD', 'CCtr', 'CC', 'PP', 'Pstar');
end


