% script_SVM_Pred.m

try BATCHMODE=BATCHMODE; catch BATCHMODE=0; end

if ~BATCHMODE
    clear all;
    BATCHMODE=0;
    
    CROSS_PRED=1; %% PREDICT y{2} from y{1}
    
    %--- signals
    DATASEL=0;  % select data set
    AGGR=0;     % aggregate over windows of size AGGR (0 or 1 = no aggretation)
    DSA=0;      % downsample aggregate by factor DSA  (0 or 1 = no downsampling)
    DDIFF=0;    % work with difference signal
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
    D20=1;       % D2 - step ahead prediction
    
    D1=D10;D2=D20;
    %if DSA==0; D1=AGGR*D10; D2=AGGR*D20; end
    
    %% PREDICT y{2} from y{1}
    if CROSS_PRED
        y0=y;clear y; y{1}=y0;y{2}=y0.^2;
    end
end

%--- set SVM parameters
DDV=[4 6 8 10]; % embedding dimension
MFACT = [0.5 1 1.5 2 3 5 10]; % data scaling
CV = [10 100 1000 10000];
Llambda=1/2000;
Eepsilon = .001;
%Kernel='poly';      Kerneloption = [1 2 3 4 5];
Kernel='gaussian';  Kerneloption = [0.1 0.25 0.5 1 2 3 5 10];






% --- cross validation for SVM parameter selection
clear SD* CC* arSD* arCC* maSD* maCC* i*
ii=1;
params.kernel=Kernel;
for did=1:length(DDV);                              DD=DDV(did);
    for mid=1:length(MFACT);                        Mfact=MFACT(mid);
        for cid=1:length(CV);                       params.C = CV(cid);
            for lid=1:length(Llambda);              params.lambda=Llambda(lid);
                for eid=1:length(Eepsilon);         params.epsilon = Eepsilon(eid);
                    for kid=1:length(Kerneloption); params.kerneloption = Kerneloption(kid);
                        
                        [~, SDR,CX] = SVMprediction(y,NtrainCV,NtestCV,D1,D2,DD,Mfact,params, 1 );
                        SDtr(ii)=SDR.train;
                        SD(ii)  =SDR.test;
                        CCtr(ii)=CX.train;
                        CC(ii)  =CX.test;
                        
                        iDD(ii)=DD;
                        iMM(ii)=Mfact;
                        iCC(ii)=params.C;
                        iLL(ii)=params.lambda;
                        iEE(ii)=params.epsilon;
                        iKK(ii)=params.kerneloption;
                        
                        ii=ii+1;
                    end;end;end;end;end;end

% select best parameter combination
[a,optid]=max(SD);
%[a,Did]=max(CC);
params0=params;clear params;
Dstar=iDD(optid);
Mfstar=iMM(optid);
params.kernel=Kernel;
params.C=iCC(optid);
params.lambda=iLL(optid);
params.epsilon=iEE(optid);
params.kerneloption=iKK(optid);

if ~BATCHMODE
    Mfstar
    params
end


% --- predict training and test data
[py,SDR,CX,ID] = SVMprediction(y,Ntrain,Ntest,D1,D2,Dstar,Mfstar,params);

% --- baseline prediction
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,Dstar,Ntrain,Ntest,D1,D2);
baseline = inputsT(:,1);
tmp=corrcoef(targetsT,baseline); CXbaseline=tmp(2);
tmp=corrcoef(sign(targetsT),sign(baseline)); CXSbaseline=tmp(2);
SDRbase=10*log10(mean(targetsT.^2)/mean((targetsT-baseline).^2));

% for plots
iZOOM=2000; nZOOM=100; % for plots
if ~BATCHMODE
    % --- plots
    if ~iscell(y)
        figure(1); clf;
        subplot(211);plot(ta,y,'k-');  grid on;  title('Signal');axis tight;
        [R2] = ACF_fft(y);
        %subplot(212);loglog(ta(1:floor(N/20)),abs(R2(1:floor(N/20))),'k-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
        subplot(212);plot(ta(1:50),(R2(1:50))/R2(1),'k-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
    else
        figure(1); clf;
        subplot(211);plot(ta,y{1},'k-');  grid on;  title('Input Signal');axis tight;
        subplot(212);plot(ta,y{2},'k-');  grid on;  title('Target Signal');axis tight;
    end
    %
    figure(2);clf;
    subplot(211);
    hold on;
    plot(ta(id{3}),targetsall,'k');
    plot(ta(id{2}),py.test,'b');
    plot(ta(id{1}),py.train,'r');
    xlim(ta([1 N]));
    hold off; grid on;
    title(['SVM predict- Embedding=',num2str(Dstar),', Train=',num2str(Ntrain),', D1=',num2str(D1),', D2=',num2str(D2),'  - SDR=',num2str(SDR.test),'dB - Rho=',num2str(CX.test),'  - SDR base=',num2str(SDRbase),'dB']);
    subplot(212);
    hold on;
    plot(ta(id{3}),targetsall,'k');
    plot(ta(id{2}),py.test,'b');
    plot(ta(id{1}),py.train,'r');
    xlim(ta(iZOOM+[0 nZOOM-1]));
    hold off; grid on;
else
    % --- save
    save([SPTH,BASESTR,'_SVM_',Kernel,'.mat'],'py', 'SDR', 'CX', 'ID', 'baseline', 'CXbaseline', 'SDRbase', 'id', 'targetsall', ...
        'SDtr', 'SD', 'CCtr', 'CC', 'iDD','iMM','Mfact','iCC','iLL','iEE','iKK', 'Dstar', 'Mfstar', 'params');
end

