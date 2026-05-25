
clear all;

addpath(genpath('./'));

BATCHMODE=1;

%--- signals
DATASEL=2;   % data set

AGGR=1;      % aggregation level
DSA=0;       % downsample aggregate
DDIFF=1;     % use derivative
D10=1;       % embedding distance
D20=1;       % D2 - step ahead prediction

% --- set training and test parameters
Ntrain=500; % training examples
NtrainCV=250;
NtestCV=250;

CROSS_PRED=0;

tic
for AGGR=1:5
    AGGR
    for D20=1:10
        D20
        switch DATASEL
            case 0;
                N=27998;dt=100; % signal length
                NF=20; % NF=20;
                rng(1234,'twister'); b = fir1(NF-1, .5); [d,p0] = lpc(b,NF); u = sqrt(p0)*randn(1,N);
                y = filter(1,d,u);clear u
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
        y=y-min(y);y=y/max(y); y=y-mean(y); y0=y(:);
        N=length(y);
        Ntest=N;
        
        
        D1=D10;D2=D20;
        %if DSA==0; D1=AGGR*D10; D2=AGGR*D20; end
        
        SPTH = '/Applications/MATLAB/work/IRIT/CONTRACTS/2018/CNES_Prediction/Codes_LOT2/MATresults/'    ;
        BASESTR = ['DAT',num2str(DATASEL),'_A',num2str(AGGR),'_DS',num2str(DSA),'_DD',num2str(DDIFF),'_Del',num2str(D1),'_Pred',num2str(D2)];
        save([SPTH,BASESTR,'.mat']);
        
%         % AR
%         demo_cval_AR_Pred;
%         % ARMA
%         demo_cval_ARMA_Pred;
        % SVM
        demo_cval_SVM_Pred;
        % AR + SVM
        demo_cval_AR_SVM_Pred;
%         % NN
%         demo_cval_NN_Pred;
%         % AR + NN
%         demo_cval_AR_NN_Pred;
    end
    toc
end