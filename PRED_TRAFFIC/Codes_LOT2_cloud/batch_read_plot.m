
clear all;

addpath(genpath('./'));
%addpath('/Applications/MATLAB/work/');

BATCHMODE=1;

%--- signals
DATASEL=2;   % data set

AGGR=1;      % aggregation level
DSA=0;       % downsample aggregate
DDIFF=1;     % use derivative
D10=1;       % embedding distance
D20=5;       % D2 - step ahead prediction

% --- set training and test parameters
Ntrain=500; % training examples
NtrainCV=250;
NtestCV=250;

MSTR{1}='_AR';
MSTR{2}='_ARMA';
MSTR{3}='_SVM_poly';
MSTR{4}='_SVM_gaussian';
MSTR{5}='_NN';
MSTR{6}='_AR_NN';
MSTR{7}='_AR_SVM_poly';
MSTR{8}='_AR_SVM_gaussian';

TSTR{1}='AR';
TSTR{2}='ARMA';
TSTR{3}='SVM poly';
TSTR{4}='SVM gaussian';
TSTR{5}='NN';
TSTR{6}='AR NN';
TSTR{7}='AR SVM poly';
TSTR{8}='AR SVM gaussian';


MID=1:8;[5];% 2 5 6];

aggrV=1:5;
d2V=1:10;

PRINTHW=1;

SPTH = '/Applications/MATLAB/work/IRIT/CONTRACTS/2018/CNES_Prediction/Codes_LOT2/MATresults/';
FPTH = '/Applications/MATLAB/work/IRIT/CONTRACTS/2018/CNES_Prediction/Codes_LOT2/Figures/';

for mid = 1:length(MID);
    for AGGR=aggrV % aggregation level
        for D20=d2V % D2 - step ahead prediction
            D1=D10;D2=D20;
            %if DSA==0; D1=AGGR*D10; D2=AGGR*D20; end
            
            
            BASESTR = ['DAT',num2str(DATASEL),'_A',num2str(AGGR),'_DS',num2str(DSA),'_DD',num2str(DDIFF),'_Del',num2str(D1),'_Pred',num2str(D2)];
            d1=load([SPTH,BASESTR,'.mat']);
            d2=load([SPTH,BASESTR,MSTR{MID(mid)},'.mat']);
            
            SDR{MID(mid)}(AGGR,D20)=d2.SDR.test;
            CX{MID(mid)}(AGGR,D20)=d2.CX.test;
            CXS{MID(mid)}(AGGR,D20)=d2.CX.testS;
            L2str{D20}=[num2str(D20),'-step (*',num2str(d1.dt),'ms)'];
            SDRbase{MID(mid)}(AGGR,D20)=d2.SDRbase;
            CXbase{MID(mid)}(AGGR,D20)=d2.CXbaseline;
            %CXSbase(AGGR,D20)=d2.CXSbaseline;
            if mid==1 & AGGR==1 & D20==d2V(1)
                iZOOM=2000; nZOOM=100;
                figure(111); clf;plot(d1.ta,d1.y,'k-');  grid on;  title('Signal');axis tight;
                figure(112); clf;plot(d1.ta,d1.y,'k-');  grid on;  title('Signal (zoom)');axis tight;xlim(d1.ta(iZOOM+[0 nZOOM-1]));
                [R2] = ACF_fft(d1.y);
                figure(113); clf; plot(d1.ta(1:50),(R2(1:50))/R2(1),'ro-'); grid on; hold off;  title('Signal auto-correlation');axis tight;
                if PRINTHW
                    FSTR = ['DAT',num2str(DATASEL),'_DD',num2str(DDIFF),'_Del',num2str(D1)];
                    hh1=figure(111);xlabel('t (ms)');FIGNAME=[FSTR,'_data']; set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
                    hh1=figure(112);xlabel('t (ms)');FIGNAME=[FSTR,'_datazoom']; set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
                    hh1=figure(113);xlabel('t (ms)');FIGNAME=[FSTR,'_datacov']; set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
                end
                %pause
            end
            if MID(mid)==1|MID(mid)==2|MID(mid)>5
                Pest(AGGR,D20)=d2.Pstar;
            end
            if MID(mid)==2
                Qest(AGGR,D20)=d2.Qstar;
            end
            if MID(mid)==3|MID(mid)==4|MID(mid)==7|MID(mid)==8
                iKK{MID(mid)}(AGGR,D20)=d2.params.kerneloption;
                iCC{MID(mid)}(AGGR,D20)=d2.params.C;
                iLL{MID(mid)}(AGGR,D20)=d2.params.lambda;
                iEE{MID(mid)}(AGGR,D20)=d2.params.epsilon;
                iOrd{MID(mid)}(AGGR,D20)=d2.Dstar;
                iFact{MID(mid)}(AGGR,D20)=d2.Mfstar;
            end
            if MID(mid)==5|MID(mid)==6
                iNN{MID(mid)}(AGGR,D20)=d2.NNHLstar(1);
                iOrd{MID(mid)}(AGGR,D20)=d2.Dstar;
            end
            
        end 
        L1str{AGGR}=['aggr=',num2str(AGGR*d1.dt),'ms'];
    end
    pSDR{MID(mid)}=SDR{MID(mid)};
    pCX{MID(mid)}=CX{MID(mid)};
    pCXS{MID(mid)}=CXS{MID(mid)};
    pSDRbase{MID(mid)}=SDRbase{MID(mid)};
    pCXbase{MID(mid)}=CXbase{MID(mid)};
    for AGGR=aggrV % aggregation level
        for D20=d2V
            if D20<AGGR
                SDR{MID(mid)}(AGGR,D20)=nan;
                CX{MID(mid)}(AGGR,D20)=nan;
                CXS{MID(mid)}(AGGR,D20)=nan;
                SDRbase{MID(mid)}(AGGR,D20)=nan;
                CXbase{MID(mid)}(AGGR,D20)=nan;
            end
        end
    end
    FSTR = ['DAT',num2str(DATASEL),'_DD',num2str(DDIFF),'_Del',num2str(D1)];
    
    % linear distortion to signal ratio
    figure(MID(mid)+1000);clf;
    plot(d2V*d1.dt,10.^(-(SDR{MID(mid)}')/10),'o-','markersize',12,'linewidth',2);grid on;
    %hold on;plot(d2V*d1.dt,pSDR{MID(mid)}','linewidth',2);hold off;
    legend(L1str,'location','southwest');xlabel('prediction horizon (ms)');
    title(['lin DSR ',TSTR{MID(mid)}]);ylabel('DSR');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),1)]);
    figure(MID(mid)+1001);clf;
    plot(d2V*d1.dt,10.^((SDR{MID(mid)}'/10))./10.^((SDRbase{MID(mid)}'/10)),'o-','markersize',12,'linewidth',2);grid on;
    legend(L1str,'location','southwest');xlabel('prediction horizon (ms)');
    title(['error Baseline / error ',TSTR{MID(mid)}]);ylabel('DSR ratio');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),0)]);
    if PRINTHW
        hh1=figure(MID(mid)+1000);FIGNAME=[FSTR,'_linSDR_',MSTR{MID(mid)}];
        set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
        hh1=figure(MID(mid)+1001);FIGNAME=[FSTR,'_linSDRbaseD_',MSTR{MID(mid)}];
        set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
    end
    
%     figure(MID(mid));clf;
%     plot(d2V*d1.dt,SDR{MID(mid)}','o-','markersize',12,'linewidth',2);grid on;
%     %hold on;plot(d2V*d1.dt,pSDR{MID(mid)}','linewidth',2);hold off;
%     legend(L1str,'location','northeast');xlabel('prediction horizon (ms)');
%     title(['SDR ',TSTR{MID(mid)}]);ylabel('SDR (dB)');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),0)]);
%     figure(MID(mid)+100);clf;
%     plot(d2V*d1.dt,SDR{MID(mid)}'-SDRbase{MID(mid)}','o-','markersize',12,'linewidth',2);grid on;
%     legend(L1str,'location','southwest');xlabel('prediction horizon (ms)');
%     title(['SDR above Baseline ',TSTR{MID(mid)}]);ylabel('SDR (dB)');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),0)]);
%         figure(MID(mid)+200);clf;
%     plot(d2V*d1.dt,CX{MID(mid)}','o-','markersize',12,'linewidth',2);grid on;
%     %hold on;plot(d2V*d1.dt,pSDR{MID(mid)}','linewidth',2);hold off;
%     legend(L1str,'location','southwest');xlabel('prediction horizon (ms)');
%     title(['\rho ',TSTR{MID(mid)}]);ylabel('\rho');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),0)]);
%     figure(MID(mid)+300);clf;
%     plot(d2V*d1.dt,CX{MID(mid)}'-CXbase{MID(mid)}','o-','markersize',12,'linewidth',2);grid on;
%     %hold on;plot(d2V*d1.dt,pSDR{MID(mid)}','linewidth',2);hold off;
%     legend(L1str,'location','northeast');xlabel('prediction horizon (ms)');
%     title(['\rho above Baseline  ',TSTR{MID(mid)}]);ylabel('\rho');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),0)]);
%   
%     if PRINTHW
%         hh1=figure(MID(mid));FIGNAME=[FSTR,'_SDR_',MSTR{MID(mid)}];
%         set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
%         hh1=figure(MID(mid)+100);FIGNAME=[FSTR,'_SDRbaseD_',MSTR{MID(mid)}];
%         set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
% %         hh1=figure(MID(mid)+1000);FIGNAME=[FSTR,'_SDRbase_',MSTR{MID(mid)}];
% %         set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
%         hh1=figure(MID(mid)+200);FIGNAME=[FSTR,'_RHO_',MSTR{MID(mid)}];
%         set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
%         hh1=figure(MID(mid)+300);FIGNAME=[FSTR,'_RHObaseD_',MSTR{MID(mid)}];
%         set(gca,'fontsize',17);printstr=[FPTH,FIGNAME,'.eps']; print(hh1,'-depsc','-loose', printstr);eps2xxx(printstr,{'pdf'}); eval(['! rm ',printstr]);
%     end
    

end

if 0
    figure(100);clf;
    subplot(121);
    plot(aggrV*d1.dt,SDRbase);grid on;
    legend(L2str,'location','northwest');xlabel('aggregation (ms)');
    title('SDR baseline');ylabel('SDR');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),0)]);
    subplot(122);
    plot(d2V,SDRbase');grid on;
    legend(L1str,'location','northeast');xlabel('k-step (*100 ms)');
    title('SDR baseline');ylabel('SDR');axis tight;yl=get(gca,'ylim');ylim([0 max(yl(2),0)]);
end