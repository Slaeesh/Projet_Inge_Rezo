function [py,SDR,CX,ID] = ARMAprediction(y,Ntrain,Ntest,D1,D2,P,Q)


% --- construct data vectors for training and test
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,P,Ntrain,Ntest,D1,D2);

% --- estimate coefficients
%xpast=inputs(:,1);
xpast=y(1:id{2}(1)-id{1}(1));
[a] = AR_est(xpast, P, D1, D2); % AR coefficients
mer = targets - AR_predict(inputs,a);
[b] = AR_est(mer, Q, D1, D2); % MA coefficients

% --- predict training and test data
[~,~,inputsAR,targetsAR,idAR,targetsallAR] = vectorize_data(y,P,0,Ntest+Ntrain,D1,D2);
%[~,~,inputsAR,targetsAR,idAR,targetsallAR] = vectorize_data(y,P,0,Ntest,D1,D2);
ARpred= AR_predict(inputsAR,a); % predict AR


[~,~,inputsMA,targetsMA,idMA,targetsallMA] = vectorize_data(targetsAR-ARpred,Q,0,Ntest,D1,D2);
MApred= AR_predict(inputsMA,b); % predict residuals with MA

ARMApred=ARpred; ARMApred(idMA{2})=ARMApred(idMA{2})+MApred; % sum

pred0=ARMApred(1:Ntrain);
pred=ARMApred(Ntrain+1:end);

% --- quantify errors ARMA
tmp=corrcoef(targets,pred0); CXtrain=tmp(2); tmp=corrcoef(targetsT,pred); CXtest=tmp(2);
tmp=corrcoef(sign(targets),sign(pred0)); CXtrainS=tmp(2); tmp=corrcoef(sign(targetsT),sign(pred)); CXtestS=tmp(2);

SDRtrain=10*log10(mean(targets.^2)/mean((targets-pred0).^2));
SDRtest=10*log10(mean(targetsT.^2)/mean((targetsT-pred).^2));


py.train=(pred0);%+meany+1/2)*My+my;
py.test=(pred);%+meany+1/2)*My+my;
SDR.train=SDRtrain;
SDR.test=SDRtest;
CX.train=CXtrain;
CX.test=CXtest;
CX.trainS=CXtrainS;
CX.testS=CXtestS;
ID{1}=id{1};ID{2}=id{2};

