function [py,SDR,CX,ID] = AR_SVM_prediction(y,Ntrain,Ntest,D1,D2,P,DD,Mfact,params)

C=params.C;
lambda=params.lambda;
epsilon=params.epsilon;
kernel=params.kernel;
kerneloption=params.kerneloption;
verbose=0;

% --- construct data vectors for training and test
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,P,Ntrain,Ntest,D1,D2);

% --- estimate AR coefficients & train NN on residual
xpast=y(1:id{2}(1)-id{1}(1));
[a] = AR_est(xpast, P, D1, D2); % AR coefficients
mer = targets - AR_predict(inputs,a);
% --- train SVM
[~,~,inputsSVM0,targetsSVM0] = vectorize_data(mer*Mfact,DD,0,Ntest+Ntrain,D1,D2);
[xsup,ysup,w,w0] = svmreg(inputsSVM0,targetsSVM0,C,epsilon,kernel,kerneloption,lambda,verbose);



% --- predict training and test data
[~,~,inputsAR,targetsAR,idAR,targetsallAR] = vectorize_data(y,P,0,Ntest+Ntrain,D1,D2);
%[~,~,inputsAR,targetsAR,idAR,targetsallAR] = vectorize_data(y,P,0,Ntest,D1,D2);
ARpred= AR_predict(inputsAR,a); % predict AR

[~,~,inputsSVM,targetsSVM,idSVM] = vectorize_data((targetsAR-ARpred)*Mfact,DD,0,Ntest,D1,D2);
SVMpred  = svmval(inputsSVM,xsup,w,w0,kernel,kerneloption)/Mfact;   % prediction on test set

ARSVMpred=ARpred; ARSVMpred(idSVM{2})=ARSVMpred(idSVM{2})+SVMpred; % sum

pred0=ARSVMpred(1:Ntrain);
pred=ARSVMpred(Ntrain+1:end);


% --- quantify errors NN
tmp=corrcoef(targets,pred0); CXtrain=tmp(2);tmp=corrcoef(targetsT,pred); CXtest=tmp(2);
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

