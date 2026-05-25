function [py,SDR,CX,ID] = SVMprediction(y,Ntrain,Ntest,D1,D2,DD,Mfact,params, iRD)
if nargin<9; iRD=0; end
y0=y;

C=params.C;
lambda=params.lambda;
epsilon=params.epsilon;
kernel=params.kernel;
kerneloption=params.kerneloption;
verbose=0;

if iscell(y);
    for ii=1:length(y);y{ii}=y{ii}*Mfact; end
else
    y=y*Mfact;
end
        
% --- construct data vectors for training and test
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,DD,Ntrain,Ntest,D1,D2,iRD);

% --- train SVM
[xsup,ysup,w,w0] = svmreg(inputs,targets,C,epsilon,kernel,kerneloption,lambda,verbose);

% --- test SVM on training and test data
pred0 = svmval(inputs, xsup,w,w0,kernel,kerneloption)/Mfact;   % prediction on training set
pred  = svmval(inputsT,xsup,w,w0,kernel,kerneloption)/Mfact;   % prediction on test set


% --- quantify errors SVM
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

