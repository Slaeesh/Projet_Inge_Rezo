function [py,SDR,CX,ID] = NNprediction(y,Ntrain,Ntest,D1,D2,DD,NNHL)

% --- construct data vectors for training and test
[inputs,targets,inputsT,targetsT,id,targetsall] = vectorize_data(y,DD,Ntrain,Ntest,D1,D2);

% --- train NN
net=feedforwardnet(NNHL);
net = train(net,inputs',targets');

% --- test NN on training and test data
pred0 = net(inputs'); pred0=pred0(:);
pred = net(inputsT'); pred=pred(:);

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

