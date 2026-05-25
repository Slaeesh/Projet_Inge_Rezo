function [py,SDR,CX,ID] = ARprediction(y,Ntrain,Ntest,D1,D2,P)

%my=min(y); y=y-my; My=max(y); y=(y/My - 1/2); meany=mean(y); y=y-meany;

% --- construct data vectors for training and test
    [inputs,targets,inputsT,targetsT,id] = vectorize_data(y,P,Ntrain,Ntest,D1,D2);
    
if ~iscell(y) % standard prediction
    
    % --- estimate coefficients
    %xpast=inputs(:,1);
    xpast=y(1:id{2}(1)-id{1}(1));
    [a] = AR_est(xpast, P, D1, D2);
else % predict y{2} from y{1}
    % --- construct data vectors for training and test
    %[inputs,targets,inputsT,targetsT,id] = vectorize_crossdata(y,P,Ntrain,Ntest,D1,D2);
    xpast{1}=y{1}(1:id{2}(1)-id{1}(1)); xpast{2}=y{2}(1:id{2}(1)-id{1}(1));
    [a] = AR_est_cross(xpast, P, D1, D2);
end
    

% --- predict training and test data
pred0 = AR_predict(inputs,a);
pred  = AR_predict(inputsT,a);
baseline = inputsT(:,1);      % baseline prediction on test set

% --- quantify errors
%keyboard
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