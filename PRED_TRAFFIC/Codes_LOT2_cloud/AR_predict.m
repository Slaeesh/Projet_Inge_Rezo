function [pred] = AR_predict(inputs,a)
%% function [pred] = AR_predict(inputs,a)
%
% linear prediction
% "inputs" is an Nsteps x N matrix
% "a" are the N coefficient
%

for k=1:length(inputs(:,1));  % prediction on training set
    pred(k)=dot(a(:),inputs(k,:)');
end
pred=pred(:);