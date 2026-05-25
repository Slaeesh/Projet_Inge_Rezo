function [a0] = AR_est(x, P0, D1, D2)
%% function [a0] = AR_est(x, P0, D1, D2)
%
% Estimates P0 linear prediction coefficients (using Yule-Walker equations)
%
% x     - training samples
% P0    - model order (i.e., # coefficients)
% D1    - use every D1-th sample
% D2    - predict D2 samples avead
    
P=(P0-1)*D1+D2;

N = length (x);
x = x(:)';
%P = N-1;
[m,n] = size(x);
if (n>1) && (m==1)
    x = x(:);
    [m,n] = size(x);
end
% Compute autocorrelation vector or matrix
X = fft(x,2^nextpow2(2*size(x,1)-1));
R = ifft(abs(X).^2);
r=R(1:P+1);
r = r(:);

% Estimate coefficients: Yule-Walker equations
% Rmat = toeplitz (r(1 : end - 1));
% rvec = r(2 : end);
% a =  Rmat \ rvec;
%a0=a(D2:D1:end);



% Rmat = toeplitz (r(1 : end - D2));
% rvec = r(D2+1 : end);
% a =  Rmat \ rvec;
% a0=a(1:D1:end);

Rmat = toeplitz (r(1 : D1 : end - D2));
rvec = r(D2+1 : D1 :  end);
a =  Rmat \ rvec;
a0=a;

end
