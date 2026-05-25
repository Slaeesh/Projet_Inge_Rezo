function [a0] = AR_est_cross(x, P0, D1, D2)
%% function [a0] = AR_est(x, P0, D1, D2)
%
% Estimates P0 linear prediction coefficients (using Yule-Walker equations)
%
% x{1}, x{2}     - training samples: input - target
% P0    - model order (i.e., # coefficients)
% D1    - use every D1-th sample
% D2    - predict D2 samples avead
    
P=(P0-1)*D1+D2;

N = length (x{1});
for ii=1:2
x{ii} = x{ii}(:)';
%P = N-1;
[m,n] = size(x{ii});
if (n>1) && (m==1)
    x{ii} = x{ii}(:);
    [m,n] = size(x{ii});
end
% Compute crosscorrelation vector or matrix
X{ii} = fft(x{ii},2^nextpow2(2*size(x{ii},1)-1));
end
R = ifft(X{1}.*conj(X{2}));
%R = ifft(abs(X).^2);
r=real(R(1:P+1));
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
