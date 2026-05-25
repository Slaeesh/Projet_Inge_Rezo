function [R] = ACF_fft(x,NW)
%% function [R] = ACF_fft(x,NW)
% computes autocorrelation function of x
% NW - number of windows to use (default: 1 - use entire signal x as is - RECOMMENDED)

x=x(:);
if nargin<2
    X = fft(x,2^nextpow2(2*size(x,1)-1));R = ifft(abs(X).^2);
else
    NN=floor(length(x)/NW);
    for i=1:NW;
        X = fft(x((i-1)*NN+1:i*NN),2^nextpow2(2*size(x,1)-1));R0(i,:) = ifft(abs(X).^2);
    end
    R=mean(R0);
end
        
    