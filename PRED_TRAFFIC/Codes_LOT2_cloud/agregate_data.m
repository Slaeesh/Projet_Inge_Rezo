function [ya,ta] = agregate_data(y,af,ds,dt)
if nargin<2; af=1; end
if nargin<3; ds=1; end
if nargin<4; dt=1; end

af=max(1,ceil(af));
if ds
    yi=cumsum(y);
    ya=diff(yi(af:af:end));
    ta=(1/2+(0:length(ya)-1))*af*dt;
else
    y=y(:);y=y(1:floor(length(y)/af)*af);
    yi=cumsum(y);
    for k=1:af;
        yy(k,:)=diff(yi(k+af:af:end));
    end
    ya=yy(:);
    ta=(1/2+(0:length(ya)-1))*af*dt + dt;
end

%figure(111);plot(ta,ya,'ro-');

