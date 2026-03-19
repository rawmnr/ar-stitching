function varargout = unevencontour(x,y,z,v)
z=z(:);
x=x(:);
y=y(:);
%interpolate
[X,Y] = ndgrid(linspace(min(x),max(x),300),linspace(min(y),max(y),300));
Z = griddata(x,y,z,X,Y,'cubic');
%naning outside max radius
[ts,rhosori]=cart2pol(x,y);
maxr=max(rhosori(:));
R=sqrt(X.^2+Y.^2);
Z(R>maxr)=nan;
if nargin>3
    [C,h] = contourf(X,Y,Z,v,'LineColor','none');
else
    [C,h] = contourf(X,Y,Z,20,'LineColor','none');
end
colorbar
%colormap(brewermap(20,'spectral'));
colormap(inferno)
%xlim([1.05*min(x),1.05*max(x)])
%ylim([1.05*min(y),1.05*max(y)])

if nargout
   varargout{1}=[C,h];
end
end
