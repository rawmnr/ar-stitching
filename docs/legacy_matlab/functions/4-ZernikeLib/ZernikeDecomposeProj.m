function [a,ns,ms,RMSNorm]=ZernikeDecomposeProj(X,Y,Zs,vec,type)
%--------------------------
% function to decompose a scattered cloud (X,Y,Z) in terms of zernike
% polynomials. The output is a vector of zernike polynomials a.
% X,Y are the NORMALIZED coordinates of the points (typically real
% coordinate / normalization radius) and Z is the altitude. vec is a vector
% containing the zernikes desired in the ouptut vector (example [1:36] for
% FRINGE Zernikes).
% The output vector a is given in the same unit as Z.
%
% Example : Decompose a scatter cloud X,Y,Z on the fringe zernike basis
% a=ZernikestdDecompose(X,Y,Zs,[1:36]);
%--------------------------
X=X(:);Y=Y(:);Zs=Zs(:);
a=zeros(max(vec),1);    
[Zz,ns,ms]=zernike_fcn3(vec, X, Y, true(size(X)),type);
for i=vec
    %%
    a(i)=sum(Zz(:,i).*Zs)/sum(Zz(:,i).*Zz(:,i));
end
RMSNorm=( sqrt((1+(ms(1:end)~=0)).*(ns(1:end)+1)) )'; %To Be tested
end