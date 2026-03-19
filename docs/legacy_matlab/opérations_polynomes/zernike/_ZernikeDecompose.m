%%Fonction SCRIPT

function [a,Z,res,ns,ms,RMSNorm]=ZernikeDecompose(carteObj,vec,type)
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
pup=~isnan(carteObj.carte(:)) & sqrt(carteObj.grilleX(:).^2+carteObj.grilleY(:).^2)<1;
X=carteObj.grilleX(pup);
Y=carteObj.grilleY(pup);
Zs=carteObj.carte(pup);

X=X(:);Y=Y(:);Zs=Zs(:);
[~,rs] = cart2pol(X,Y); 
%disp(rs)
Rs=rs<=1;


%disp(size(X))
%disp(size(Y))
[Z,ns,ms]=zernike_fcn3(vec, X, Y, Rs,type);

%%
%a = Z\Zs(rs<=1);

a=pinv(Z)*Zs;
res=Zs(rs<=1)-Z*a;
RMSNorm=( sqrt((1+(ms(1:end)~=0)).*(ns(1:end)+1)) )'; %To Be tested

end