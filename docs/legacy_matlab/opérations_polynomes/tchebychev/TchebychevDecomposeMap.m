%%Fonction SCRIPT

%--------------------------
% function to decompose a scattered cloud (X,Y,Z) in terms of Tchebychev
% polynomials. The output is a vector of Thcebychev polynomials a.
% X,Y are the NORMALIZED coordinates of the points (typically real
% coordinate / normalization radius) and Z is the altitude. vec is a vector
% containing the zernikes desired in the ouptut vector (example [1:36] for
% ESO Tchebychev Zernikes).
% The output vector a is given in the same unit as Z.
%
% Example : Decompose a scatter cloud X,Y,Z on the legendre ESO basis
% a=LegendreDecomposeMap(X,Y,Zs,[1:36]);
%--------------------------

function [a,T,res,ns,ms,RMSNorm]=TchebychevDecomposeMap(carteObj,vec)

%Preparing the matrix of Tchebychev Polynomials
[T,ns,ms] = tchebychev2D(vec,carteObj.grilleX,carteObj.grilleY);

%Dealing with mask
carteDat=carteObj.carte(~isnan(carteObj.carte));
T(:,isnan(carteObj.carte(:)))=[];T=T';
std(T);

%Fit coefficients
a=pinv(T)*carteDat;

%Fit residual
res=carteDat-T*a;

%RMSNorm
RMSNorm=sqrt(1./(2*ns+1)./(2*ms+1));

