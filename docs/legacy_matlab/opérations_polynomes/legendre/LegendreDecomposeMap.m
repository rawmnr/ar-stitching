%%Fonction SCRIPT

%--------------------------
% function to decompose a scattered cloud (X,Y,Z) in terms of Legendre
% polynomials. The output is a vector of legendre polynomials a.
% X,Y are the NORMALIZED coordinates of the points (typically real
% coordinate / normalization radius) and Z is the altitude. vec is a vector
% containing the zernikes desired in the ouptut vector (example [1:36] for
% ESO Legendre Zernikes).
% The output vector a is given in the same unit as Z.
%
% Example : Decompose a scatter cloud X,Y,Z on the legendre ESO basis
% a=LegendreDecomposeMap(X,Y,Zs,[1:36]);
%--------------------------

function [a,L,res,ns,ms,RMSNorm]=LegendreDecomposeMap(carteObj,vec)

%Preparing the matrix of Legendre Polynomials
[L,ns,ms] = Legendre2D([1:max(vec)],carteObj.grilleX,carteObj.grilleY);
L=L(vec,:);
ms=ms(vec);
ns=ns(vec);

%Dealing with mask
carteDat=carteObj.carte(~isnan(carteObj.carte));
L(:,isnan(carteObj.carte(:)))=[];L=L';

%Fit coefficients
%a=pinv(L)*carteDat;
cov=L'*L; %c'est mieux car L est en fait de rang NZern....
pcov=pinv(cov);
uvec=carteDat'*L;
a=pcov*uvec';

%Fit residual
res=carteDat-L*a;

%RMSNorm
RMSNorm=sqrt(1./(2*ns+1)./(2*ms+1));


end

