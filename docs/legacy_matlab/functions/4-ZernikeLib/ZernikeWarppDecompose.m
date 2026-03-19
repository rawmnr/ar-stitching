function [a,Z,residuals,ns,ms,piston,RMSNorm]=ZernikeWarppDecompose(X,Y,Zs,vec)
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
% a=ZernikeWarppDecompose(X,Y,Zs,[1:36]);
% RMSNorm is Zi/RMS to normalize coeefs in terms of RMS.
% residuals id the residuals from the zernike decomposition
%--------------------------
[~,rs] = cart2pol(X,Y); 
[Z,ns,ms]=zernike_fcn3([1;vec(:)+1], X, Y, rs<=1, 'FringeExt'); %if you add +1
%%
%a = Z\Zs(rs<=1);

aTemp=pinv(Z)*Zs;
residuals=Zs(rs<=1)-Z*aTemp;

a=aTemp(2:end);
piston=aTemp(1);
RMSNorm=( sqrt((1+(ms(2:end)~=0)).*(ns(2:end)+1)) )'; %To Be tested
end