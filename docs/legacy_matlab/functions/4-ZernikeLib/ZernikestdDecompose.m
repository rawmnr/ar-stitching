function [a,Z,res,ns,ms]=ZernikestdDecompose(X,Y,Zs,vec)
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
[~,rs] = cart2pol(X,Y); 
[Z,ns,ms]=zernike_fcn3(vec, X, Y, rs<=1, 'FringeExt');
%%
%a = Z\Zs(rs<=1);

a=pinv(Z)*Zs;
res=Zs(rs<=1)-Z*a;
end