function [a,Z,res]=ZernikestdDecomposeFit(X,Y,Zs,vec,plotting)

[~,rs] = cart2pol(X,Y); 
Z=zernike_fcn3(vec, X, Y, rs<=1, 'FringeExt');
%% gramm schmidt
[Q,R] = qr(Z);
%% 

a = Z\Zs(rs<=1);
res=Zs(rs<=1)-Z*a;
end