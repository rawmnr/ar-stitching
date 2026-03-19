function [a,Z,res,Norms]=ZernikestdDecomposeFit(X,Y,Zs,vec,plotting)

[~,rs] = cart2pol(X,Y); 
[Z,n,m]=zernike_fcn3(vec, X, Y, rs<=1, 'FringeExt');
Norms=(sqrt((1+(m~=0)).*(n+1)/pi))';
%%
%alternative 1
a = Z\Zs(rs<=1);


res=Zs(rs<=1)-Z*a;
end