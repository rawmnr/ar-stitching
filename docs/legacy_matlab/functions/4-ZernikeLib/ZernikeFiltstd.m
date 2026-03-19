function [res,a]=ZernikeRemFringe(X,Y,Zs,vec,plotting)

[thetas,rs] = cart2pol(X,Y); 
%Z = zernfun2([0:35],rs,thetas); %2ms WRONG ORDER of polynomia
Z=zernike_fcn3([1:max(vec)], X, Y, rs<=1, 'FringeExt');
%%
a = Z\Zs(rs<=1);
b=a;b(1:min(vec)-1)=zeros(size(b(1:min(vec)-1)));
if plotting
        figure(50)
        clf
        subplot(2,3,1)
        scatter(X(rs<=1),Y(rs<=1),5,Zs(rs<=1),'filled')
        subplot(2,3,2)
        scatter(X(rs<=1),Y(rs<=1),5,Z*a,'filled')
        subplot(2,3,[4,5,6])
        bar(1e6*a)
        subplot(2,3,3)
        scatter(X(rs<=1),Y(rs<=1),5,Zs(rs<=1)-Z*a,'filled')
end
res=Z*b;

if plotting
    for k=1:36
        figure(51)
        subplot(6,6,k)
        b=zeros(size(a));
        b(k)=a(k);
        scatter(X(rs<=1),Y(rs<=1),5,Z*b,'filled')
        %title(['RMS = ',num2str(1e6*std(Z*b)),' \mu m'])
        title([num2str(1e6*std(Z*b)),'   ',num2str(k)])
    end
end


end