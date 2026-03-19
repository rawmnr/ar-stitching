function [a,b]=ZernikestdDecomposeAlb(X,Y,Zs,N,As,Zmax,plotting)
%Careful, A must be ordered same as X,Y

[thetas,rs] = cart2pol(X,Y); 
vec=[1:Zmax];
[Z,ns,ms]=zernike_fcn3(vec, X, Y, rs<=1, 'FringeExt');
AsM=sqrt(2*repmat(ns',1,length(X))+(repmat(ms',1,length(X))~=0)+1).*(sqrt(repmat(As,size(Z,2),1)/2/(4*N+1)));
AsM(ms==0,:)=sqrt(repmat(ns(ms'==0)',1,length(X))+1).*(sqrt(repmat(As,sum((ms'==0)),1)/2/(4*N+1)));
a=(Z'.*AsM)*(AsM'.*Zs);
a=diag(a);
%%
b = Z\Zs(rs<=1);
if plotting
figure(1)
clf
plot(b(1:20))
hold on
plot(a(1:20))
%%
figure(1)
clf
plot(a)
hold on 
plot(b)
end
%%
if 0
    figure(1)
    clf
    plot(a)
    hold on
    plot(5*b)
end
%%

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
res=Zs(rs<=1)-Z*a;
rmsZ=std(Z*a);
%ESO spec
acorr=a;acorr([4,10,11])=0.15*a([4,10,11]);acorr(5:6)=0.05*a(5:6);
%Comparetoremisremovedmaterial
acorr2=zeros(size(a));acorr2(5:8)=a(5:8);acorr2(10:11)=a(10:11);
rmsZcorr=std(Z*acorr);
rmsZcorr2=std(Z*acorr2);
rmsres=std(res);

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