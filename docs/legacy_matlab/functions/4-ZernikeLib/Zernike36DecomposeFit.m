function [a,b,Z,res,rmsZ,rmsZcorr,rmsZcorr2,rmsres,anorm]=Zernike36Decompose(X,Y,Zs,plotting)

[thetas,rs] = cart2pol(X,Y); 
%Z = zernfun2([0:35],rs,thetas); %2ms WRONG ORDER of polynomia
Z=zernike_fcn3([1:37], X, Y, rs<=1, 'fringe');
%%
a = Z\Zs(rs<=1);
b=a;
%build 36 coeff list
coefflist={};
coeffstr='';
for i=1:36
    coefflist{i}=['a',num2str(i)];
    if i<36
        coeffstr=[coeffstr,coefflist{i},','];
    else
        coeffstr=[coeffstr,coefflist{i}];
    end
end
%%
chi2=@(vec)(sum((Zs'-vec'*Z').^2));

if 0
    options = optimset('Display', 'off','TolFun',1e-15,'TolX',1e-15);
    [af,fval,exitflag,output,lambda,grad] = fmincon(chi2,a,[],[],[],[],[],[],[],options);
elseif 1
    options = optimoptions(@fminunc,'Display', 'off','Algorithm','Quasi-Newton');
    [af,fval,exitflag,output] = fminunc(chi2,a,options);
else
    options = optimoptions(@fminunc,'Display', 'iter-detailed','Algorithm','Quasi-Newton');
    [af,fval,exitflag,output] = fminunc(chi2,std(Zs(:)).*[1:37]',options);
end
%if abs(chi2(a)-chi2(af))<1e-50
 %   a=af;
%else
 %   a=af;
%end
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