% Copyright Camille Frappoli
%generateMHFMap  genere une carte de HF en simulant le passage d'un outil
%qui parcourt un miroir sphérique "comme un serpent", par ligne successives.
%

function  [zsg,pup]=generateMHFMap(Rseg, resolution);
size=resolution;
% Parametres
par1=1;
par2=2.5/0.08;
% snakelike defects generator vs
Nptssg=floor(900);
[xsg,ysg]=squaregrid(Rseg,Nptssg,0);
uxsg=unique(xsg);
for i=1:floor(0.5*length(uxsg))
    ysg(xsg==uxsg(2*i))=flip(ysg(xsg==uxsg(2*i)));
end

%make the vector suitable
%rescale vector
xsR=size*0.5+xsg*size/Rseg/2;
ysR=size*0.5+ysg*size/Rseg/2;
radiuses=(size/1001)*par1*floor(5+12*randn(length(xsg),1));
vecX=[];
for i=1:length(xsg)
    vecX=[vecX; [floor(xsR(i)) floor(ysR(i)) radiuses(i)]];
end
X=1:size;
Y=1:size;
zsg=zeros(size);
for i=1:length(xsg)
    for X=1:size
        for Y=1:size
            if (X-vecX(i,1)).^2+(Y-vecX(i,2)).^2<=vecX(i,3)^2
                zsg(X,Y)=1;
            end
        end
    end
end

%Ce bloc permet de retourner la carte des altitudes avec le bon pas.
[X,Y]=meshgrid(Rseg*linspace(-1,1,size),Rseg*linspace(-1,1,size));
pup=sqrt(X.^2+Y.^2)<Rseg;

%Bloc ecrit pour eviter d'utiliser fspecial et imfilter qui appartiennent à
%la biblioteque image processing toolbox, dont le nombre de licence Safran
%est largement insuffisant. 2018/10/24.
sigma=20*size/1001;
mu=size/2;
X=[1:size];
Y=[1:size];
kernel=exp(-(X-mu).^2/(2*sigma^2))'*exp(-(Y-mu).^2/(2*sigma^2));
kernel=kernel/sum(sum(kernel));
zsg=par2*1000*conv2(zsg,kernel,'same');

%h = fspecial('gaussian', [size,size],20*size/1001);
%zsg = par2*1000*imfilter(zsg, h);
zsg(~pup)=nan;
zsg=(zsg-mean(zsg(:),'omitnan'))/30; %100 nm PV
