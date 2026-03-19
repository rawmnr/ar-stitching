%Auteur : Camille Frapolli a partir de 735160
%Cette version est une proposition ou on utilise un tableau sparse en
%entrée de taille NPix * NSSPP

function [map,mismatch]=Subaper_Core_v2_sparse(TableData, Type, indice_alignement)
%%
[GrilleX,GrilleY]=meshgrid(linspace(-1,1,sqrt(size(TableData,1))),linspace(-1,1,sqrt(size(TableData,1))));
switch Type
    case 'Z'
        pup=GrilleX.^2+GrilleY.^2<1;
        Por=NaN(length(GrilleX(:)),length(indice_alignement));
        Por(pup,:)=zernike_fcn3(indice_alignement,GrilleX,GrilleY,pup,'Fringe');
    case 'L'
        pup=true(size(GrilleX));
        Por=NaN(length(GrilleX(:)),length(indice_alignement));
        Por(pup,:)=Legendre2D(indice_alignement,GrilleX,GrilleY)';
end
%On vectorise avec le bon nombre de variables
Ncoeffs=length(indice_alignement);
NSSPP=size(TableData,2);
NPix=size(TableData,1);
%On copie les coeffs d'alignement NSSPP fois chacun de manière contigue
%dans une matrice sparse
P=sparse(NPix,Ncoeffs*NSSPP);
for i=1:NSSPP
    P(TableData(:,i)~=0,1+Ncoeffs*(i-1):Ncoeffs*i)=Por(TableData(:,i)~=0,:);
end
%Ensuite on applique le masque pour chaque SSPP
MSK=(TableData~=0);

%% 3300339321
M=zeros((NSSPP-1)*Ncoeffs);
b=zeros(1,(NSSPP-1)*Ncoeffs);
eta=sum(TableData~=0,2);
%setup waitbar
hchrgt = waitbar(1,['Sous-pupille : 1/' num2str(NSSPP)],'Name','Stitching en cours');
for k=2:NSSPP
    waitbar(k/NSSPP,hchrgt,sprintf('Sous-pupille : %d/%d',k,NSSPP))
    for m=1:Ncoeffs
        Tempb=zeros(size(MSK));     
        for kkk=1:NSSPP %corrected 31/03/2020
            TempMSK=MSK(:,kkk) & MSK(:,k) & pup(:);
            Tempb(TempMSK,kkk)=(1./eta(TempMSK)-(k==kkk)).*Por(TempMSK,m).*TableData(TempMSK,kkk);
        end
        b(Ncoeffs*(k-2)+m)=sum(sum(Tempb));
        for kp=2:NSSPP
            for mp=1:Ncoeffs
                TempMSK=MSK(:,kp) & MSK(:,k) & pup(:);
                M(Ncoeffs*(k-2)+m,Ncoeffs*(kp-2)+mp)=sum(((k==kp)-1./eta(TempMSK)).*(Por(TempMSK,m).*Por(TempMSK,mp)),'omitnan');
            end
        end
    end
end
delete(hchrgt)

%Invert system
Cs=M\b';
Cs=[zeros(Ncoeffs,1);Cs];
%Build stitched map 3300339321
MMap=(sum(TableData,2)+P*Cs)./eta;
a=Por(~isnan(MMap)&pup(:),:)\MMap(~isnan(MMap)&pup(:)); %pup added 31/03/2020
map=reshape(MMap-Por*a,size(GrilleX));

%Build mismatch as std of all SPP
RMap=sparse(NPix,Ncoeffs*NSSPP);
for k=1:NSSPP
    RMap(TableData(:,k)~=0,k)=TableData(TableData(:,k)~=0,k)+Por(TableData(:,k)~=0,:)*Cs([1:Ncoeffs]+(k-1)*Ncoeffs);
end
mismatch=real(sqrt((sum(RMap.^2,2)./sum(TableData~=0,2))-(sum(RMap,2).^2./sum(TableData~=0,2).^2)));
mismatch(mismatch==0)=NaN;
mismatch=double(reshape(mismatch,size(GrilleX)));

end
