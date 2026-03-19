%Auteur : Camille Frapolli a partir de 735160

function [map,mismatch]=Subaper_Core_v2(TableData, Type, indice_alignement)
%%
[GrilleX,GrilleY]=meshgrid(linspace(-1,1,size(TableData,1)),linspace(-1,1,size(TableData,2)));
switch Type
    case 'Z'
        pup=GrilleX.^2+GrilleY.^2<1;
        P=NaN(length(GrilleX(:)),length(indice_alignement));
        P(pup,:)=zernike_fcn3(indice_alignement,GrilleX,GrilleY,pup,'Fringe');
    case 'L'
        pup=true(size(GrilleX));
        P=NaN(length(GrilleX(:)),length(indice_alignement));
        P(pup,:)=Legendre2D(indice_alignement,GrilleX,GrilleY)';
end
Por=P;
%On vectorise avec le bon nombre de variables
Ncoeffs=length(indice_alignement);
NSSPP=size(TableData,3);
NPix=size(P,1);
%On copie les coeffs d'alignement NSSPP fois chacun de manière contigue
P=reshape(repmat(P,1,NSSPP),NPix,Ncoeffs*NSSPP);
%Ensuite on applique le masque pour chaque SSPP
MSK=squeeze(reshape(~isnan(TableData),NPix,NSSPP));
MSKn=double(MSK);MSKn(MSKn==0)=NaN;
P=P.*reshape(repmat(MSKn,Ncoeffs,1),NPix,Ncoeffs*NSSPP);
%Du coup, on peut construire la fonction d'erreur facilement
T=reshape(TableData,NPix,NSSPP);
P(isnan(P))=0;
TZ=T; 
%Version of subpupils with zeros to allow summing in mean map
TZ(isnan(TZ))=0;
%DIRECT VERSION DOES NOT WORK :(
%On crée la fonction qui calcule la différence entre chaque SSPP avec ses
%coeffs d'alignement et la carte reconstruite. On devra minimiser
% residuals=@(Cs)(sum((T+sum(reshape(P.*repmat(Cs',NPix,1),NPix,NSSPP,Ncoeffs),3)-...
%     repmat((sum(TZ,2)+sum(P*Cs,2))./sum(MSK,2),1,NSSPP)).^2,2,'omitnan'));
% chi2=@(Cs)(1e-6*sqrt(sum(residuals(1e4.*[0;0;0;Cs]),'omitnan')));
% %On minimise
% options = optimset('Display','iter');
% res=fminsearch(chi2,rand((NSSPP-1)*Ncoeffs,1),options); DOES NOT WORK

%% 3300339321
M=zeros((NSSPP-1)*Ncoeffs);
MM=zeros((NSSPP),Ncoeffs,(NSSPP),Ncoeffs);
b=zeros(1,(NSSPP-1)*Ncoeffs);
eta=sum(MSK,2);
%setup waitbar
ppm = ParforProgMon('', NSSPP-1);
parfor k=2:NSSPP
    ppm.increment();
    for m=1:Ncoeffs
        Tempb=zeros(size(MSK));     
        for kkk=1:NSSPP %corrected 31/03/2020
            TempMSK=MSK(:,kkk) & MSK(:,k) & pup(:);
            Tempb(TempMSK,kkk)=(1./eta(TempMSK)-(k==kkk)).*Por(TempMSK,m).*T(TempMSK,kkk);
        end
        b(k-1,m)=sum(sum(Tempb));
        for kp=2:NSSPP
            for mp=1:Ncoeffs
                TempMSK=MSK(:,kp) & MSK(:,k) & pup(:);
                temp=((k==kp)-1./eta(TempMSK));
                MMT1(mp)=sum(temp.*(Por(TempMSK,m).*Por(TempMSK,mp)),'omitnan');
            end
            MMT2(kp,:)=MMT1
        end
        MMT3(m,:,:)=MMT2
    end
    MM(k,:,:,:)=MMT3
end
delete(hchrgt)
b=reshape(b',size(b(:)'));
%moche mais rapide...
for k=2:NSSPP
    for m=1:Ncoeffs
        for kp=2:NSSPP
            for mp=1:Ncoeffs
                M(Ncoeffs*(k-2)+m,Ncoeffs*(kp-2)+mp)=MM(k-1,m,kp-1,mp);
            end
        end
    end
end

close ppm
%Invert system
Cs=M\b';
Cs=[zeros(Ncoeffs,1);Cs];
%Build stitched map 3300339321
MMap=(sum(TZ,2)+P*Cs)./eta;
a=Por(~isnan(MMap)&pup(:),:)\MMap(~isnan(MMap)&pup(:)); %pup added 31/03/2020
map=reshape(MMap-Por*a,size(GrilleX));

%Build mismatch as std of all SPP
RMap=zeros(size(MMap));
for k=1:NSSPP
    RMap(:,k)=(TZ(:,k)+Por*Cs([1:Ncoeffs]+(k-1)*Ncoeffs)).*MSKn(:,k);
end
mismatch=reshape(std(RMap,0,2,'omitnan'),size(GrilleX));
mismatch(mismatch==0)=NaN;



end