%Auteur : Camille Frapolli a partir de 735160 / Modifié par Renaud Mercier
%Ythier pour prendre en compte des compensateurs fixes par pupille
%Recodage des matrices d'optimisation dans ce sens
%On définit des compensateurs fixes ans le repère instrument 
%dont l'amplitude est la même pour chaque sous-pupille. Arguments :
%TableData=tableau des cartes de dimensions (#pixelX,#pixelY,#Sous-Pupilles)
%Type= 'Z' ou 'L'
%indice_alignement=vecteur ligne comprenant les numéros des Zernikes ou Legendre
%Compfixes=tableau contenant les compensateurs fixes par sous-pupilles sous forme de
%vecteurs colonnes de dimensions (#pixels,#Sous-Pupilles,#compensateurs)
%Sorties :
%map2=carte stitchée
%mismatch2=carte de mismatch
%moy_mismatch, rms_mismatch = moyenne et rms de la carte de mismatch
%cs1 : vecteur colonne contenant les amplitudes des compensateurs par sous-pupilles

function [map2,mismatch2,moy_mismatch,std_mismatch,cs1,a2]=Subaper_Compensateurs_fixes(TableData, Type, indice_alignement,indice_carte, Compfixes)
    arguments
        TableData (:,:,:) double
        Type {mustBeMember(Type,["L","Z"])} = "L"
        indice_alignement (1,:) double = [1:3]
        indice_carte double = 1
        Compfixes (:,:,:) double = []
     end
%%
%On vectorise avec le bon nombre de variables
Ncoeffs=length(indice_alignement);
NSSPP=size(TableData,3);
sz=size(Compfixes);
if sz(1)==0
     Ncompf=0;
else
    Ncompf=sz(3);
end

[GrilleX,GrilleY]=meshgrid(linspace(-1,1,size(TableData,1)),linspace(-1,1,size(TableData,2)));
switch Type
    case 'Z'
        pup=GrilleX.^2+GrilleY.^2<1;
        P=NaN(length(GrilleX(:)),length(indice_alignement));
        P(pup,:)=zernike_fcn3(indice_alignement,GrilleX,GrilleY,pup,'Fringe');
        P=P*indice_carte;
    case 'L'
        pup=true(size(GrilleX));
        P=NaN(length(GrilleX(:)),length(indice_alignement));
        P(pup,:)=Legendre2D(indice_alignement,GrilleX,GrilleY)';
        P=P*indice_carte;
end
NPix=size(P,1);
Por=P;

%On copie les coeffs d'alignement NSSPP fois chacun de manière contigue
P=reshape(repmat(P,1,NSSPP),NPix,Ncoeffs*NSSPP);
%Ensuite on applique le masque pour chaque SSPP
MSK=squeeze(reshape(~isnan(TableData),NPix,NSSPP));
MSKn=double(MSK);MSKn(MSKn==0)=NaN;
P=P.*reshape(repmat(MSKn,Ncoeffs,1),NPix,Ncoeffs*NSSPP);
Por1=P;
%Mise en forme des compensateurs fixes
Mean_Comp=zeros(NPix,Ncompf);
if sz(1)~=0
    Mean_Comp=mean(Compfixes,2,'omitnan');
    Compfixes2=-Compfixes+repmat(Mean_Comp,[1 NSSPP 1]);
    Mean_Comp=squeeze(Mean_Comp);
end
%Du coup, on peut construire la fonction d'erreur facilement
T=reshape(TableData,NPix,NSSPP);
P(isnan(P))=0;
TZ=T; 
%Version of subpupils with zeros to allow summing in mean map
TZ(isnan(TZ))=0;

%% 3300339321
eta=sum(MSK,2);
Por2=Por1./eta;
Por2(isinf(Por2))=0;


%Initialisation de la matrice à inverser avec les la moyenne des
%compensateurs sur la SPP1. Pour cette sous-pupille, on ne prend pas en
%compte les compensateurs de la première sous-pupille, sauf pour les
%compensateurs fixes.
P1=Por2;
if sz(1)~=0
    P1=[P1 squeeze(Compfixes2(:,1,:))];
end
P1(isnan(P1))=0;  
MSKtmp=repmat(MSK(:,1),1,NSSPP*Ncoeffs+Ncompf);
P2=[reshape(P1(MSKtmp),[],NSSPP*Ncoeffs+Ncompf)];
MSK0=zeros(NPix,NSSPP*Ncoeffs);
%Calcul de la matrice des compensateurs P2
%setup waitbar
hchrgt = waitbar(1,['Sous-pupille : 1/' num2str(NSSPP)],'Name','Stitching en cours');
for q=2:NSSPP
    waitbar(q/NSSPP,hchrgt,sprintf('Sous-pupille : %d/%d',q,NSSPP))
    MSKq1=double(repmat(MSK(:,q),1,Ncoeffs));
    MSKq2=[MSK0(:,1:(q-1)*Ncoeffs) MSKq1 MSK0(:,q*Ncoeffs+1:NSSPP*Ncoeffs)];
    P11=Por1.*MSKq2;
    P11=Por2-P11;
    if sz(1)~=0
        P1=[P11 squeeze(Compfixes2(:,q,:))];
    else
        P1=P11;
    end
    P1(isnan(P1))=0;  
    MSKtmp=repmat(MSK(:,q),1,NSSPP*Ncoeffs+Ncompf);
    P2=[P2;reshape(P1(MSKtmp),[],NSSPP*Ncoeffs+Ncompf)];
end
delete(hchrgt)

P2(isnan(P2))=0;
% On enlève les termes correspondants à la première sous-pupille qui ne sont
% pas calculés.
P2=P2(:,Ncoeffs+1:Ncoeffs*NSSPP+Ncompf);
M1=P2'*P2;
T1=T-mean(T,2,'omitnan');
T2=reshape(T1(MSK),[],1);
b1=P2'*T2;
% Calcul de l'amplitude des compensateurs cs1
cs1=M1\b1;
cs1=[zeros(Ncoeffs,1);cs1];
% Calcul de la carte corrigée
Mean_Comp(isnan(Mean_Comp))=0;  
MMap2=mean(T,2,'omitnan')+P*cs1(1:NSSPP*Ncoeffs)./eta+Mean_Comp*cs1(NSSPP*Ncoeffs+1:end);
% Projection de la carte reconstruite sur les compensateurs et suppression
% des compensateurs moyens
a2=Por(~isnan(MMap2)&pup(:),:)\MMap2(~isnan(MMap2)&pup(:)); %pup added 31/03/2020
map2=reshape(MMap2-Por*a2,size(GrilleX));

%Build mismatch as std of all SPP
RMap2=zeros(size(MMap2));
for k=1:NSSPP
    comp=zeros(NPix,Ncompf);
    if sz(1)~=0
       comp=squeeze(Compfixes(:,k,:));
    end
    RMap2(:,k)=(TZ(:,k)+Por*cs1([1:Ncoeffs]+(k-1)*Ncoeffs)+comp*cs1(NSSPP*Ncoeffs+1:end)).*MSKn(:,k);
end
mismatch2=reshape(std(RMap2,0,2,'omitnan'),size(GrilleX));
mismatch2(mismatch2==0)=NaN;

%calcul rms du mismatch 
std_mismatch=std(mismatch2,[],'all','omitnan');
moy_mismatch=mean(mismatch2,'all','omitnan');

end