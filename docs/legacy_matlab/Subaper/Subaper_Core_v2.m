%Auteur : Camille Frapolli a partir de 735160

function [map,mismatch,Cs,a,RMS_Diff_Matrix,sppAdjMatrix]=Subaper_Core_v2(TableData, Type, indice_alignement,indice_carte,wb)
%%
if(nargin < 4)
    indice_carte = 1;
end
%
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
    case 'LM' % => a utiliser sur pieces carrees pour ne pas utiliser le terme L11 comme compensateur libre
        pup=true(size(GrilleX));
        PP=NaN(length(GrilleX(:)),length([1:6]));
        PP(pup,:)=Legendre2D([1:6],GrilleX,GrilleY)';
        PP(:,4)=PP(:,4)+PP(:,6); %Legendre modifies ou L20_new = L20 + L02 (somme des termes cylindriques)
        P=PP(:,indice_alignement);
        P=P*indice_carte;
end
Por=P;
%On vectorise avec le bon nombre de variables
if indice_carte==1
    Ncoeffs=length(indice_alignement);
else
    Ncoeffs=size(indice_carte,2);
end
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

%Calcul de la matrice différence RMS
nbCartes = NSSPP;
RMS_Diff_Matrix = nan(nbCartes, nbCartes);

for i = 1:nbCartes
    for j = i+1:nbCartes
        diffMatrix = TableData(:, :, i) - TableData(:, :, j);
        RMS_Diff_Matrix(i, j) = std(std(diffMatrix,'omitnan'),'omitnan');
        RMS_Diff_Matrix(j, i) = RMS_Diff_Matrix(i, j); % Symmetrique
    end
end




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
b=zeros(1,(NSSPP-1)*Ncoeffs);
BigM=zeros(Ncoeffs,Ncoeffs,length(MSK));
eta=sum(MSK,2);
%setup waitbar
if nargin<5
    hchrgt = waitbar(1,['Sous-pupille : 1/' num2str(NSSPP)],'Name','Stitching en cours');
end
%precomputing Zernikes square
for m=1:Ncoeffs
    for mp=1:Ncoeffs
        if mp>=m
            BigM(m,mp,:)=Por(:,m).*Por(:,mp);
        else
            BigM(m,mp,:)=BigM(mp,m,:); %amélioration marginale...
        end
    end
end
%% 
for k=2:NSSPP
    if nargin<5
        waitbar(k/NSSPP,hchrgt,sprintf('Sous-pupille : %d/%d',k,NSSPP))
    elseif ~isempty(wb)
        wb.Message='Construction de la matrice de coefficients';wb.Value=k/NSSPP;
    end
    MSKBase=MSK(:,k) & pup(:);
    for m=1:Ncoeffs
        for kkk=1:NSSPP %corrected 31/03/2020
            TempMSK=MSK(:,kkk) & MSKBase;
            if any(TempMSK)
                b(Ncoeffs*(k-2)+m)=b(Ncoeffs*(k-2)+m)+sum((1./eta(TempMSK)-(k==kkk)).*Por(TempMSK,m).*T(TempMSK,kkk));
            end
        end
        for kp=2:NSSPP
            for mp=1:Ncoeffs
                TempMSK=MSK(:,kp) & MSKBase;
                if any(TempMSK)
                    M(Ncoeffs*(k-2)+m,Ncoeffs*(kp-2)+mp)=sum(((k==kp)-1./eta(TempMSK)).*squeeze(BigM(m,mp,TempMSK)),'omitnan');  
                end
            end
        end
    end
end
if nargin<5
    delete(hchrgt)
end

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
    sppAdjMatrix(:,:,k) = reshape(RMap(:,k),size(GrilleX));
end
mismatch=reshape(std(RMap,0,2,'omitnan'),size(GrilleX));
mismatch(mismatch==0)=NaN;



end