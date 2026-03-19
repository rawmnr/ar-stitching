%cartesSimulees créé des cartes, à la données de
%coefficients de Zernike, et peu rajouter des défauts HVHF.
%Actuellement, elle créé 66 nm de défauts BF et 0.5 nm de défauts HVHF
%ENTREES:
% - resolution : resolution du tableau 
% - Rpupille : rayon du miroir à générer en mm
% - repertoire_cartes : adresse du dossier ou sera contenue la carte
% - nom_carte : nom du fichier de la carte
% - CoefBF : vecteur contenant les coefficient de Zernikes de la carte à
%générer.
% - HF_YesNo : booleen, indiquant si l'on souhaite générer ou non des HF et VHF.
% - HVHFlevel : ~Amplitude RMS en nm des défauts de surface HVHF
% - LMFLevel : Amplitude RMS en nm des défauts de surface BMF
%%% - CoefMiroir : Coefficient de la forme théorique du miroir.
%SORTIES :
% - carteGen: carte Aléatoire retournée

function carteGen=cartesSimulees(resolution,Rpupille, lambda, repertoire_cartes, nom_carte, CoefBF, HF_YesNo, HVHFlevel, LMFLevel, main_folder)

%Creation du tableau
Carte=zeros(resolution,resolution);

%Generation des HVHF
if HF_YesNo
    CarteHVHF=GenerateurCartesHF(HVHFlevel,resolution, Rpupille, main_folder);
else
    CarteHVHF=zeros(resolution,resolution);
end

% Generation des BF
%Le bloc ci-dessous génère la carte des BF à partir du vecteur de
%coefficients donné en paramètre.
[coord1,coord2] = grille(Carte);
CarteBF=troncatureCercle(generer_carte(coord1,coord2,CoefBF,lambda,'Z'));

if LMFLevel==0
    carteGen=CarteBF+CarteHVHF;
else
    [a,b]=size(CarteBF);
    A=reshape(CarteBF,1,a*b);
    A=A(~isnan(A));
    c=sqrt(sum(sum((A-mean(A)).^2))/length(A));
    carteGen=(LMFLevel/c)*CarteBF+CarteHVHF;
end;
clearvars A c;
end