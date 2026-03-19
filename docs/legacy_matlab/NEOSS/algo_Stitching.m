%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Code de stitching NEOSS                 %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% A.DOCHE 20201114     
%%%%%%%%%%%%%%%%% CFR, nettoyage 12/07/23                  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Certaines fonctions son hérités de la   %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Toolbox MSO.                            %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% ENTREES : Path_Param, Path_cartes :     %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Chemins vers le ficier text de paramètre, %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% chemin vers le dossier contenant les    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% cartes                                  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% SORTIES : map, mismatch                 %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Carte stitchée, carte de mismatch       %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [map, Mismatch,TableData,RMS_Diff_Matrix,sppAdjMatrix,sspStack]=algo_Stitching(Path_Param,pos_sp, Path_cartes,UseCarteRandom,wb)


if nargin==5;wb.Message='Lecture des cartes et paramètres';end
[NEOSS_Param,TableData]=lectureParametresNEOSS(Path_Param, Path_cartes,pos_sp);
%Code
%Cette fonction sert à créer la carte moyenne des SSPP pour supprimer les
%HF de CS avant de réaliser le FIT MLR
%[carte_random]=calculCarteRandom(TableData,NEOSS_Param.Coord1,NEOSS_Param.Coord2,NEOSS_Param.limit*NEOSS_Param.RpupilleCS);

if nargin==5;wb.Message='Construction de la carte random';wb.Value=0.1;end
%[carte_random]=calculCarteRandom(TableData,NEOSS_Param.Coord1,NEOSS_Param.Coord2,NEOSS_Param.limit.*NEOSS_Param.RpupilleCS); %Versionend
if nargin<4 || UseCarteRandom
    carte_random=calculCarteRandomLegacy(TableData,NEOSS_Param); %Version
else
    carte_random=zeros(NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
end
    %NEOSS originale

%Cette fonction réalise le FIT de TP, CS et des coefficients d'alignement.
if nargin==5;wb.Message='Construction de la matrice de coefficients';wb.Value=0.2;end
[x,carte_Instrument]=MLR(TableData,carte_random,NEOSS_Param);


%Cette fonction recolle les SSPP pour réaliser la reconstruction de la
%carte finale
if nargin==5;wb.Message='Construction de la carte stitchée';wb.Value=0.7;end
[map,Mismatch,RMS_Diff_Matrix,sppAdjMatrix,sspStack]=stitchingSspp(x,NEOSS_Param,TableData,carte_Instrument);

if nargin==5;wb.Message='Sortie des données';wb.Value=1;end


end