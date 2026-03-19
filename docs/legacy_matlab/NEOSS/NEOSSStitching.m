%%Fonction SCRIPT

%Cette fonction permet de calculer la carte stitchée a partir des autres

%Entrées : cartes a stitcher
%Sorties : carte stitchée

function [CarteOut,MM,RMS_Diff_Matrix,sppAdjMatrix,sspStack]=NEOSSStitching(Maps,pos_sspp,Path_Param,UseCarteRandom,wb)
    arguments
        Maps cell
        pos_sspp cell
        Path_Param 
        UseCarteRandom
        wb
    end
    
    %Initialise les cartes de sortie
    Param=readcell(Path_Param);
    CarteOut=GenerateMapObject(NaN(Param{2,1}),Param{1,1},Param{5,1},'Stiched_Map');    
    MM=GenerateMapObject(NaN(Param{2,1}),Param{1,1},Param{5,1},'Mismatch');    
    if nargin==5
        [CarteOut.carte, MM.carte, table, RMS_Diff_Matrix,sppAdjMatrix,sspStack]=algo_Stitching(Path_Param, pos_sspp,Maps,UseCarteRandom,wb);
    else
        [CarteOut.carte, MM.carte, table, RMS_Diff_Matrix,sppAdjMatrix]=algo_Stitching(Path_Param, pos_sspp,Maps,UseCarteRandom);
    end
    %[R1, R2]=algo_stitching_repoNeoss(Path_Param, [fileparts(Maps{1}),'\'])
end

