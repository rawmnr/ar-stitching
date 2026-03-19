%%Fonction SCRIPT

%Cette fonction permet de calculer la carte stitchée a partir des autres

%Entrées : cartes a stitcher
%Sorties : carte stitchée

function [CarteOut,MM,RMS_Diff_Matrix,sppAdjMatrix]=SubaperStitching(Maps, Type, Compensateurs,wb)
    arguments
        Maps cell
        Type {mustBeMember(Type,["L","Z","LM"])} = "LM"
        Compensateurs (1,:) double = [1:3]
        wb (1,1) = 0
    end
    
    dim=length(Maps);
    %On convertit les cartes au bon formata
    for i=1:dim(1)
        TableData(:,:,i)=Maps{i}.carte;
    end
    %Initialise les cartes de sortie
    CarteOut=Maps{1};
    CarteOut.titre='Stiched_Map';
    MM=Maps{1};
    MM.titre='Mismatch';
    if wb~=0
        [CarteOut.carte,MM.carte,Cs,a,RMS_Diff_Matrix,sppAdjMatrix]=Subaper_Core_v2(TableData, Type, Compensateurs,1,wb);
    else
        [CarteOut.carte,MM.carte,Cs,a,RMS_Diff_Matrix,sppAdjMatrix]=Subaper_Core_v2(TableData, Type, Compensateurs);
    end
    CarteOut=updateProp(CarteOut);
    MM=updateProp(MM);

end