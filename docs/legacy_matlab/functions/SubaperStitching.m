%%Fonction SCRIPT

%Cette fonction permet de calculer la carte stitchée a partir des autres

%Entrées : cartes a stitcher
%Sorties : carte stitchée

function [CarteOut,MM]=SubaperStitching(Maps, Type, Compensateurs)
    arguments
        Maps cell
        Type {mustBeMember(Type,["L","Z"])} = "L"
        Compensateurs (1,:) double = [1:3]
    end
    
    dim=length(Maps);
    %On convertit les cartes au bon formata
    for i=1:dim(1)
        TableData(:,:,i)=Maps{i}.carte;
    end
    %Initialise les cartes de sortie
    CarteOut=Maps{1};
    CarteOut.titre='Stiched Map';
    MM=Maps{1};
    MM.titre='Mismatch';
    [CarteOut.carte,MM.carte]=Subaper_Core_v2(TableData, Type, Compensateurs);
    CarteOut=updateProp(CarteOut);
    MM=updateProp(MM);

end