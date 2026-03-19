%Cette fonction lit les coordonnées des SSPP
    function [Coord1,Coord2]=getSsppCoordinates(position_sspp)
        for ii=1:length(position_sspp)
            Coord1(ii)=-position_sspp{ii,1};
            Coord2(ii)=-position_sspp{ii,2};
        end
    end