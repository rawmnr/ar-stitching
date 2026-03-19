function [carteCirculaire]=troncatureCercle(carte, radiuspix)
    if nargin < 2
        radiuspix = floor(size(carte, 1) / 2);
    end
    sz=size(carte);
    i0=floor(sz(1)/2)+1;
    j0=floor(sz(2)/2)+1;
    carteCirculaire=nan(sz(1),sz(2));

    for i=1:sz(1)
        for j=1:sz(2)
            if (i-i0)^2+(j-j0)^2<=radiuspix^2
                carteCirculaire(i,j)=carte(i,j);
            end
        end
    end
end