%%Fonction SCRIPT

%Calcul du volume matière d'une carte

%Entrées :  carte
%Sorties :  Volume matière a pârtir du minimum.

function V=VMMat(carte,rayon)

pist=mean(carte(:),'omitnan');
%on met le minimum a zero
mat=carte-min(carte(:));
msk=~isnan(mat);
pxsize=2*rayon./(size(carte,1)-1);
%puis on retranche le volume a partir du fond
V=sum(1e-6*mat(msk))*pxsize*pxsize; %mat est en nm

end