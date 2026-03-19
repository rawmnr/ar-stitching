function CarteSortie=ZernKeep(carteEntree,vec,OrdreAnalyse,RNorm,type)
%Fonction SCR
%Fonction qui donne une carte projetée sur les zernikes specifiés

if nargin<3
    OrdreAnalyse=vec;
end
if nargin<4
    RNorm=carteEntree.rayon;
end
if nargin<5
    type='fringe';
end

[a,Z,~,~,~,~,pup]=ZernikeDecomposeMap(carteEntree,OrdreAnalyse,RNorm,type);

for i=1:length(vec)
    idx(i)=find(vec(i)==OrdreAnalyse);
end

CarteSortie=carteEntree;
CarteSortie.titre=strcat(carteEntree.titre,'_Keep');
%'keep' les n premiers Zernike
CarteSortie.carte(pup)=Z(:,idx)*a(idx);
CarteSortie.carte(~pup)=NaN;

CarteSortie=updateProp(CarteSortie);

end
