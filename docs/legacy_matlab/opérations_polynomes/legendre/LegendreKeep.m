function CarteSortie=LegendreKeep(carteEntree,vec,OrdreAnalyse)
%Fonction SCR
%Fonction qui donne une carte projetée sur les zernikes specifiés

if nargin<3
    OrdreAnalyse=vec;
end

[a,L]=LegendreDecomposeMap(carteEntree,OrdreAnalyse);
pup=~isnan(carteEntree.carte);

for i=1:length(vec)
    idx(i)=find(vec(i)==OrdreAnalyse);
end

CarteSortie=carteEntree;
CarteSortie.titre=strcat(carteEntree.titre,'_KeepL');
%'keep' les n premiers Zernike
CarteSortie.carte(pup)=L(:,idx)*a(idx);
CarteSortie.carte(~pup)=NaN;

CarteSortie=updateProp(CarteSortie);

end
