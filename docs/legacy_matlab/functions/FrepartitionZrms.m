function [Zrms,meanrms,stdrms,Zcdf]=FrepartitionZrms(Moy,STD,ind,q,type)
% Calcule la fonction de répartition d'une somme quadratique en fonction de
% la distribution de chacun de ses termes. Chaque terme suit une loi
% définie par la moyenne Moy et l'écart type STD. q donne le nombre de
% tirage que l'on effectue pour trouver la distribution de la somme
% quadratique. Type définit le type de loi de chaque paramètre.
for m=1:q
    Z=random(type,Moy(ind),STD(ind)); 
    Zrms(m:m)=(Z'*Z)^0.5; %Zrms contient les évènements de la somme quadratique des polynômes de Zernike définis 
    %par les indices ind, obéissants à une loi normale de moyenne Moy et d'écart type STD
end
meanrms=mean(Zrms);
stdrms=std(Zrms);
Z = paretotails(Zrms, 0, 0.999); % interpole la distribution définie par les éléments de Zrms dans l'objet Z
Zcdf = icdf(Z, [0.5 0.95 0.997]); % calcule l'erreur cumulée à 50%, 95% et 99.7%
Z50 = Zcdf(1);
Z95 = Zcdf(2);
Z997 = Zcdf(3);
%%%% FIN AJOUT %%%%

end