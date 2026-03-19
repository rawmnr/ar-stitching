function [Zrms,meanrms,stdrms,Zcdf]=FrepartitionZrms_MF(Moy,Max,STD,q,type)
%%%%%%%%%%%%%%%%%% valable uniquement pour MF

% Calcule la fonction de répartition d'une somme quadratique en fonction de
% la distribution de chacun de ses termes. Chaque terme suit une loi
% définie par la moyenne Moy et l'écart type STD. q donne le nombre de
% tirage que l'on effectue pour trouver la distribution de la somme
% quadratique. Type définit le type de loi de chaque paramètre.
%Tous les paramètres d'entrée sont des vecteurs colonne à l'exeption des 3
%derniers
for m=1:q
	n=size(Moy);
    Z=random(type,Max-Max,Max)+rndlnrep(Moy,STD,rand(n))';%erreur ITB
    Zrms(m:m)=(Z'*Z)^0.5; %Zrms contient la somme quadratique pour chaque tirage des erreurs inclues dans Z
end
meanrms=mean(Zrms);
stdrms=std(Zrms);
Z2 = paretotails(Zrms, 0, 0.999); % interpole la distribution définie par les éléments de Zrms dans l'objet Z
Zcdf = icdf(Z2, [0.5 0.95 0.997]); % calcule l'erreur cumulée à 50%, 95% et 99.7%
Z50 = Zcdf(1);
Z95 = Zcdf(2);
Z997 = Zcdf(3);
end