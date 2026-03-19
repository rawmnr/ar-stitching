function A = basezernike(theta,r,term,lambda,modo)

% La fonction basezernike évalue le 'term'-ième polynôme de Zernike à la
% coordonnée polaire (r,theta) à la longueur d'onde lambda
%
% Le paramètre fact peut être 'pv' ou 'rms' s'il est nécessaire que le
% polynome soit normalisé en valeur unité de pv ou rms. Par défaut le
% paramètre vaut 'pv'.

% Entrées:
% theta: coordonnée polaire theta en radian
% r: coordonnée polaire r
% term: terme du polynôme à évaluer, indice (entier) du zernike qu'on veut
% évaluer, ex: pour Z3, écrire 3
% lambda: longueur d'onde
% modo: 'pv' ou 'rms'
% Sortie:
% Evaluation des polynômes de Zernike aux coordonnées (theta,r)

switch nargin
    case 4
        modo='pv';
end

[zn,zm]=zernikeISO2nm(term+1);
zn=zn(term+1);  % +1 Parce que ISO Z_0 est le premier
zm=zm(term+1);  % +1 Parce que ISO Z_0 est le premier
Znm=zernike(zn,zm,modo);
if zm>=0
   A=polyval(Znm,r).*cos(zm*theta);
else
   A=polyval(Znm,r).*sin(-zm*theta);
end

A=A*lambda;

end

