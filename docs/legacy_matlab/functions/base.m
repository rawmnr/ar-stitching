function A = base(coord1,coord2,term,lambda,mode)

% Cette fonction évalue le term-ième polynôme de Zernike ou de Legendre à
% la longueur d'onde lambda aux coordonnées [coord1,coord2]
% Entrées: grille
% [coord1,coord2] = [X,Y] pour le mode 'L'
%                 = [theta,r] pour le mode 'Z' 2 vecteurs colonne 1xn ou 2
% matrices nxm.
% term: terme du polynôme à évaluer, indice (entier) du zernike qu'on veut
% évaluer, ex: pour Z3, écrire 3
% lambda: longueur d'onde
% mode: 'Z' ou 'L', Zernike ou Legendre
% Sortie:
% Evaluation des polynômes de Zernike ou de Legendre en les points de la
% grille

if strcmp(mode,'Z')
    A = basezernike(coord1,coord2,term,lambda);
elseif strcmp(mode,'L')
    A = baselegendre(coord1,coord2,term,lambda);
end

end