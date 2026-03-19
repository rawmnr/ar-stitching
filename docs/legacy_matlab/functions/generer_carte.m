function carte = generer_carte(coord1,coord2,coefficients,lambda,mode)

% Cette fonction permet de tracer les polynômes 2D
% Entrées:
% [coord1,coord2] = [X,Y] pour le mode 'L'
%                 = [theta,r] pour le mode 'Z'
% 2 vecteurs colonne 1xn ou 2 matrices nxn.
% coefficients: liste des valeurs des coefficients en lambda de Zernike ou Legendre ex: 1*Z0+0*Z1+3*Z2<->[1,0,3]
% lambda: longueur d'onde (en m ou nm)
% mode: 'Z' ou 'L', Zernike ou Legendre


carte = zeros(size(coord1));

for i = 1:length(coefficients)
    carte = carte + coefficients(i)*base(coord1,coord2,i-1,lambda,mode);
end

end