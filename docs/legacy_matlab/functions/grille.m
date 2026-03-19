function [coord1,coord2] = grille(carte,rho,mode,param)

% Cette fonction crée deux grilles [X,Y] pour évaluer les polynômes de 
% Zernike ou [theta,r] pour évaluer les polynômes de Legendre
%
% Entrées:
% carte: carte des altitudes obtenue avec la fonction read_opd
% rho: borne maximale de la grille normalisée par le rayon de la pièce
% mode: 'Z' ou 'L', Zernike ou Legendre
% param =  [trans_X, trans_Y, trans_rad, angle_rot]
%       trans_X, trans_Y et trans_rad sont en coordonnées normalisées
%       angle_rot est en degré
% Sorties: grille
% [coord1,coord2] = [X,Y] pour le mode 'L'
%                 = [theta,r] pour le mode 'Z'

switch nargin
    case 1
        rho=1;
        mode='Z';
        param=[0 0 0 0];
    case 2
        mode='Z';
        param=[0 0 0 0];
    case 3
        param=[0 0 0 0];
end

trans_X = param(1);
trans_Y = param(2);
trans_rad = param(3);
angle_rot = param(4);

if strcmp(mode,'Z') | strcmp(mode,'L')
    [n,m] = size(carte);
    dx = 2/min(n,m);
    
    % On évalue la grille au milieu du pixel
    [X,Y] = meshgrid(((1:m)-m/2-0.5)*dx*rho,((1:n)-n/2-0.5)*dx*rho);
    Y = flipud(Y);
    
    % Translation
    if trans_X ~= 0
        X = X + trans_X;
    end
    if trans_Y ~= 0
        Y = Y + trans_Y;
    end
    
    if strcmp(mode,'L')
        coord1 = X;
        coord2 = Y;
    end
    
    if strcmp(mode,'Z')
        % Translation radiale
        if trans_rad ~= 0
            X = X + trans_rad;
        end
        
        [theta,r] = cart2pol(X,Y);
        theta = mod(theta,2*pi);
        
        % Rotation
        if angle_rot ~= 0
            theta = mod(theta + angle_rot*pi/180,2*pi);
        end
        
        coord1 = theta;
        coord2 = r;    
    end
    
else
    error('Le mode prend comme paramètre ''L'' pour Legendre ou ''Z'' pour Zernike');
end

end

