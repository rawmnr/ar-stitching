%%Fonction SCRIPT

% Cette fonction évalue une carte dont l'échelle a été changée selon les
% axes x et y par des coefficients Ex et Ey 

% Entrées: Carte, coefficients de scaling Ex et Ey 
% Sortie: nouvelle Carte 

function CarteOut=scale(Carte , Ex , Ey)

[x , y] = meshgrid(linspace(1 ,Carte.largeur , Carte.largeur ) ,linspace(1 ,Carte.hauteur , Carte.hauteur )  );
[xeq , yeq] = meshgrid(linspace(1 ,Carte.largeur , ceil(Carte.largeur*Ex) ) , linspace(1 ,Carte.hauteur , ceil(Carte.hauteur*Ey) ) );
vq = griddata(x,y ,Carte.carte,xeq,yeq , 'cubic');


if Ex>1
    scaled=vq(:, round((Ex-1)*Carte.largeur/2 +1  ): round((Ex+1)*Carte.largeur/2 ));
else
    sizex=ceil((Carte.largeur-size(vq , 2))/2);
    sizey=ceil((Carte.hauteur- size(vq , 1))/2);
    scaled=padarray(vq , [0 sizex] , NaN ,  'both');
end 

if Ey>1
    scaled=scaled(round((Ey-1)*Carte.hauteur/2  +1 ): round((Ey+1)*Carte.hauteur/2 ), :);
else
    sizex=ceil((Carte.largeur-size(scaled , 2))/2);
    sizey=ceil((Carte.hauteur- size(scaled , 1))/2);
    scaled=padarray(scaled , [sizey 0] , NaN ,  'both');
end

CarteOut=Carte;
CarteOut.carte=scaled;
    
end 