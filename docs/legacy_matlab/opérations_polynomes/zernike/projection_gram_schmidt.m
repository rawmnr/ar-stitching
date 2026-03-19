function [coefGS,transfertZGS2Z]=proj_romain(carte_opd,Nz,varargin)
% Auteur : divers
%
% La fonction proj_romain permet de réaliser la projection de cartes dont
% la pupille n'est pas circulaire sur une base orthonormée et d'obtenir les
% coefficients de décomposition. La base orthonormée en question est
% obtenue par procédé de Gram-Schmidt à partir des polynomes de zernike
% définis seulement sur le masque de la carte.
%
% ENTREE :
%   - carte_opd : Placer la carte à traiter au format opd
%   - Nz : Nombre de Zernike sur lesquels projeter (maximum 36)
%   - varargin : coefficient d'ajustement si le masque sort du disque
%   unité (erreur lors de la génération des zernike).
%
% SORTIE : 
%   - coef : coefficients de la décomposition dans la base de Gram-Schmidt
%   - transfertZGS2Z : Matrice de transfert de la base de GS vers celle de
%   Zernike.
                        
%------------------ DONNEES

%---- DEFINITION DES PARAMETRES       
    carte=carte_opd.carte;
    dim=size(carte);
    mask=isnan(carte)<1;                        %génération du masque
    Xs=carte_opd.grilleX(mask);                 %grille normalisée selon X
    Ys=carte_opd.grilleY(mask);                 %grille normalisée selon X
    
%---- REAJUSTEMENT DE LA CARTE    
    if ~isempty(varargin)                                        
        Xs=Xs*varargin{1};
        Ys=Ys*varargin{1};
    end

%---- BASE DE CALCUL DES PERFORMANCES
    [Z,~,~]=zernike_fcn3([1:Nz],Xs,Ys,true(size(Xs)),'fringe');   % Zernike basis on the clear aperture

%------------------ ORTHONORMALISATION PAR GRAM-SCHMIDT (GS)

%---- SUPPRESSION DE LA VALEUR MOYENNE DES ZERNIKE
    Zmean=ones(size(Z)).*mean(Z);
    Zmean(:,1)=0;                               % On ne touche pas au piston
    Z=Z-Zmean;

%---- NORMALISATION DES ZERNIKES
    Zrms=zeros(1,Nz);                                              
    for i=1:Nz                                  % Calcul des normes
        Zrms(i)=norm(Z(:,i));
    end
    Z=Z./Zrms;                                  % normalisationdes vecteurs

%---- CREATION DE LA NOUVELLE BASE PAR GS ET NORMALISATION
    ZGS = GramSchmidt(Z);                      % Orthogonalisation de la base 
    ZGSrms=zeros(1,Nz);
    for i=1:Nz
        ZGSrms=norm(ZGS(:,i));
    end
    ZGS=ZGS./ZGSrms;                            % Normalisation de la base par le RMS

%---- MATRICES DE TRANSFERT
    transfertZ2ZGS=ZGS'*Z;                                  
    transfertZGS2Z=pinv(transfertZ2ZGS);
                            
%------------------ Projection et obtention des coefficients

    carte_mask=carte(mask);
    coefGS=ZGS'*carte_mask;                     % Projection sur la nouvelle base (GS)
    
%   coefZ=transfertZGS2Z*coefGS;                % Coefficients dans la base de Zernike
end