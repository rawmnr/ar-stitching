
function carte=readOPD(chemin)
% Auteur : Divers
%
% La fonction read_opd permet de lire un fichier .opd sur MATLAB.
% La carte obtenue a ses axes qui ont été ajustés pour MATLAB et réprésente
% la matrice des altitudes de la carte de phase.
% !!! La grille utilisée pour définir la carte est telle que les points de la carte
% sont positionnés à l'intersection des pixels (choix historique fait sur Warpp).
% Le rayon de la pupille est donc défini par le bord des pixels situés à la frontière 
% de la pupille. 
%
% Syntaxe :
% carte=read_opd(chemin)
%
% ENTREES:
% carte=read_opd('C:\monchemin\');.
% SORTIE:
% carte: matrice des altitudes de la carte (en nm)
fid = fopen(chemin,'r');

MAP.titre = (char(fread(fid,20,'char')))';
MAP.titre=strtrim(MAP.titre);% titre de la carte
MAP.dummy1 = char(fread(fid,1,'char')); 
MAP.date = (char(fread(fid,17,'char')))';          % date et heure de création
MAP.dummy2 = char(fread(fid,1,'char'));
MAP.commentaires = (char(fread(fid,104,'char')))';
MAP.commentaires=deblank(MAP.commentaires);% commentaires
MAP.dummy3 = char(fread(fid,1,'char'));            % Lettre 'O' obligatoire	
MAP.logiciel = (char(fread(fid,16,'char')))';      % Logiciel d'origine

MAP.lambda = fread(fid,1,'float');              % longueur d'onde de la carte
MAP.wedge = fread(fid,1,'float');               % ne sert à rien dans Warpp
MAP.rayon = fread(fid,1,'float');               % rayon de la pupille en mm
MAP.dummy4 = fread(fid,3,'float');              % 3 réels qui ne sont pas utilisés
MAP.fnumber = fread(fid,1,'float');             % nombre d'ouverture
MAP.fiducialpoints = fread(fid,8,'float');      % nombres fiduciaux
MAP.cobsc = fread(fid,1,'float');               % rapport d'obturation centrale
MAP.apod = fread(fid,2,'float');                % terme d'apodisation
MAP.dummy5 = fread(fid,2,'float');              % rien

MAP.signature = fread(fid,1,'short');           % indique la simple ou double précision
MAP.largeur = fread(fid,1,'short');             % nombre de colonnes de la carte
MAP.hauteur = fread(fid,1,'short');             % nombre de lignes de la carte
MAP.aperturecode = fread(fid,1,'short');
MAP.fringepoints = fread(fid,1,'short');
MAP.dummy6 = fread(fid,2,'short');
MAP.flag = fread(fid,1,'short');           % facteur de précision, utilisés uniquement pour les cartes simple précision
MAP.dummy7 = fread(fid,3,'short');
MAP.gridsize = fread(fid,1,'short');            % ne sert à rien !!
MAP.dummy8 = fread(fid,1,'short');
MAP.dummy9 = fread(fid,7,'short');

if (MAP.rayon==0)
    MAP.rayon = 1e-14 ;
end

% LECTURE DE LA MATRICE DES ALTITUDES DE LA CARTE DE PHASE
N = MAP.largeur*MAP.hauteur ;
if (MAP.signature==18)
MAP.rawdata = fread(fid,N,'short');
end
if (MAP.signature==52)
MAP.rawdata = fread(fid,N,'float');
end

%gestion du cas simple précision
if (MAP.signature==18)
    % ne garder que les données valides de la rawcarte
    masque = find(MAP.rawdata~=32767) ; 
    MAP.data = MAP.rawdata(masque) ;
    % mettre en forme pour l'affichage
    masque = find(MAP.rawdata==32767) ;
    MAP.carte = reshape(MAP.rawdata,MAP.hauteur,MAP.largeur);
    MAP.carte(masque) = nan ;
    % valeurs en nm au lieu de lambda
    MAP.coeffmult=32760;
    MAP.carte = MAP.carte/MAP.coeffmult*MAP.lambda ;
    MAP.data = MAP.data/MAP.coeffmult*MAP.lambda ;
    MAP.stat.min = min(min(MAP.data));
    MAP.stat.max = max(max(MAP.data));
end

%gestion du cas double précision
if (MAP.signature==52)
    % ne garder que les données valides de la rawcarte
    masque = find(MAP.rawdata~=Inf) ; 
    MAP.data = MAP.rawdata(masque) ;
    % mettre en forme pour l'affichage
    masque = find(MAP.rawdata==Inf) ;
    MAP.carte = reshape(MAP.rawdata,MAP.hauteur,MAP.largeur);
    MAP.carte(masque) = nan ;
    % valeurs en nm au lieu de lambda
    MAP.carte = MAP.carte*MAP.lambda ;
    MAP.data = MAP.data*MAP.lambda ;
    MAP.stat.min = min(min(MAP.data));
    MAP.stat.max = max(max(MAP.data));
end

% CALCUL DE CERTAINES PROPRIETES DE LA CARTE
% GENERATION DES SORTIES DE LA FONCTION READOPD
tmp = size(MAP.data) ;
MAP.stat.n = tmp(1,1) ;
MAP.stat.piston = mean(MAP.data) ;
MAP.stat.ptv = MAP.stat.max - MAP.stat.min ;
MAP.stat.rms = std(MAP.data,1) ;

MAP.stat.piston;
MAP.stat.ptv;
MAP.stat.rms;

%on ajoute dans l'objet la grille appropriée
m=MAP.largeur; %(map size)
n=MAP.hauteur;
dx=2/m;
dy=2/m;
[X,Y] = meshgrid(  ((1:m)-m/2-0.5)*dx   ,  ((1:n)-n/2-0.5)*dy      );
MAP.grilleX=X;
MAP.grilleY=Y;

carte1 = MAP;
carte =flip(carte1 , 1 ) ; 
fclose(fid);
end