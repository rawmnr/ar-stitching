%%Fonction SCRIPT

% Cette fonction sert à changer la dimension d'une carte. Si la dimension
% finale entrée en paramètre est plus petite, la fonction crop la carte
% en entrée. Sinon, elle élargie le cadre en rajoutant des NaN. Si la
% différence de taille entre la carte initale et la carte finale est
% imapire, la carte finale est décalée d'un pixel vers "le coin supérieur
% gauche"

% Entrées: une carte à resizer (tableau matlab), la résolution de la carte finale

% Sortie:  tableau matlab de la carte redimensionnée.

function resizedCarte=resizeCarte(carte, resolutionFinale)

sz=size(carte);
sz=sz(1);

if sz>resolutionFinale
    resizedCarte=raccourcirCarte(carte, resolutionFinale);
elseif sz<resolutionFinale
    resizedCarte=ElargirCarte(carte, resolutionFinale);
else
    resizedCarte=carte;
end

function CarteRaccourcie=raccourcirCarte(carte, ResolutionFinale)
    CarteRaccourcie=nan(ResolutionFinale,ResolutionFinale);
    sz=size(carte);
    sz=sz(1);
    xA=floor((sz-ResolutionFinale)/2);
    CarteRaccourcie=carte(xA+1:xA+ResolutionFinale,xA+1:xA+ResolutionFinale);
end

function CarteElargie=ElargirCarte(carte, ResolutionFinale)
    CarteElargie=nan(ResolutionFinale,ResolutionFinale);
    sz=size(carte);
    sz=sz(1);
    xA=floor((ResolutionFinale-sz)/2);
    CarteElargie(xA+1:xA+sz,xA+1:xA+sz)=carte;
end


end