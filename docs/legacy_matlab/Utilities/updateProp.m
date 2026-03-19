%% Fonction SCRIPT

%Cette fonction actualise les infos de la carte (.OPD) dont on possède la
%nouvelle cartogrpahie (matrice)

%Entrées : .OPD auquel on a assigné une nouvelle cartographie 
%Sortie :  .OPD dont les informations ont été mises à jour 

function CarteOut=updateProp(Carte)
    CarteOut=Carte;
    if isfield(CarteOut,'titre')
        if isstring(CarteOut.titre)
            CarteOut.titre = string(CarteOut.titre);
        end
    end
    if isfield(CarteOut,'path')
        CarteOut=rmfield(CarteOut,'path');
    end
    CarteOut.stat.min=min(CarteOut.carte(:));
    CarteOut.stat.max=max(CarteOut.carte(:));
    CarteOut.stat.piston=mean(CarteOut.carte(:) , 'omitnan');
    CarteOut.stat.ptv=max(CarteOut.carte(:))-min(CarteOut.carte(:));
    CarteOut.stat.rms= std(CarteOut.carte(:), 1 , 'omitnan');
    [Av,Ra,Rq]=Rugoa(Carte);
    CarteOut.stat.Ra= Ra;
    CarteOut.stat.Rq= Rq;
    CarteOut.hauteur= length(Carte.carte(1,:));
    CarteOut.largeur= length(Carte.carte(:,1));
    if isfield(CarteOut,'children')
        if iscell(CarteOut.children)
            n=length(CarteOut.children)+1;
            for i=1:length(CarteOut.children)
                if iscell(CarteOut.children{i})
                    carte_replace = CarteOut.children{i}{1};
                    for j=2:length(CarteOut.children{i})
                        CarteOut.children{n}=CarteOut.children{i}{j};
                        n=n+1;
                    end
                    CarteOut.children{i}=carte_replace;
                end
            end
        end
    end
end