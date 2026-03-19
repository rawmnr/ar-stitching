function ChargerCoordsNEOSS(app)
%
try
    %
    %récupère et affiche les coordonnées des sous-pupilles depuis un fichier txt
    % à la condition d'avoir écrit le fichier correctement.
    app.NEOSSTable.Data=[];
    app.LampCoords.Color=[1 0 0];
    [files,folder]=uigetfile({'*.txt','Cartes'},'Choisir le fichier de coordonnées',app.DefaultFolder,'multiselect','off');
    app.DefaultFolder=[folder,'\'];
    figure(app.Warpp40UIFigure)
    [pos_sspp] = get_position_sspp_NEOSS([folder,files]);
    %mapNames= ls([app.path_cartes '*.opd'])
    for i=1:size(pos_sspp,1)
        FileswFold{i,2}=pos_sspp{i,2};
        FileswFold{i,3}=pos_sspp{i,3};
    end
    app.NEOSSTable.Data=FileswFold;
    app.NEOSSTable.ColumnFormat={'char','numeric','numeric'};
    app.LampCoords.Color=[0.39,0.83,0.07];
catch
    uialert(app.Warpp40UIFigure,'Erreur au chargement des paramètres NEOSS',"Erreur","Icon","warning");        
end
end