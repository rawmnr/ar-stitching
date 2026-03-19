function ChargerMapsNEOSS(app)
    %
try
    %
    %récupère et affiche les coordonnées des sous-pupilles depuis un fichier txt
    % à la condition d'avoir écrit le fichier correctement.
    app.LampMaps.Color=[1 0 0];
    [files,folder]=uigetfile({'*.opd','Cartes'},'Choisir des cartes a stitcher',app.DefaultFolder,'multiselect','on');
    app.DefaultFolder=[folder,'\']; 
    figure(app.Warpp40UIFigure)
    %
    if isempty(app.NEOSSTable.Data)
        uialert(app.Warpp40UIFigure,'Erreur : Commencer par charger les coordonnées',"Erreur","Icon","warning");
        return
    end
    FileswFold=app.NEOSSTable.Data;
    %
    if height(app.NEOSSTable.Data)~=length(files)
        uialert(app.Warpp40UIFigure,'Erreur : nombre de carte différent du nombre de coordonnées',"Erreur","Icon","warning");        
        return
    end
    for i=1:length(files)
        FileswFold{i,1}=[folder, files{i}];
    end
    app.NEOSSTable.Data=FileswFold;
    app.NEOSSTable.ColumnFormat={'char','numeric','numeric'};
    app.LampMaps.Color=[0.39,0.83,0.07];
catch
    uialert(app.Warpp40UIFigure,'Erreur au chargement des paramètres NEOSS',"Erreur","Icon","warning");        
end
end