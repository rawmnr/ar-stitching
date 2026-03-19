%%Fonction APP


function GetCarteRandom(app)
    if isempty(app.NEOSSTable.Data)
        uialert(app.Warpp40UIFigure,'Erreur : pas de données a stitcher',"Erreur","Icon","warning");
    end
    if any(app.LampMaps.Color==[1 0 0]) || any(app.LampCoords.Color==[1 0 0]) || any(app.LampParametre.Color==[1 0 0])
        uialert(app.Warpp40UIFigure,'Erreur : Données invalides',"Erreur","Icon","warning");
    end
    %On récupère la carte random
    
    [NEOSS_Param,TableData]=lectureParametresNEOSS(app.NeossParamFile,app.NEOSSTable.Data(:,1),app.NEOSSTable.Data(:,2:3));
    [carte_random]=calculCarteRandomLegacy(TableData,NEOSS_Param); %Version
    CarteOut=GenerateMapObject(carte_random,NEOSS_Param.RpupilleTP,NEOSS_Param.lambda,'Carte random'); 
    New_Map(app, CarteOut, CarteOut.titre);
end