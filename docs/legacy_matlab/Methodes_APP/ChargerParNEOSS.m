function ChargerParNEOSS(app)
    %
    try
        ListeDatNEOSS={"RPupille Totale (""CS"")";...
                "Résolution Totale (""CS"")";...
                "Résolution Sous pupille (""TP"")";...
                "Rpupille Sous pupille (""TP"")";...
                "lambda";...
                "Nombre de SSPP";...
                "sigma Pondération Pixels";...
                "mismatch";...
                "Fichier de position des sous pupilles";...
                "Polynomes décrivant l'instrument 'Z' ou 'L'";...
                "Polynome décrivant le miroir 'Z' ou 'L'";...
                "premier Indice alignement 0 pour PISTON ect.";...
                "dernier Indice alignement 0 pour PISTON ect.";...
                "premier Indice du modèle 'CS'";...
                "dernier Indice du modèle 'CS'";...
                "premier Indice de la carte de biais SSPP 'TP'";...
                "dernier Indice de la carte de biais SSPP 'TP'";...
                "Limite Calcul Carte Moyenne";...
                "Carte Supportage (Booléen)";...
                "Chemin de la carte supportage";...
                "Coordonnées (IDOINE ou Polaire)"};
        %
        app.NeossParamFile='';
        app.LampParametre.Color=[1 0 0];    
        app.LampMaps.Color=[1,0,0];
        app.LampCoords.Color=[1,0,0];
        [files,folder]=uigetfile({'*.txt','Fichier de paramètre NEOSS'},'Choisir le fichier de paramètres',app.DefaultFolder,'multiselect','off');
        app.DefaultFolder=[folder,'\'];
        figure(app.Warpp40UIFigure)
        Dat=readcell([folder files]);
        Dat=Dat(1:21,:);
        Dat(:,2)=Dat(1:21,1); %on se limite a 21 paramètres
        Dat(:,1)=ListeDatNEOSS;    
        app.NeossParamFile=[folder files];
        app.LampParametre.Color=[0.39,0.83,0.07];
    catch
        uialert(app.Warpp40UIFigure,'Erreur au chargement des paramètres NEOSS',"Erreur","Icon","warning");
    end

    try
        %Ensuite on essaye de charger les coordonnées si c'est dans le
        %fichier
        [pos_sspp] = get_position_sspp_NEOSS([folder,Dat{9,2}]);
        for i=1:size(pos_sspp,1)
            FileswFold{i,2}=pos_sspp{i,2};
            FileswFold{i,3}=pos_sspp{i,3};
        end
        app.NEOSSTable.Data=FileswFold;
        app.NEOSSTable.ColumnFormat={'char','numeric','numeric'};
        app.LampCoords.Color=[0.39,0.83,0.07];
    catch
        uialert(app.Warpp40UIFigure,'Erreur au chargement des coordonnées',"Erreur","Icon","warning");
    end
    %Ensuite on essaye de charger les cartes si c'est dans le
    %fichier
    try
        %Ensuite on essaye de charger les coordonnées si c'est dans le
        %fichier
        for indiceSSPP=1:Dat{6,2}
            FileswFold{indiceSSPP,1}=[folder 'C' sprintf('%03i',indiceSSPP) 'AP_TP.Opd'];
        end
        app.NEOSSTable.Data=FileswFold;
        app.LampMaps.Color=[0.39,0.83,0.07];
    catch
        uialert(app.Warpp40UIFigure,'Erreur au chargement des cartes',"Erreur","Icon","warning");
    end

end