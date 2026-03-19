%%Fonction APP


function AfficherSousPupilles(app)

if isempty(app.NEOSSTable.Data)
    uialert(app.Warpp40UIFigure,'Erreur : pas de données a stitcher',"Erreur","Icon","warning");  
end
if any(app.LampMaps.Color==[1 0 0]) || any(app.LampCoords.Color==[1 0 0]) || any(app.LampParametre.Color==[1 0 0])
    uialert(app.Warpp40UIFigure,'Erreur : Données invalides',"Erreur","Icon","warning");  
end
% On récupère les paramètres
[NEOSS_Param,TableData]=lectureParametresNEOSS(app.NeossParamFile,app.NEOSSTable.Data(:,1),app.NEOSSTable.Data(:,2:3));
figure()
set(gcf,'color','w')        
hold on
box on
xlabel('X [mm]')
ylabel('Y [mm]')
R=NEOSS_Param.RpupilleTP;
switch NEOSS_Param.SystemeCoordonnees
    case 'IDOINE'
        xsspp=[-R R R -R -R];ysspp=[R R -R -R R];
    case 'polaire'
        tet=linspace(0,2*pi,100);
        [xsspp,ysspp]=pol2cart(tet,R);
end
for C1=1:NEOSS_Param.nb_cartes
    plot(NEOSS_Param.Coord1(C1)+xsspp,NEOSS_Param.Coord2(C1)+ysspp,'k-')
end




end