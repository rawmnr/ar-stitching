%%Fonction APP

%callback du bouton stitcher pour NEOSS. Permet de gérer les appels de fonction pour
%chaque algorithme et de créer les cartes de sortie

%Entrées : app
%Sorties : pas de sorties

function [CarteOut,MM]=Stitch_in_NEOSS(app)
    if isempty(app.NEOSSTable.Data)
        uialert(app.Warpp40UIFigure,'Erreur : pas de données a stitcher',"Erreur","Icon","warning");
    end
    if any(app.LampMaps.Color==[1 0 0]) || any(app.LampCoords.Color==[1 0 0]) || any(app.LampParametre.Color==[1 0 0])
        uialert(app.Warpp40UIFigure,'Erreur : Données invalides',"Erreur","Icon","warning");
    end
    %On applique l'algorithme de stichting
    try
        wb = uiprogressdlg(app.Warpp40UIFigure,'Title','Merci d''attendre','Message','Stitching en cours');
        [CarteOut,MM,RMS_Diff_Matrix,sppAdjMatrix,sspStack]=NEOSSStitching(app.NEOSSTable.Data(:,1),app.NEOSSTable.Data(:,2:3),app.NeossParamFile,app.CarteRandomCheckBox.Value,wb);
        close(wb)
    catch
        close(wb)
        uialert(app.Warpp40UIFigure,'Erreur indéterminée pendant le stitching',"Erreur","Icon","warning");
        return
    end
    %Crée les cartes OPD dans la pile de cartes 
    MM.children=MM;
    MM.carte_children =MM.titre;
    New_Map(app, MM, MM.titre);
    CarteOut.children=CarteOut;
    CarteOut.carte_children =CarteOut.titre;
    New_Map(app, CarteOut, CarteOut.titre);
    showPopupOverlapDiff(RMS_Diff_Matrix);
    showPopupSPPdiff(sppAdjMatrix);
    % reset everything
    app.LampMaps.Color=[1 0 0];
    app.LampCoords.Color=[1 0 0];
    app.LampParametre.Color=[1 0 0];
    app.NeossParamFile=[];
    app.NEOSSTable.Data=[];
    % Subaper additionnel
    switch app.NEOSSSubaperCheckDD.Value
        case "Sans Subaper"
        case "Z1:3"
            Type="Z";
            Compensateurs=1:3;
        case "Z1:4"
            Type="Z";
            Compensateurs=1:4;
        case "LM1:3"
            Type="LM";
            Compensateurs=1:3;
        case "LM1:4"
            Type="LM";
            Compensateurs=1:4;
    end
    switch app.NEOSSSubaperCheckDD.Value
        case "Sans Subaper"
        otherwise
            app.SubaperResults=parfeval(backgroundPool,@Subaper_Core_v2,4,sspStack, Type, Compensateurs,1,[]);
            app.SubaperResultsTimer=timer('TimerFcn',@(x,y)TimerUpdateSubaperCX(app,CarteOut,MM),'ExecutionMode','fixedRate');
            app.SubaperResultsTimer.start()
    end
end

function TimerUpdateSubaperCX(app,CarteOut,MM)
    if strcmp(app.SubaperResults.State,'running')
        app.LampSubaperCX.Color=[0.93,0.69,0.13];
    elseif strcmp(app.SubaperResults.State,'finished')
        CarteOutSA=CarteOut;
        MMSA=MM;
        [CarteOutSA.carte,MMSA.carte,RMS_Diff_Matrix,sppAdjMatrix]=app.SubaperResults.fetchOutputs();
        CarteOutSA.titre='CX_Subaper_Map';
        MMSA.titre='CX_Subaper_MM';
        MMSA.children=MMSA;
        MMSA.carte_children =MMSA.titre;
        New_Map(app, MMSA, MMSA.titre);
        CarteOutSA.children=CarteOutSA;
        CarteOutSA.carte_children =CarteOutSA.titre;
        New_Map(app, CarteOutSA, CarteOutSA.titre);
        a=showPopupOverlapDiff(RMS_Diff_Matrix);
        a.Name='Différence des SSP superposées (Subaper)';
        b=showPopupSPPdiff(sppAdjMatrix);
        b.Name='SPP corrigées (Subaper)';        
        app.LampSubaperCX.Color=[0.65,0.65,0.65];
        app.SubaperResultsTimer.stop()
        app.SubaperResultsTimer=[];
        app.SubaperResults=[];
    else 
        app.LampSubaperCX.Color=[1.00,0.00,0.00];
    end

end

% function RunSubaperOnNEOSSData(app,sspStack,Type,Compensateurs,CarteOut,MM)
%             app
%             CarteOutSA=CarteOut;
%             MMSA=CarteOut;
%             [CarteOutSA.carte,MMSA.carte,Cs,a,RMS_Diff_Matrix,sppAdjMatrix]=Subaper_Core_v2(sspStack, Type, Compensateurs);
%             CarteOutSA.titre='CX Subaper Map';
%             MMSA.titre='CX Subaper MM';
%             New_Map(app, MMSA, MMSA.titre);
%             New_Map(app, CarteOutSA, CarteOutSA.titre);
%             a=showPopupOverlapDiff(RMS_Diff_Matrix);
%             a.Name='Différence des SSP superposées (Subaper)';
%             b=showPopupSPPdiff(sppAdjMatrix);
%             b.Name='SPP corrigées (Subaper)';
% end

% Fonction pour afficher le plot des différences rms entre les ssp 
function popupFig=showPopupOverlapDiff(RMS_Diff_Matrix)
    popupFig = figure('Name', 'Différence des SSPP superposées', 'NumberTitle', 'off');
    
    imagesc(RMS_Diff_Matrix);
    colorbar;  

end

function popupFig=showPopupSPPdiff(sppAdjMatrix)
    popupFig = figure('Name', 'SPP corrigées', 'NumberTitle', 'off');
    [n, ~, N] = size(sppAdjMatrix);
    
    numRows = floor(sqrt(N));
    numCols = ceil(N / numRows);
    
    % Loop through each slice of the matrix and plot using imagesc
    for k = 1:N
        subplot(numRows, numCols, k);
        imagesc(sppAdjMatrix(:,:,k));
%         colorbar; % Optional: Add a colorbar for each subplot
        axis off
        axis image
    end

    h = colorbar;
    h.Position = [0.93 0.11 0.02 0.815]; % Adjust position [left bottom width height]

    for k = 1:N
        subplot(numRows, numCols, k);
        pos = get(gca, 'Position');
        pos(3) = pos(3) * 0.9; % Reduce width to make space for colorbar
        set(gca, 'Position', pos);
    end
end