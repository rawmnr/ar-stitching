%%Fonction APP

%callback du bouton stitcher. Permet de gérer les appels de fonction pour
%chaque algorithme et de créer les cartes de sortie

%Entrées : app
%Sorties : pas de sorties

function Stitch(app)

        if endsWith(app.DefaultFolder,'opdx')
            [cartes,~,~,~]=readOPDx(app.DefaultFolder);
            for i=1:size(app.FiletableStitching.Data,1)
                for j=1:length(cartes)
                    if isequal(app.FiletableStitching.Data{i,1},cartes{1,j}.titre)
                        Maps{i}=cartes{1,j};
                    end
                end
            end
        else
            %Charge les cartes
            for i=1:size(app.FiletableStitching.Data,1)
                    Maps{i}=readOPD(app.FiletableStitching.Data{i,1});
            end
        end
        %Compensateurs
        switch app.CompensateursStitchingDropDown.Value
            case 'Piston et Tilts'
                Compensateurs=[1:3];
            case 'Piston, Tilts et Focus'
                Compensateurs=[1:4];
        end
        %Type of polynomials
        switch app.PolynomesStitchingDropDown.Value
            case 'Zernike'
                Type='Z';
            case 'Legendre Mod.'
                Type='LM';
            case 'Legendre.'
                Type='L';
        end
        %Stitch !
        wb = uiprogressdlg(app.Warpp40UIFigure,'Title','Merci d''attendre','Message','Stitching en cours');
        [CarteOut,MM,RMS_Diff_Matrix,sppAdjMatrix]=SubaperStitching(Maps , Type, Compensateurs,wb);
        close(wb)
    
        MM.children=MM;
        MM.carte_children  =MM.titre;
        CarteOut.children=CarteOut;
        CarteOut.carte_children  =CarteOut.titre;
        New_Map(app, MM, MM.titre);
        New_Map(app, CarteOut, CarteOut.titre);
        showPopupOverlapDiff(RMS_Diff_Matrix);
        showPopupSPPdiff(sppAdjMatrix);


end

% Fonction pour afficher le plot des différences rms entre les ssp 
function showPopupOverlapDiff(RMS_Diff_Matrix)
    popupFig = figure('Name', 'Difference in overlapped SSP', 'NumberTitle', 'off');
    
    imagesc(RMS_Diff_Matrix);
    colorbar;  

end

function showPopupSPPdiff(sppAdjMatrix)
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