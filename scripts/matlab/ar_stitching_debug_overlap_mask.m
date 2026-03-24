function ar_stitching_debug_overlap_mask(input_path, output_path)
%AR_STITCHING_DEBUG_OVERLAP_MASK Export legacy NEOSS overlap mask.

    arguments
        input_path (1,:) char
        output_path (1,:) char
    end

    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(genpath(fullfile(repo_root, 'docs', 'legacy_matlab')));

    payload = load(input_path);
    TableData = payload.TableData;
    cfg = payload.cfg;

    NEOSS_Param = struct();
    NEOSS_Param.RpupilleCS = double(cfg.RpupilleCS);
    NEOSS_Param.resolutionCS = double(cfg.resolutionCS);
    NEOSS_Param.resolutionTP = double(cfg.resolutionTP);
    NEOSS_Param.nb_cartes = double(cfg.nb_cartes);
    NEOSS_Param.mode_CS = char(cfg.mode_CS);
    NEOSS_Param.Coord1 = double(cfg.Coord1(:).');
    NEOSS_Param.Coord2 = double(cfg.Coord2(:).');

    overlap_count = zeros(NEOSS_Param.resolutionCS, NEOSS_Param.resolutionCS);
    tile_masks = cell(NEOSS_Param.nb_cartes, 1);
    for ii = 1:NEOSS_Param.nb_cartes
        carteSSPP = reshape(TableData(ii, :), NEOSS_Param.resolutionTP, NEOSS_Param.resolutionTP);
        [X, Y] = CalculXY(NEOSS_Param, ii);
        A = ~isnan(reinterpSspp(X, Y, carteSSPP, NEOSS_Param));
        overlap_count = overlap_count + double(A);
        tile_masks{ii} = A;
    end
    overlap_mask = overlap_count > 1;

    [ii, jj] = meshgrid(1:NEOSS_Param.resolutionCS, 1:NEOSS_Param.resolutionCS);
    if strcmp(NEOSS_Param.mode_CS, 'Z')
        overlap_mask((2 * (ii - 0.5 * NEOSS_Param.resolutionCS) / NEOSS_Param.resolutionCS).^2 + ...
            (2 * (jj - 0.5 * NEOSS_Param.resolutionCS) / NEOSS_Param.resolutionCS).^2 > 1) = 0;
    end

    save(output_path, 'overlap_mask', 'overlap_count', 'tile_masks', '-v7');
end

function [X, Y] = CalculXY(NEOSS_Param, ii)
    X = NEOSS_Param.Coord1(ii) * NEOSS_Param.resolutionCS / (2 * NEOSS_Param.RpupilleCS);
    Y = NEOSS_Param.Coord2(ii) * NEOSS_Param.resolutionCS / (2 * NEOSS_Param.RpupilleCS);
end

function [carteRecadree] = reinterpSspp(TransX, TransY, SSPP, Parameters)
    carteRecadree = nan(Parameters.resolutionCS, Parameters.resolutionCS);
    Xa = Parameters.resolutionCS / 2 - Parameters.resolutionTP / 2 + 1;
    Xb = Parameters.resolutionTP / 2 + Parameters.resolutionCS / 2;
    carteRecadree(Xa:Xb, Xa:Xb) = SSPP;
    carteRecadree = reinterpADO(TransX, TransY, carteRecadree);
end

function [carteRecadree] = reinterpADO(TransX, TransY, carte)
    [sz, ~] = size(carte);
    [Xq, Yq] = meshgrid(1:sz, 1:sz);
    X = Xq + TransX;
    Y = Yq + TransY;
    carteRecadree = interp2(X, Y, carte, Xq, Yq, 'cubic', nan);
end
