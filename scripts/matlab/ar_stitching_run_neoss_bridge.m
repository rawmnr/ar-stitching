function ar_stitching_run_neoss_bridge(input_path, output_path)
%AR_STITCHING_RUN_NEOSS_BRIDGE Run legacy NEOSS on exported inputs.
%
% The Python side exports:
% - TableData: Nobs x Npix matrix of local SSPP tiles
% - cfg: structure holding the legacy NEOSS parameters

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
    NEOSS_Param.RpupilleTP = double(cfg.RpupilleTP);
    NEOSS_Param.lambda = double(cfg.lambda);
    NEOSS_Param.nb_cartes = double(cfg.nb_cartes);
    NEOSS_Param.sigma = double(cfg.sigma);
    NEOSS_Param.mismatch = double(cfg.mismatch);
    NEOSS_Param.mode_TP = char(cfg.mode_TP);
    NEOSS_Param.mode_CS = char(cfg.mode_CS);
    NEOSS_Param.indice_alignement = double(cfg.indice_alignement(:).');
    NEOSS_Param.indice_CS = double(cfg.indice_CS(:).');
    NEOSS_Param.indice_TP = double(cfg.indice_TP(:).');
    NEOSS_Param.limit = double(cfg.limit);
    NEOSS_Param.supportage = double(cfg.supportage);
    NEOSS_Param.pathSupportage = char(cfg.pathSupportage);
    NEOSS_Param.SystemeCoordonnees = char(cfg.SystemeCoordonnees);
    NEOSS_Param.Coord1 = double(cfg.Coord1(:).');
    NEOSS_Param.Coord2 = double(cfg.Coord2(:).');
    NEOSS_Param.cartePonderation = build_carte_ponderation(NEOSS_Param);

    use_random_map = logical(cfg.use_random_map);
    if use_random_map
        carte_random = calculCarteRandomLegacy(TableData, NEOSS_Param);
    else
        carte_random = zeros(NEOSS_Param.resolutionTP, NEOSS_Param.resolutionTP);
    end

    [x, carte_Instrument] = MLR(TableData, carte_random, NEOSS_Param);
    [map, Mismatch, RMS_Diff_Matrix, sppAdjMatrix, sspStack] = stitchingSspp(x, NEOSS_Param, TableData, carte_Instrument);
    save(output_path, 'map', 'Mismatch', 'carte_Instrument', 'RMS_Diff_Matrix', 'sppAdjMatrix', 'sspStack', '-v7');
end

function cartePonderation = build_carte_ponderation(NEOSS_Param)
    resTP = double(NEOSS_Param.resolutionTP);
    sigma = double(NEOSS_Param.sigma);
    [k, l] = meshgrid(1:resTP, 1:resTP);
    k = k - resTP / 2;
    l = l - resTP / 2;
    cartePonderation = max(0.1, min(1, exp(((resTP / 2 - sigma)^2 - (k.^2 + l.^2)) / (2 * sigma^2))));
    if strcmpi(NEOSS_Param.SystemeCoordonnees, 'Z')
        cartePonderation = troncature_cercle(cartePonderation);
    end
end

function carteCirculaire = troncature_cercle(carte)
    sz = size(carte);
    i0 = floor(sz(1) / 2) + 1;
    j0 = floor(sz(2) / 2) + 1;
    carteCirculaire = nan(sz(1), sz(2));
    for i = 1:sz(1)
        for j = 1:sz(2)
            if (i - i0)^2 + (j - j0)^2 <= floor(sz(1) / 2)^2
                carteCirculaire(i, j) = carte(i, j);
            end
        end
    end
end
