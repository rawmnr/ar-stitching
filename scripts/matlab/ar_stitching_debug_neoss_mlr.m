function ar_stitching_debug_neoss_mlr(input_path, output_path, debug_obs_idx)
%AR_STITCHING_DEBUG_NEOSS_MLR Export legacy NEOSS MLR intermediates.
%
% The Python side exports:
% - TableData: Nobs x Npix matrix of local SSPP tiles
% - cfg: structure holding the legacy NEOSS parameters
%
% This bridge mirrors the legacy MLR block assembly and saves per-observation
% intermediates for Python/MATLAB comparison.

    arguments
        input_path (1,:) char
        output_path (1,:) char
        debug_obs_idx (1,1) double = 0
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

    debug_data = neoss_mlr_debug(TableData, carte_random, NEOSS_Param, debug_obs_idx);
    debug_data.carte_random = carte_random;

    save(output_path, '-struct', 'debug_data', '-v7');
    fprintf('Debug data saved to %s\n', output_path);
end

function debug_data = neoss_mlr_debug(TableData, carte_random, NEOSS_Param, debug_obs_idx)
    nb_term_alignement = length(NEOSS_Param.indice_alignement);
    nb_term_HA = length(NEOSS_Param.indice_CS) + length(NEOSS_Param.indice_TP);
    nb_elmt_y_i = nb_term_HA + nb_term_alignement;
    M = sparse(zeros(NEOSS_Param.nb_cartes * nb_elmt_y_i, nb_term_HA + nb_term_alignement * NEOSS_Param.nb_cartes));
    y = zeros(NEOSS_Param.nb_cartes * nb_elmt_y_i, 1);

    [coord1_TP_save, coord2_TP_save] = grille(carte_random, 1, NEOSS_Param.mode_TP, [0 0 0 0], NEOSS_Param);
    masqueRecouvrement = calculMasquerecouvrement(TableData, NEOSS_Param);

    debug_data = struct();
    debug_data.nb_obs = NEOSS_Param.nb_cartes;
    debug_data.resolutionTP = NEOSS_Param.resolutionTP;
    debug_data.resolutionCS = NEOSS_Param.resolutionCS;
    debug_data.nb_elmt_y_i = nb_elmt_y_i;
    debug_data.nb_term_HA = nb_term_HA;
    debug_data.nb_term_alignement = nb_term_alignement;
    debug_data.align_terms = NEOSS_Param.indice_alignement;
    debug_data.tp_terms = NEOSS_Param.indice_TP;
    debug_data.cs_terms = NEOSS_Param.indice_CS;
    debug_data.obs_index = debug_obs_idx;
    debug_data.overlap_support = masqueRecouvrement;

    for numero_sspp = 1:NEOSS_Param.nb_cartes
        obs_name = sprintf('obs_%02d', numero_sspp);
        carte = reshape(TableData(numero_sspp, :), NEOSS_Param.resolutionTP, NEOSS_Param.resolutionTP);
        carte = carte - carte_random;
        param = calculParametreGrille(NEOSS_Param, numero_sspp);
        [coord1_CS, coord2_CS] = grille(carte, NEOSS_Param.resolutionTP / NEOSS_Param.resolutionCS, NEOSS_Param.mode_CS, param, NEOSS_Param);
        coord1_TP = coord1_TP_save;
        coord2_TP = coord2_TP_save;
        [donnees_non_masquees, carte_masked] = masquageIntersectRecouvrement(carte, numero_sspp, masqueRecouvrement, NEOSS_Param);
        [carte_fit, ensCoord] = masquageDonnees(donnees_non_masquees, carte_masked, coord1_TP, coord2_TP, coord1_CS, coord2_CS);

        T = remplissageMatriceFit(NEOSS_Param, nb_elmt_y_i, ensCoord);
        [U, S, ~] = svd(T, 0);
        U = canonicalize_svd_columns(U);
        inv_U = pinv(U);
        y_i = inv_U * carte_fit(:);
        M_i = inv_U * T;
        indices = calculIndices(numero_sspp, nb_elmt_y_i, nb_term_HA, nb_term_alignement);

        y(indices.Y) = y_i;
        M(indices.M1a, indices.M1b) = M_i(:, end - nb_term_HA + 1:end);
        M(indices.M2a, indices.M2b) = M_i(:, 1:nb_term_alignement);

        obs_struct = struct();
        obs_struct.obs_idx = numero_sspp;
        obs_struct.translation_xy = [NEOSS_Param.Coord1(numero_sspp), NEOSS_Param.Coord2(numero_sspp)];
        obs_struct.param = param;
        obs_struct.carte_raw = reshape(TableData(numero_sspp, :), NEOSS_Param.resolutionTP, NEOSS_Param.resolutionTP);
        obs_struct.carte_masked = carte_masked;
        obs_struct.carte_masked_full = carte;
        obs_struct.carte_fit = carte_fit;
        obs_struct.coord1_TP_full = coord1_TP;
        obs_struct.coord2_TP_full = coord2_TP;
        obs_struct.coord1_CS_full = coord1_CS;
        obs_struct.coord2_CS_full = coord2_CS;
        obs_struct.coord1_TP = ensCoord.coord1_TP;
        obs_struct.coord2_TP = ensCoord.coord2_TP;
        obs_struct.coord1_CS = ensCoord.coord1_CS;
        obs_struct.coord2_CS = ensCoord.coord2_CS;
        obs_struct.T = T;
        obs_struct.U = U;
        obs_struct.S = S;
        obs_struct.y_i = y_i;
        obs_struct.M_i = M_i;
        obs_struct.indices = indices;
        obs_struct.mask = ~isnan(carte_masked);
        obs_struct.fit_mask = ~isnan(carte_fit);
        obs_struct.fit_indices = donnees_non_masquees;
        obs_struct.keep = (debug_obs_idx == 0) || (numero_sspp == debug_obs_idx);
        debug_data.(obs_name) = obs_struct;
    end

    debug_data.M = M;
    debug_data.y = y;
    debug_data.carte_random = carte_random;

    x = M \ y;
    debug_data.x = x;

    BFResidu = zeros(max(NEOSS_Param.indice_TP), 1);
    BFResidu(NEOSS_Param.indice_TP) = x(1:length(NEOSS_Param.indice_TP));
    debug_data.BFResidu = BFResidu;
    debug_data.align_coeffs = x(length(NEOSS_Param.indice_CS) + length(NEOSS_Param.indice_TP) + 1:end);

    if NEOSS_Param.mode_TP == 'L'
        carte_Instrument = carte_random + genererCarte(coord1_TP_save, coord2_TP_save, BFResidu, NEOSS_Param.lambda, NEOSS_Param.mode_TP);
    elseif NEOSS_Param.mode_TP == 'Z'
        carte_Instrument = troncatureCercle(carte_random + genererCarte(coord1_TP_save, coord2_TP_save, BFResidu, NEOSS_Param.lambda, NEOSS_Param.mode_TP));
    else
        carte_Instrument = carte_random + genererCarte(coord1_TP_save, coord2_TP_save, BFResidu, NEOSS_Param.lambda, NEOSS_Param.mode_TP);
    end
    debug_data.carte_Instrument = carte_Instrument;

    [map, Mismatch, RMS_Diff_Matrix, sppAdjMatrix, sspStack] = stitchingSspp(x, NEOSS_Param, TableData, carte_Instrument);
    debug_data.map = map;
    debug_data.Mismatch = Mismatch;
    debug_data.RMS_Diff_Matrix = RMS_Diff_Matrix;
    debug_data.sppAdjMatrix = sppAdjMatrix;
    debug_data.sspStack = sspStack;
end

function cartePonderation = build_carte_ponderation(NEOSS_Param)
    resTP = double(NEOSS_Param.resolutionTP);
    sigma = double(NEOSS_Param.sigma);
    [k, l] = meshgrid(1:resTP, 1:resTP);
    k = k - resTP / 2;
    l = l - resTP / 2;
    cartePonderation = max(0.1, min(1, exp(((resTP / 2 - sigma)^2 - (k.^2 + l.^2)) / (2 * sigma^2))));
    if strcmpi(NEOSS_Param.SystemeCoordonnees, 'Z')
        cartePonderation = troncatureCercle(cartePonderation);
    end
end

function [coord1, coord2] = grille(carte, rho, mode, param, param_NEOSS)
    trans_rad = param(3);
    angle_rot = param(4);
    [n, m] = size(carte);
    [X, Y] = meshgrid(1:m, 1:n);
    X = (X - mean(X(:))) / floor(m / 2);
    Y = (Y - mean(Y(:))) / floor(m / 2);
    if strcmp(param_NEOSS.SystemeCoordonnees, 'polaire')
        Y = flipud(Y);
    end
    X = rho * X + param(1);
    Y = rho * Y + param(2);
    if strcmp(mode, 'L')
        coord1 = X;
        coord2 = Y;
    else
        if trans_rad ~= 0
            X = X + trans_rad;
        end
        [theta, r] = cart2pol(X, Y);
        theta = mod(theta, 2 * pi);
        if angle_rot ~= 0
            theta = mod(theta + angle_rot * pi / 180, 2 * pi);
        end
        coord1 = theta;
        coord2 = r;
    end
end

function masqueRecouvrement = calculMasquerecouvrement(TableData, NEOSS_Param)
    masqueRecouvrement = zeros(NEOSS_Param.resolutionCS, NEOSS_Param.resolutionCS);
    for ii = 1:NEOSS_Param.nb_cartes
        carteSSPP = reshape(TableData(ii, :), NEOSS_Param.resolutionTP, NEOSS_Param.resolutionTP);
        [X, Y] = CalculXY(NEOSS_Param, ii);
        A = ~isnan(reinterpSspp(X, Y, carteSSPP, NEOSS_Param));
        masqueRecouvrement = masqueRecouvrement + A;
    end
    masqueRecouvrement = masqueRecouvrement > 1;
    [ii, jj] = meshgrid(1:NEOSS_Param.resolutionCS, 1:NEOSS_Param.resolutionCS);
    if strcmp(NEOSS_Param.mode_CS, 'Z')
        masqueRecouvrement((2 * (ii - 0.5 * NEOSS_Param.resolutionCS) / NEOSS_Param.resolutionCS).^2 + ...
            (2 * (jj - 0.5 * NEOSS_Param.resolutionCS) / NEOSS_Param.resolutionCS).^2 > 1) = 0;
    end
end

function [X, Y] = CalculXY(NEOSS_Param, ii)
    X = NEOSS_Param.Coord1(ii) * NEOSS_Param.resolutionCS / (2 * NEOSS_Param.RpupilleCS);
    Y = NEOSS_Param.Coord2(ii) * NEOSS_Param.resolutionCS / (2 * NEOSS_Param.RpupilleCS);
end

function param = calculParametreGrille(ParametresNEOSS, ii)
    if strcmp(ParametresNEOSS.SystemeCoordonnees, 'polaire')
        param = [0, 0, -2 * ParametresNEOSS.Coord1(ii) / ParametresNEOSS.resolutionCS, -ParametresNEOSS.Coord2(ii)];
    elseif strcmp(ParametresNEOSS.SystemeCoordonnees, 'IDOINE')
        [X, Y] = CalculXY(ParametresNEOSS, ii);
        param = [2 * X / ParametresNEOSS.resolutionCS, 2 * Y / ParametresNEOSS.resolutionCS, 0, 0];
    elseif strcmp(ParametresNEOSS.SystemeCoordonnees, 'IRIDE')
        [X, Y] = CalculXY(ParametresNEOSS, ii);
        param = [2 * X / ParametresNEOSS.resolutionCS, 2 * Y / ParametresNEOSS.resolutionCS, 0, 0];
    else
        [X, Y] = CalculXY(ParametresNEOSS, ii);
        param = [2 * X / ParametresNEOSS.resolutionCS, 2 * Y / ParametresNEOSS.resolutionCS, 0, 0];
    end
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

function [donnees_non_masquees, carteSSPP] = masquageIntersectRecouvrement(carteSSPP, numeroSSPP, CarteRecouvrement, Parameters)
    if strcmp(Parameters.SystemeCoordonnees, 'IDOINE') || strcmp(Parameters.SystemeCoordonnees, 'IRIDE')
        [X, Y] = CalculXY(Parameters, numeroSSPP);
        masqueIntersection = not(isnan(reinterpSspp(X, Y, carteSSPP, Parameters))) & CarteRecouvrement ~= 0;
        masqueIntersection = double(masqueIntersection);
        carteRecadree = reinterpADO(-X, -Y, masqueIntersection);
        masqueIntersection = resizeCarte(carteRecadree, Parameters.resolutionTP);
        carteSSPP(masqueIntersection == 0) = nan;
    elseif strcmp(Parameters.SystemeCoordonnees, 'polaire')
        [X, Y] = CalculXY(Parameters, numeroSSPP);
        masqueIntersection = not(isnan(reinterpSspp(X, Y, carteSSPP, Parameters))) & CarteRecouvrement ~= 0;
        masqueIntersection = double(masqueIntersection);
        carteRecadree = reinterpADO(-X, -Y, masqueIntersection);
        masqueIntersection = resizeCarte(carteRecadree, Parameters.resolutionTP);
        carteSSPP(masqueIntersection == 0) = nan;
    end
    donnees_non_masquees = find(~isnan(carteSSPP));
end

function [u, v] = masquageDonnees(booleens, carte1, carte2, carte3, carte4, carte5)
    u = carte1(booleens);
    v.coord1_TP = carte2(booleens);
    v.coord2_TP = carte3(booleens);
    v.coord1_CS = carte4(booleens);
    v.coord2_CS = carte5(booleens);
end

function T = remplissageMatriceFit(Param, length, Coordinates)
    [sz1, sz2] = size(Coordinates.coord1_TP);
    T = zeros(sz1 * sz2, length);
    k = 0;
    for i = Param.indice_alignement
        if Param.mode_TP == 'L'
            if i < 3
                k = k + 1;
                A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, i, Param.lambda, Param.mode_TP);
                T(:, k) = A(:);
            elseif i == 3
                k = k + 1;
                A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, 3, Param.lambda, Param.mode_TP) + base(Coordinates.coord1_TP, Coordinates.coord2_TP, 5, Param.lambda, Param.mode_TP);
                T(:, k) = A(:);
            elseif i == 4
                k = k + 1;
                A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, 4, Param.lambda, Param.mode_TP);
                T(:, k) = A(:);
            elseif i == 5
                k = k + 1;
                A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, 3, Param.lambda, Param.mode_TP) - base(Coordinates.coord1_TP, Coordinates.coord2_TP, 5, Param.lambda, Param.mode_TP);
                T(:, k) = A(:);
            else
                k = k + 1;
                A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, i, Param.lambda, Param.mode_TP);
                T(:, k) = A(:);
            end
        else
            k = k + 1;
            A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, i, Param.lambda, Param.mode_TP);
            T(:, k) = A(:);
        end
    end
    for i = Param.indice_TP
        if (Param.mode_TP == 'L' && i == 5)
            k = k + 1;
            A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, 3, Param.lambda, Param.mode_TP) - base(Coordinates.coord1_TP, Coordinates.coord2_TP, 5, Param.lambda, Param.mode_TP);
            T(:, k) = A(:);
        else
            k = k + 1;
            A = base(Coordinates.coord1_TP, Coordinates.coord2_TP, i, Param.lambda, Param.mode_TP);
            T(:, k) = A(:);
        end
    end
    for i = Param.indice_CS
        k = k + 1;
        A = base(Coordinates.coord1_CS, Coordinates.coord2_CS, i, Param.lambda, Param.mode_CS);
        T(:, k) = A(:);
    end
end

function [indices] = calculIndices(numero_sspp, nb_elmt_y_i, nb_term_HA, nb_alignement_total)
    indices.Y = (numero_sspp - 1) * nb_elmt_y_i + 1 : numero_sspp * nb_elmt_y_i;
    indices.M1a = (numero_sspp - 1) * nb_elmt_y_i + 1 : numero_sspp * nb_elmt_y_i;
    indices.M1b = 1:nb_term_HA;
    indices.M2a = (numero_sspp - 1) * nb_elmt_y_i + 1 : numero_sspp * nb_elmt_y_i;
    indices.M2b = nb_term_HA + 1 + nb_alignement_total * (numero_sspp - 1) : nb_term_HA + 1 + nb_alignement_total - 1 + nb_alignement_total * (numero_sspp - 1);
end

function [carte] = genererCarte(coord1, coord2, coefficients, lambda, mode)
    carte = zeros(size(coord1));
    for i = 1:length(coefficients)
        carte = carte + coefficients(i) * base(coord1, coord2, i - 1, lambda, mode);
    end
end

function [carteCirculaire] = troncatureCercle(carte)
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

function resizedCarte = resizeCarte(carte, resolutionFinale)
    [sz, ~] = size(carte);
    if sz > resolutionFinale
        resizedCarte = raccourcirCarte(carte, resolutionFinale);
    elseif sz < resolutionFinale
        resizedCarte = elargirCarte(carte, resolutionFinale);
    else
        resizedCarte = carte;
    end
end

function CarteRaccourcie = raccourcirCarte(carte, ResolutionFinale)
    CarteRaccourcie = nan(ResolutionFinale, ResolutionFinale);
    [sz1, ~] = size(carte);
    xA = floor((sz1 - ResolutionFinale) / 2);
    CarteRaccourcie = carte(xA + 1:xA + ResolutionFinale, xA + 1:xA + ResolutionFinale);
end

function CarteElargie = elargirCarte(carte, ResolutionFinale)
    CarteElargie = nan(ResolutionFinale, ResolutionFinale);
    [sz1, ~] = size(carte);
    xA = floor((ResolutionFinale - sz1) / 2);
    CarteElargie(xA + 1:xA + sz1, xA + 1:xA + sz1) = carte;
end

function A = base(coord1, coord2, term, lambda, mode)
    if strcmp(mode, 'Z')
        A = basezernike(coord1, coord2, term, lambda);
    elseif strcmp(mode, 'L')
        A = baselegendre(coord1, coord2, term, lambda);
    end
end

function A = basezernike(theta, r, term, lambda, modo)
    switch nargin
        case 4
            modo = 'pv';
    end
    [zn, zm] = zernikeISO2nm(term + 1);
    zn = zn(term + 1);
    zm = zm(term + 1);
    Znm = zernike(zn, zm, modo);
    if zm >= 0
        A = polyval(Znm, r) .* cos(zm * theta);
    else
        A = polyval(Znm, r) .* sin(-zm * theta);
    end
    A = A * lambda;
end

function A = baselegendre(x, y, term, lambda)
    i = 1;
    a = 0;
    ix = zeros(term + 1, 1);
    iy = zeros(term + 1, 1);
    while i <= term + 1
        for j = 0:a
            ix(i) = a - j;
            iy(i) = j;
            i = i + 1;
            if i > term + 1
                break;
            end
        end
        a = a + 1;
    end
    A = polyval(legendrecoeff(ix(term + 1)), x) .* polyval(legendrecoeff(iy(term + 1)), y);
    A = A * lambda;
end

function Pn = legendrecoeff(n)
    Pn = zeros(1, n + 1);
    for k = 0:floor(n / 2)
        Pn(1 + 2 * k) = (-1)^k * nchoosek(n, k) * nchoosek(2 * n - 2 * k, n);
    end
    Pn = 2^(-n) * Pn;
end

function U = canonicalize_svd_columns(U)
    if isempty(U)
        return;
    end
    for col = 1:size(U, 2)
        column = U(:, col);
        if all(~isfinite(column))
            continue;
        end
        [~, anchor] = max(abs(column));
        if column(anchor) < 0
            U(:, col) = -U(:, col);
        end
    end
end

function Znm = zernike(n, m, fact)
    if nargin < 2
        error('Pas assez d''arguments');
    elseif nargin > 3
        error('Trop d''arguments');
    else
        if nargin == 2
            fact = 'pic';
        else
            if ~strcmp(fact, 'pv') && ~strcmp(fact, 'rms')
                error('La normalisation doit être en pv ou rms');
            end
        end
    end
    m = abs(m);
    if rem((n + m), 2) ~= 0
        error(sprintf('   n=%g n''est pas compatible avec m=%g, m+n doit être pair', n, m));
    end
    if m > n
        error(sprintf('   n=%g n''est pas compatible avec m=%g, n<m', n, m));
    end
    Znm = zeros(1, n + 1);
    for s = 0:((n - m) / 2)
        Znm(2 * s + 1) = (-1)^s * prod(1:(n - s)) / prod(1:s) / prod(1:((n + m) / 2 - s)) / prod(1:((n - m) / 2 - s));
    end
    if strcmp(fact, 'rms')
        if m == 0
            Znm = sqrt(n + 1) * Znm;
        else
            Znm = sqrt(2 * (n + 1)) * Znm;
        end
    end
end

function [n, m] = zernikeISO2nm(nterms)
    i = 0;
    n = zeros(nterms, 1);
    m = zeros(nterms, 1);
    for no = 0:2:nterms
        for in = no / 2:no
            i = i + 1;
            n(i) = in;
            m(i) = no - in;
            if i == nterms
                break;
            end
            if m(i) ~= 0
                i = i + 1;
                n(i) = in;
                m(i) = -(no - in);
                if i == nterms
                    break;
                end
            end
        end
        if i == nterms
            break;
        end
    end
end
