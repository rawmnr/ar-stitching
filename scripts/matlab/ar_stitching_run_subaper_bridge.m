function ar_stitching_run_subaper_bridge(input_path, output_path)
%AR_STITCHING_RUN_SUBAPER_BRIDGE Run legacy Subaper on exported inputs.
%
% This helper is invoked from Python with a .mat input file that contains:
% - TableData: NSSPP x NPIX matrix
% - Type: legacy basis mode (char array or string)
% - Compensateurs: alignment term indices (MATLAB 1-based)

    arguments
        input_path (1,:) char
        output_path (1,:) char
    end

    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(genpath(fullfile(repo_root, 'docs', 'legacy_matlab')));

    payload = load(input_path);
    TableData = payload.TableData;
    Type = char(payload.Type);
    Compensateurs = payload.Compensateurs;
    if isvector(Compensateurs)
        Compensateurs = Compensateurs(:).';
    end

    [map, mismatch, Cs, a, RMS_Diff_Matrix, sppAdjMatrix] = Subaper_Core_v2(TableData, Type, Compensateurs);
    save(output_path, 'map', 'mismatch', 'Cs', 'a', 'RMS_Diff_Matrix', 'sppAdjMatrix', '-v7');
end
