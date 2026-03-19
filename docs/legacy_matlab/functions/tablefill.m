function [smalltable] = tablefill(FocusCalFit,ErrorSource)
    FromnmRMS2Zi = [ sqrt(3) sqrt(6) sqrt(6) sqrt(8) sqrt(8) sqrt(5) sqrt(8) sqrt(8) sqrt(10) sqrt(10) sqrt(12) sqrt(12) sqrt(7)...
                sqrt(10) sqrt(10) sqrt(12) sqrt(12) sqrt(14) sqrt(14) sqrt(16) sqrt(16) sqrt(9) sqrt(12) sqrt(12) sqrt(14) sqrt(14)...
                sqrt(16) sqrt(16) sqrt(18) sqrt(18) sqrt(20) sqrt(20) sqrt(11) sqrt(13) 1];
    
    FromZi2nmRMS = FromnmRMS2Zi.^(-1);
    S_vec = FocusCalFit.Zcoeff';
    S_vec = S_vec(4:length(S_vec));     %Removing Piston, TiltX and TiltY
    S_vec = [S_vec FocusCalFit.MFHF];    %Adding MF/HF
    S_vec = S_vec.*FromZi2nmRMS;
    colname = ErrorSource.label;
    smalltable = table;
    if ErrorSource.type == 1
        smalltable.colname = [ErrorSource.variation;ErrorSource.coeff;0;ErrorSource.ContingencyMargin;ErrorSource.variation;0;S_vec'];
    else
        smalltable.colname = [ErrorSource.variation;0;ErrorSource.coeff;ErrorSource.ContingencyMargin;ErrorSource.variation;0;S_vec'];
    end
 
end
