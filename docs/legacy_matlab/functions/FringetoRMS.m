function [S_vec] = FringetoRMS(FocusCalFit)
FromnmRMS2Zi = [ sqrt(3) sqrt(6) sqrt(6) sqrt(8) sqrt(8) sqrt(5) sqrt(8) sqrt(8) sqrt(10) sqrt(10) sqrt(12) sqrt(12) sqrt(7)...
                sqrt(10) sqrt(10) sqrt(12) sqrt(12) sqrt(14) sqrt(14) sqrt(16) sqrt(16) sqrt(9) sqrt(12) sqrt(12) sqrt(14) sqrt(14)...
                sqrt(16) sqrt(16) sqrt(18) sqrt(18) sqrt(20) sqrt(20) sqrt(11) sqrt(13) 1];
    
    FromZi2nmRMS = FromnmRMS2Zi.^(-1);
    S_vec = FocusCalFit.Zcoeff';
    S_vec = S_vec(4:length(S_vec));     %Removing Piston, TiltX and TiltY
    S_vec = [S_vec FocusCalFit.MFHF];    %Adding MF/HF
    S_vec = S_vec.*FromZi2nmRMS;
end

