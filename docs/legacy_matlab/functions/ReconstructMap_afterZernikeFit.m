function [Recon_map] = ReconstructMap_afterZernikeFit(map,rs,dim)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%Reference mask
MASK = rs;
MASK(rs<=1)=1;
MASK(rs>1)=0;

MASK_1d = MASK(:);
numb_pixel = 1;

for it = 1:length(MASK_1d)
    
    if MASK_1d(it)==1
        
        Recon_map(it) =  map(numb_pixel,1);
        numb_pixel = numb_pixel + 1;
        
    else
       
        Recon_map(it) = nan;
        
    end

end

Recon_map = reshape(Recon_map,dim,dim);

end

