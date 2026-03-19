function overlapRatio = overlap(d,R)
% Gives the overlap ratio between 2 pupils of the same diameter R
% Centers separated by a distance d 
if d~=0    
    overlapRatio = (2/pi)*acos(d^2/(2*d*R)) ;
else 
    overlapRatio = (2/pi)*acos(0) ;
end    
end

