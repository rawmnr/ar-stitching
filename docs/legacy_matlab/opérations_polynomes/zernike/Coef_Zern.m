%% Fonction SCRIPT

%Fonction retournant l'ordre et les indices n et m pour un ordre maximum
%spécifié 

%Entrées :  ordre maximum 
%Sorties :  array des indices, des n et des m 

function [indice, n , m]=coef_Zern(Max)
    indice=(1:Max);
    n=[0];
    m=[0];
    N=1;
    ni=1;
    while N<Max 
        for i=-ni:2:ni
            n(end+1)=ni;
            m(end+1)=i;
            N=N+1;
            if N==Max
                break
            end
        end
        ni=ni+1;
    end
end

