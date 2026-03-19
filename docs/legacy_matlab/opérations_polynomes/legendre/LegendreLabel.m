%%Fonction SCRIPT

%Fonction retournant les labels des polynomes de legendre pour un
%coefficient maximal N spécifié (différent de l'ordre du polynomes) )

%Entrées :  coefficient N max 
%Sorties :  Cell array des coefficients  

function Coef=LegendreLabel(N)

    Coef={};
    a=0;
    for j=0:N
        for k=0:j
            a=a+1;
            term=strcat('L ' , num2str(j-k) , '-' , num2str(k));
            Coef{end+1}=term;
        end 
    end 

end 