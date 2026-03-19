%%Fonction SCRIPT

%Fonction retournant les labels des polynomes de Zernike (categorical array) e
%n fonction de l'ordre maximum spécifié 

%Entrées :  ordre maximum 
%Sorties :  categorical array des coefficients de Zernike 

function Coef=ZernikeLabel(order)
    Coef=strings(order+1 , 1);
    for j=1:order+1
        Coef(j)=strcat('Z',num2str(j)) ; 
    end 
end 

