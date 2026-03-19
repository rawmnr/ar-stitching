function z = loinormalerepliee(x, Mean, STD)
%Calcule la fonction de répartition pour la loi normale repliée
z=0.5*(erf((x+Mean)/2^0.5/STD)+erf((x-Mean)/2^0.5/STD));
end

