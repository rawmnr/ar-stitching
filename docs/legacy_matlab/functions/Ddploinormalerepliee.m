function z = Ddploinormalerepliee(x)
%Calcule la densité de probabilité pour la loi normale repliée
%Slt valable pour des valeurs positives de x
global Mean STD
z=1/STD/(2*pi)^0.5*(exp(-(-x-Mean).^2/2/STD^2)+exp(-(x-Mean).^2/2/STD^2));

end

