function [T,Xind,Yind] = tchebychev2D(n,X,Y)
%tchebychev2D returns the 2D Tchebychev polynomials # = n
%   The X and Y matrices cannot have different lenght
%   n is the vector containaing order numbers.

%Tchebychev ordering definition up to the number 36 - ESO ELT M2 - AIn 30112018
%X_n = [0 1 0 2 1 0 3 2 1 0 4 3 2 1 0 5 4 3 2 1 0 6 5 4 3 2 1 0 7 6 5 4 3 2 1 0];
%Y_n = [0 0 1 0 1 2 0 1 2 3 0 1 2 3 4 0 1 2 3 4 5 0 1 2 3 4 5 6 0 1 2 3 4 5 6 7];

%% Ordering for the convolution product
counter=1;Xind=zeros(1,5151);Yind=zeros(1,5151);
for i=0:100
    for j=0:i
        Xind(counter)=i-j;
        Yind(counter)=j;
        counter=counter+1;
    end
end
%%
T=zeros(length(n),length(X(:)));
for i=1:length(n)
    Mat = tchebychev_polyXY(Xind(n(i)),X,1).* tchebychev_polyXY(Yind(n(i)),Y,2);
    T(i,:)=Mat(:);
end

%reshape indices
Xind=Xind(n);
Yind=Yind(n);
end