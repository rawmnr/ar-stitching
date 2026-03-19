function [T,Xind,Yind] = Legendre2D(number_LP,X,Y)
%Legendre_2D returns the 2D Legendre polynomials # = number_LP
%   The X and Y matrices cannot have different lenght

%Legendre ordering definition up to the number 36 - ESO ELT M2 - AIn 30112018
%X_n = [0 1 0 2 1 0 3 2 1 0 4 3 2 1 0 5 4 3 2 1 0 6 5 4 3 2 1 0 7 6 5 4 3 2 1 0];
%Y_n = [0 0 1 0 1 2 0 1 2 3 0 1 2 3 4 0 1 2 3 4 5 0 1 2 3 4 5 6 0 1 2 3 4 5 6 7];

counter=1;Xind=zeros(1,5151);Yind=zeros(1,5151);
for i=0:100
    for j=0:i
        Xind(counter)=i-j;
        Yind(counter)=j;
        counter=counter+1;
    end
end
%%
T=zeros(length(number_LP),length(X(:)));
%2Dfy it
for i=1:length(number_LP)
    Mat = Legendre_polyXY(Xind(number_LP(i)),X,1).* Legendre_polyXY(Yind(number_LP(i)),Y,2);
    T(i,:)=Mat(:);
end

%reshape indices
Xind(length(number_LP)+1:end)=[];
Yind(length(number_LP)+1:end)=[];
end
% 
% function [LnXY] = Legendre_polyXY(n,X,axis)
% %Legendre_polyXY returns the 2D Legendre polynomial of order n along the X
% %axis or the Y axis. This means we can define trough this function all the
% %Legendre polynomials like L00, L01, L02, L03, L04,... , L10, L20, L30,...
% %   INPUTS:
% %               n    =  Legendre polynomial order
% %               X    =  2D matrix obtained with meshgrid for the X or Y
% %               axis
% %               axis = 1 if you are computing the Legendre of L00, L10,
% %               L20,... type
% %   OUTPUTS: 
% %               LnXY = 2D matrix representation of the Legendre polynomial
% 
% if axis == 1    % -> it's the X axis we are computing
% 
%     %Choosing the axis
%     x = X(1,:);
%     x = x(:)';
%     
%     %1D Legendre
%     
%     %Factorial coefficient f_p
%     f_p = 0:n;
%     f_p = (factorial(f_p).^(-1)).*(factorial(n-f_p).^(-1)) * factorial(n);
%     f_p = f_p.^2;
% 
%     %Polynomial coefficient
%     y = 0:n;
%     y = ((x+1)./(x-1)).^(y');
% 
%     %Legendre
%     Ln = f_p * y;
%     Ln = (x-1).^n / 2^(n) .* Ln;
%     
%     %Creating the matrix
%     LnXY = [X(:,1)/X(1,1)] * Ln;
%     
%     
% else            % -> it's the X axis we are computing
%     
%     %Choosing the axis
%     x = X(:,1);
%     x = x(:)';
%     
%     %1D Legendre    
%     
%     %Factorial coefficient f_p
%     f_p = 0:n;
%     f_p = (factorial(f_p).^(-1)).*(factorial(n-f_p).^(-1)) * factorial(n);
%     f_p = f_p.^2;
% 
%     %Polynomial coefficient
%     y = 0:n;
%     y = ((x+1)./(x-1)).^(y');
% 
%     %Legendre
%     Ln = f_p * y;
%     Ln = (x-1).^n / 2^(n) .* Ln;
% 
%     %Creating the matrix
%     LnXY = [X(1,:)/X(1,1)]' * Ln ;
%     LnXY = LnXY';
%     
% end
% 
% end

