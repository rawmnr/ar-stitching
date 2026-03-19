function [LnXY] = Legendre_polyXY(n,X,axis)
%Legendre_polyXY returns the 2D Legendre polynomial of order n along the X
%axis or the Y axis. This means we can define trough this function all the
%Legendre polynomials like L00, L01, L02, L03, L04,... , L10, L20, L30,...
%   INPUTS:
%               n    =  Legendre polynomial order
%               X    =  2D matrix obtained with meshgrid for the X or Y
%               axis
%               axis = 1 if you are computing the Legendre of L00, L10,
%               L20,... type
%   OUTPUTS: 
%               LnXY = 2D matrix representation of the Legendre polynomial

if axis == 1    % -> it's the X axis we are computing

    %Choosing the axis
    x = X(1,:);
    x = x(:)';
    
    %1D Legendre
    
    %Factorial coefficient f_p
    f_p = 0:n;
    f_p = (factorial(f_p).^(-1)).*(factorial(n-f_p).^(-1)) * factorial(n);
    f_p = f_p.^2;

    %Polynomial coefficient
    y = 0:n;
    y = ((x+1)./(x-1)).^(y');

    %Legendre
    Ln = f_p * y;
    Ln = (x-1).^n / 2^(n) .* Ln;
    
    %Creating the matrix
    LnXY = [X(:,1)/X(1,1)] * Ln;
    
    
else            % -> it's the X axis we are computing
    
    %Choosing the axis
    x = X(:,1);
    x = x(:)';
    
    %1D Legendre    
    
    %Factorial coefficient f_p
    f_p = 0:n;
    f_p = (factorial(f_p).^(-1)).*(factorial(n-f_p).^(-1)) * factorial(n);
    f_p = f_p.^2;

    %Polynomial coefficient
    y = 0:n;
    y = ((x+1)./(x-1)).^(y');

    %Legendre
    Ln = f_p * y;
    Ln = (x-1).^n / 2^(n) .* Ln;

    %Creating the matrix
    LnXY = [X(1,:)/X(1,1)]' * Ln ;
    LnXY = LnXY';
    
end

end

