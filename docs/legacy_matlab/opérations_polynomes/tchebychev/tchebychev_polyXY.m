%%Fonction SCRIPT

function [TnXY] = tchebychev_polyXY(n,X,axis)
%tchebychev_polyXY returns the 2D Tchebychev polynomial of size length(n)*length(theta). This means we can define trough this function all the
%Tchebychev polynomials like T00, T01, T02, T03, T04,... , T10, T20, T30,...
%   INPUTS:
%               n    =  Tchebychev polynomial order
%               X    =  meshgrid of the vector x
%               Y    =  meshgrid of the vector y 
%   OUTPUTS: 
%               t = 2D matrix representation of the Tchebychev polynomials


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Tchebychev Polynomials
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if axis == 1    % -> it's the X axis we are computing

    %Choosing the axis
    x = X(1,:);
    x = x(:)';
    
    %1D TChebychev
    Tn = cos(n*acos(x));

    %Creating the matrix
    TnXY = [X(:,1)/X(1,1)] * Tn;
    
    
else            % -> it's the X axis we are computing
    
    %Choosing the axis
    x = X(:,1);
    x = x(:)';

    %1D TChebychev
    Tn = cos(n*acos(x));
    
    %Creating the matrix
    TnXY = [X(1,:)/X(1,1)]' * Tn ;
    TnXY = TnXY';
    
end

end

