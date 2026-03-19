% The function Rugoa computes the rugosity of a map Carte
%       INPUTS: Carte, '.opd' format
%       OUTPUT: Av, average value of the map 
%               Ra, rugosity, sum of absolute values
%               Rq, rigosity, square root of sum of squared values

function [Av,Ra,Rq]=Rugoa(Carte)
carto=Carte;Av=mean(carto.carte(:),'omitnan');
carto.carte=carto.carte-Av;
carto1=abs(carto.carte(~isnan(carto.carte)));
Ra=sum(carto1,'all')/size(carto1,1)/size(carto1,2);
Rq=(sum(carto1.^2,'all')/size(carto1,1)/size(carto1,2))^0.5;
end