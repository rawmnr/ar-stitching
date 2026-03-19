function [carteObj,Z] = Zernike_Analysis(map,IMAGE,Title)
%Zernike_Analysis computes the Zernike decomposition of the map and gives the contribution over the  
%   Detailed explanation goes here
sizemap = size(map,2);
%Filling the structure
[X Y]= meshgrid(((1:sizemap)-sizemap/2-0.5)*2/sizemap,...
                ((1:sizemap)-sizemap/2-0.5)*2/sizemap);
carteObj.carte = map;
carteObj.grilleX = X;
carteObj.grilleY = Y;

pup=~isnan(carteObj.carte(:)) & sqrt(carteObj.grilleX(:).^2+carteObj.grilleY(:).^2)<1;

%Decomposition
[a,Z,res,ns,ms,RMSNorm,rs]=ZernikeDecomposeMap(carteObj,[1:37],'Fringe');
carteObj.carte_fit = carteObj.carte;
carteObj.carte_fit(pup)=Z(:,:)*a(:);
carteObj.Zcoeff = a;
junk = map - carteObj.carte_fit;
junk = junk(:);
junk = junk(~isnan(junk));
carteObj.MFHF =sqrt(sum(junk.*junk)/length(junk));



%Checking the fit result
if IMAGE == true
    figure
    plotWARPPloc(map,820,750,Title,'','MSE',1)
%     figure
%     plotWARPPloc(carteObj.carte_fit,820,750,'Carte Fit','','MSE',1)
%     figure
%     plotWARPPloc(map - carteObj.carte_fit,820,750,'HF','','MSE',1)
end

end

