%%Fonction SCRIPT

function [a,RMSNorm,res]=LegFit(carteObj,vec,OrdreAnalyse)
%--------------------------
% Fonction qui donne les coeffs de Zernike d'une carte et la norme RMS pour
% un certain rayon de normalisation et une analyse d'un certain ordre
%--------------------------

[b,Z,res,~,~,RMSNorm]=LegendreDecomposeMap(carteObj,OrdreAnalyse);

%calculer le résidu en ajoutant les zernikes retirés du fit
vecdiff=setdiff(OrdreAnalyse,vec);
res=res+Z(:,vecdiff)*b(vecdiff);

a=b;a(vecdiff)=[];
RMSNorm(vecdiff)=[];

end

