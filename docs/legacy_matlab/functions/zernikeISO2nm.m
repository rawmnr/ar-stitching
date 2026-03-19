function [n,m]=zernikeISO2nm(nterms);
%
% Cette fonction calcule l'ordre radial 'n' et angulaire 'm' pour les
% 'nterms' premiers termes de la norme ISO.
% On rappelle qu'en norme ISO le premier coefficient est Z_0, obtenu avec
% nterms=1 donne n=[0],m=[0] 
% 
% Entrées:
% nterms: entier pour lequel on veut connaître n,m dans les polynômes de Zernike
% Sortie:
% [n,m]: vecteur qui contient les n et m jusqu'au nterms donné
% ex: pour nterms= 3 (qui correspond à Z2 dans la convention Warpp)
% n = [0;1;1] et m = [0;1;-1]


i=0;
n=zeros(nterms,1);
m=zeros(nterms,1);
for no=0:2:nterms
   for in=no/2:no
      i=i+1;
      n(i)=in;
      m(i)=no-in;
      if i==nterms,break, end
      if m(i)~=0
         i=i+1;
         n(i)=in;
         m(i)=-(no-in);
         if i==nterms, break, end
      end
   end
   if i==nterms,break, end
end
