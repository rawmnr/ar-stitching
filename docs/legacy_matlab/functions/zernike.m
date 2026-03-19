function Znm=zernike(n,m,fact);
%
% Znm=zernike(n,m,fact)
% Cette fonction calcule les coefficients du polynome radial de 
% Zernike d'ordre 'n' et de numéro d'onde angulaire 'm'
% Le paramètre fact peut être 'pv' ou 'rms' s'il est nécessaire que le
% polynome soit normalisé en valeur unité de pv ou rms. Par défaut le
% paramètre vaut 'pv'.
% 
% Entrées:
% n: polynome radial de Zernike d'ordre 'n'
% m: numéro d'onde angulaire 'm'
% fact: 'pv' ou 'pic' (argument facultatif)
% Sortie:
% Znm: coefficient devant les polynômes 
% ex: pour l'AS3 (6*rho^4 ? 6*rho^2 + 1), 
% on a [n,m]=[4,0] et Znm = [6 0 -6 0 1]

if (nargin<2)
   error('Pas assez d''arguments')
elseif (nargin>3)
   error('Trop d''arguments')
else
   if (nargin==2)
      fact='pic';
   else
      if ~strcmp(fact,'pv') & ~strcmp(fact,'rms')
         error('La normalisation doit être en pv ou rms')
      end
   end      
end
m=abs(m);
if rem((n+m),2)~=0
   error(sprintf('   n=%g n''est pas compatible avec m=%g, m+n doit être pair',n,m))
end
if m>n
   error(sprintf('   n=%g n''est pas compatible avec m=%g, n<m',n,m))
end
Znm=zeros(1,n+1);
for s=0:((n-m)/2)
   Znm(2*s+1)=(-1)^s*prod(1:(n-s))/prod(1:s)/prod(1:((n+m)/2-s))/prod(1:((n-m)/2-s));
end
if strcmp(fact,'rms')
   if m==0
      Znm=sqrt(n+1)*Znm;
   else
      Znm=sqrt(2*(n+1))*Znm;
   end
end