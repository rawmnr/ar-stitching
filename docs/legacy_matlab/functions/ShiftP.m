 % Recale la SSPP centrée dans la carte CS à sa position finale dans la carte CS
 % INPUT : 
 %     - X0Bis         : centre de la SSPP en X dans le repère de la carte CS 
 %     - Y0Bis         : centre de la SSPP en Y dans le repère de la carte CS
 %     - carte         : carte de la SSPP centrée dans la carte CS
 % OUTPUT : 
 %     - carteRecadree : carte de la SSPP replacée dans la pupille de la CS
 % 
 % 1 : Création d'une carte vide de résolutionCS
 % 2 : Calcul de shift à réaliser en fonction du cadran dans lequel doit se trouver la SSPP après recalage
 % 3 : Placement de la carte SSPP au centre de la carte vide de résolutionCS

 
 
 
 function [carteRecadree]=ShiftP(X0Bis,Y0Bis,carte)
        sz=size(carte);
        sz=sz(1);
        carteRecadree=nan(sz,sz);
        xA=X0Bis;
        yA=Y0Bis;
        if xA<0
            if yA<0
                carteRecadree(1:sz+xA,1:sz+yA)=carte(1-xA:sz,1-yA:sz);
            elseif yA==0
                carteRecadree(1:sz+xA,:)=carte(1-xA:sz,:);
            elseif yA>0
                carteRecadree(1:sz+xA,1+yA:sz)=carte(1-xA:sz,1:sz-yA);
            end
        elseif xA==0
            if yA<0
                carteRecadree(:,1:sz+yA)=carte(:,1-yA:sz);
            elseif yA==0
                carteRecadree(:,:)=carte(:,:);
            elseif yA>0
                carteRecadree(:,1+yA:sz)=carte(:,1:sz-yA);
            end
        elseif xA>0
            if yA<0
                carteRecadree(1+xA:sz,1:sz+yA)=carte(1:sz-xA,1-yA:sz);
            elseif yA==0
                carteRecadree(1+xA:sz,:)=carte(1:sz-xA,:);
            elseif yA>0
                carteRecadree(1+xA:sz,1+yA:sz)=carte(1:sz-xA,1:sz-yA);
            end
        end
         
    end
    