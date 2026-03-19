    function [carteRecadree]=reinterpADO(TransX,TransY,carte)
        [sz,~]=size(carte);
        [Xq,Yq]=meshgrid(1:sz,1:sz);
        X=Xq+TransX;
        Y=Yq+TransY;
        carteRecadree=interp2(X,Y,carte,Xq,Yq,'cubic',nan);
    end