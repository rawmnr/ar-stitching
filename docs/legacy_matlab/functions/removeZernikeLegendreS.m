    function [carteOutput]=removeZernikeLegendreS(carteInput, mode, numZer, lambda)
        [sz,~]=size(carteInput);
        [coord1,coord2]=meshgrid(1:sz,1:sz);
        coord1=2*(coord1-mean(mean(coord1)))/sz;
        coord2=2*(coord2-mean(mean(coord2)))/sz;
        if mode == 'Z'
            A=fitZernike(carteInput ,0:max(50,max(numZer)),lambda);
            coord2=flipud(coord2);
            [coord1,coord2]=cart2pol(coord1,coord2);
        elseif mode == 'L'
            A=fitLegendre(carteInput ,numZer,lambda);
        end
        carteOutput=carteInput-genererCarte(coord1,coord2,A(numZer+1),lambda,mode);
      function [coefZernike] = fitZernike(carte, NumZernike,lambda)
        [n,m] = size(carte);
        dx = 2/min(n,m);
        [coord2,coord1] = meshgrid(1:m,1:n);
        coord2=(coord2-mean(coord2(:)))/floor(m/2);
        coord1=(coord1-mean(coord1(:)))/floor(m/2);
        coord1=flipud(coord1);
        [theta,r] = cart2pol(coord2,coord1);
        k=0;
        indices=~isnan(carte);
        for i = NumZernike
            k = k+1;
            A=base(theta,r,i,lambda,'Z');
            T(:,k) = A(indices);
        end
        coefZernike=T\carte(indices);
      end 
      function [carte] = genererCarte(coord1,coord2,coefficients,lambda,mode)
        carte = zeros(size(coord1));
        for i = 1:length(coefficients)
            carte = carte + coefficients(i)*base(coord1,coord2,i-1,lambda,mode);
        end
    end
        function A = base(coord1,coord2,term,lambda,mode)
        if strcmp(mode,'Z')
            A = basezernike(coord1,coord2,term,lambda);
        elseif strcmp(mode,'L')
            A = baselegendre(coord1,coord2,term,lambda);
        end
    end
    function [coefLegendre]=fitLegendre(carte, NumLegendre,lambda)
        [n,m] = size(carte);
        [coord2,coord1] = meshgrid(1:m,1:n);
        coord2=(coord2-mean(coord2(:)))/floor(m/2);coord1=(coord1-mean(coord1(:)))/floor(m/2);
        k=0;
        indices=~isnan(carte);
        for i = NumLegendre
            k = k+1;
            A=base(coord2,coord1,i,lambda,'L');
            T(:,k) = A(indices);
        end
        coefLegendre=T\carte(indices);
    end
    
    end
