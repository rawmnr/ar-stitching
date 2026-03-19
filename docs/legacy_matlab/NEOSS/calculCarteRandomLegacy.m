 function [carteRandom]=calculCarteRandomSmart(TableData,NEOSS_Param)
        carteRandom=zeros(NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
        limitInf=NEOSS_Param.limit;
        ponderation=ones(NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
        for i=1:NEOSS_Param.nb_cartes
            carte=reshape(TableData(i,:),NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
            A=~isnan(carte);
            ratio=sum(sum(A))/NEOSS_Param.resolutionTP^2;
            if ratio>limitInf
                ponderation=ones(NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
                carteRandom=carte;
                limitInf=ratio;
            elseif ratio==limitInf
                for ii=1:NEOSS_Param.resolutionTP
                    ponderation(ii,:)=(~isnan(A(ii,:)).*(~isnan(A(ii,:)))).*(ponderation(ii,:)+1);
                    carteRandom(ii,:)=carteRandom(ii,:)+ (~isnan(A(ii,:)).*(~isnan(A(ii,:)))).*carte(ii,:);
                end
            end
        end
        carteRandom=carteRandom./ponderation;
        carteRandom=removeZernikeLegendre(carteRandom, NEOSS_Param.mode_TP, 0:3,NEOSS_Param.lambda,NEOSS_Param);
 end


 %Cette fonction retire des coefficients de Zernikes ou Legendre à une
%carte
    function [carteOutput]=removeZernikeLegendre(carteInput, mode, numZer, lambda,Param_NEOSS)
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
    end


    %Fit d'une surface sur les polynomes de Legendre.
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


    %Contruction d'une carte à partir d'une liste de coefficients de Z ou L
    function A = base(coord1,coord2,term,lambda,mode)
        if strcmp(mode,'Z')
            A = basezernike(coord1,coord2,term,lambda);
        elseif strcmp(mode,'L')
            A = baselegendre(coord1,coord2,term,lambda);
        end
    end

%    Contruction d'une carte à partir d'une liste de coefficients de Z 
    function A = basezernike(theta,r,term,lambda,modo)
        switch nargin
            case 4
                modo='pv';
        end
        [zn,zm]=zernikeISO2nm(term+1);
        zn=zn(term+1);  % +1 Parce que ISO Z_0 est le premier
        zm=zm(term+1);  % +1 Parce que ISO Z_0 est le premier
        Znm=zernike(zn,zm,modo);
        if zm>=0
            A=polyval(Znm,r).*cos(zm*theta);
        else
            A=polyval(Znm,r).*sin(-zm*theta);
        end
        A=A*lambda;
    end

%Contruction d'une carte à partir d'une liste de coefficients de L
    function A = baselegendre(x,y,term,lambda)
        i=1;
        a=0;
        ix=zeros(term+1,1);
        iy=zeros(term+1,1);
        while i <= term+1
            for j=0:a
                ix(i)=a-j;
                iy(i)=j;
                i=i+1;
                if i>term +1, break, end
            end
            a=a+1;
        end
        A = polyval(legendrecoeff(ix(term+1)),x).*polyval(legendrecoeff(iy(term+1)),y);
        A = A*lambda;
    end

%Définition des coefficients de L
    function Pn = legendrecoeff(n)
        Pn = zeros(1,n+1);
        for k=0:floor(n/2)
            Pn(1+2*k) = (-1)^k * nchoosek(n,k) * nchoosek(2*n-2*k,n);
        end
        Pn = 2^(-n)*Pn;
    end

%Définition des coefficients de Z
    function Znm=zernike(n,m,fact)
        if (nargin<2)
            error('Pas assez d''arguments')
        elseif (nargin>3)
            error('Trop d''arguments')
        else
            if (nargin==2)
                fact='pic';
            else
                if ~strcmp(fact,'pv') && ~strcmp(fact,'rms')
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
    end


%Fonction de calcul des indices pour la fonction MLR : définition de la
%taille des tableau et vecteurs.
    function [indices]=calculIndices(numero_sspp,nb_elmt_y_i,nb_term_HA,nb_alignement_total)
        indices.Y=(numero_sspp-1)*nb_elmt_y_i+1 : numero_sspp*nb_elmt_y_i;
        indices.M1a=(numero_sspp-1)*nb_elmt_y_i+1:numero_sspp*nb_elmt_y_i;
        indices.M1b=1:nb_term_HA;
        indices.M2a=(numero_sspp-1)*nb_elmt_y_i+1:numero_sspp*nb_elmt_y_i ;
        indices.M2b=nb_term_HA+1 + nb_alignement_total*(numero_sspp-1):nb_term_HA+1 + nb_alignement_total-1 + nb_alignement_total*(numero_sspp-1);
    end

    %Convertir des coefficients de Z ou L en carte.
    function [carte] = genererCarte(coord1,coord2,coefficients,lambda,mode)
        carte = zeros(size(coord1));
        for i = 1:length(coefficients)
            carte = carte + coefficients(i)*base(coord1,coord2,i-1,lambda,mode);
        end
    end

    %Fit d'une surface sur les polynomes de zernikes.
    function [coefZernike]=fitZernike(carte, NumZernike,lambda)
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

    %DEfinition des polynomes de zernikes
    function [n,m]=zernikeISO2nm(nterms)
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
    end
