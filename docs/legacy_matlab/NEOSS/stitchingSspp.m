%Cette fonction recolle les SSPP en utilisant la pondération définie
function [CarteCS,Mismatch,RMS_Diff_Matrix,sppAdjMatrix,sspStack]=stitchingSspp(x,NEOSS_Param,TableData,carte_Instrument)
nbAlignement=numel(NEOSS_Param.indice_alignement);
CarteCS=zeros(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS);
ponderation=zeros(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS);
Mismatch2=zeros(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS);
sspStack = NaN(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS,NEOSS_Param.nb_cartes); % Stack pour stocker les cartes recalées
sppAdjMatrix = zeros(NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP,NEOSS_Param.nb_cartes);
for i=1:NEOSS_Param.nb_cartes
    carteSSPP=reshape(TableData(i,:),NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP)-carte_Instrument;
    A=calculCarteAlignement(carteSSPP,x(nbAlignement*(i-1)+1:nbAlignement*i),NEOSS_Param);
    carteSSPP=carteSSPP-A;
    sppAdjMatrix(:,:,i) = carteSSPP; % Store de la carte
    pupillePonderation=NEOSS_Param.cartePonderation;
    if strcmp(NEOSS_Param.SystemeCoordonnees,'IDOINE')
        [X,Y]=CalculXY(NEOSS_Param,i);
        mask=ones(NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
        mask(isnan(carteSSPP))=nan;
        [carteSSPPRecalee,PonderationRecalee]=NaningContour(reinterpSspp(X,Y,carteSSPP, NEOSS_Param),...
            reinterpSspp(X,Y,pupillePonderation,NEOSS_Param),reinterpSspp(X,Y,mask, NEOSS_Param));
        CarteCS=CarteCS+carteSSPPRecalee.*PonderationRecalee;
        Mismatch2=Mismatch2+(carteSSPPRecalee.^2).*PonderationRecalee;
        ponderation=ponderation+PonderationRecalee;
    elseif strcmp(NEOSS_Param.SystemeCoordonnees,'polaire')
        carteSSPP(carteSSPP==0)=nan;
        carteSSPPRecalee=replacementSsppPolaire(carteSSPP,i,NEOSS_Param);
        PonderationRecalee=replacementSsppPolaire(pupillePonderation,i,NEOSS_Param);

        masking=~isnan(carteSSPPRecalee);
        CarteCS(masking)=CarteCS(masking)+carteSSPPRecalee(masking).*PonderationRecalee(masking);
        Mismatch2(masking)=Mismatch2(masking)+(carteSSPPRecalee(masking).^2).*PonderationRecalee(masking);
        ponderation(masking)=ponderation(masking)+PonderationRecalee(masking);
    elseif strcmp(NEOSS_Param.SystemeCoordonnees,'IRIDE')
                [X,Y]=CalculXY(NEOSS_Param,i);
                mask=ones(NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
                mask(isnan(carteSSPP))=nan;
                [carteSSPPRecalee,PonderationRecalee]=NaningContour(reinterpSspp(X,Y,carteSSPP, NEOSS_Param),...
                    reinterpSspp(X,Y,pupillePonderation,NEOSS_Param),reinterpSspp(X,Y,mask, NEOSS_Param));
                CarteCS=CarteCS+carteSSPPRecalee.*PonderationRecalee;
                Mismatch2=Mismatch2+(carteSSPPRecalee.^2).*PonderationRecalee;
                ponderation=ponderation+PonderationRecalee;
    end
    sspStack(:,:,i) = carteSSPPRecalee; % Store de la carte recalée 
end

CarteCS=CarteCS./ponderation;
Mismatch = sqrt((Mismatch2./ponderation) - CarteCS.^2);
Mismatch(Mismatch<0.001)=nan;

%Calcul de la matrice différence RMS
nbCartes = NEOSS_Param.nb_cartes;
RMS_Diff_Matrix = nan(nbCartes, nbCartes);

for i = 1:nbCartes
    for j = i+1:nbCartes
        diffMatrix = sspStack(:, :, i) - sspStack(:, :, j);
        RMS_Diff_Matrix(i, j) = std(std(diffMatrix,'omitnan'),'omitnan');
        RMS_Diff_Matrix(j, i) = RMS_Diff_Matrix(i, j); % Symmetrique
    end
end

%Soustraction carte de supportage inutile pour TALISSMAN
if NEOSS_Param.supportage==1
    fid=fopen(NEOSS_Param.pathSupportage);
    output=readOpd(fid);
    Supportage=rotate_image_ADO(output.carte,90);
    CarteCS=CarteCS-Supportage;
    CarteCS=fliplr(CarteCS);
end


CarteCS=flipud(CarteCS);
Mismatch=flipud(Mismatch);

end


%CalculCarte d'alignement permet de convertir les coefficient d'alignement
%calculer en carte à superposé à une SSPP.
    function [CarteOutput]=calculCarteAlignement(CarteInput,A,parameters)
        [coord1,coord2] = grille(CarteInput,1,parameters.mode_TP,[0,0,0,0],parameters);
        if parameters.mode_TP=='Z'
            CarteOutput = troncatureCercle(genererCarte(coord1,coord2,A,parameters.lambda,parameters.mode_TP));
        elseif parameters.mode_TP=='L'
            if length(A)<4
                CarteOutput = genererCarte(coord1,coord2,A,parameters.lambda,parameters.mode_TP);
            elseif length(A)==4
                CarteOutput = genererCarte(coord1,coord2,[A(1:3);A(4);0;A(4)],parameters.lambda,parameters.mode_TP);
            elseif length(A)==5
                CarteOutput = genererCarte(coord1,coord2,[A(1:3);A(4);A(5);A(4)],parameters.lambda,parameters.mode_TP);
            elseif length(A)==6
                CarteOutput = genererCarte(coord1,coord2,[A(1:3);A(4)+A(6);A(5);A(4)-A(6)],parameters.lambda,parameters.mode_TP);
            elseif length(A)>6
                CarteOutput = genererCarte(coord1,coord2,[A(1:3);A(4)+A(6);A(5);A(4)-A(6);A(7:end)],parameters.lambda,parameters.mode_TP);
            end
        end
    end


    %Troncature d'une carte par le cercle de taille maximale qu'elle contient.
%L'extérieur est laissé "nan"
    function [carteCirculaire]=troncatureCercle(carte)
        sz=size(carte);
        i0=floor(sz(1)/2)+1;
        j0=floor(sz(2)/2)+1;
        carteCirculaire=nan(sz(1),sz(2));
        for i=1:sz(1)
            for j=1:sz(2)
                if (i-i0)^2+(j-j0)^2<=floor(sz(1)/2)^2
                    carteCirculaire(i,j)=carte(i,j);
                end
            end
        end
    end


    %Cette fonciton produit un meshgrid correspondant à une sous-pupilles,
%grace au parametre "param"
    function [coord1,coord2] = grille(carte,rho,mode,param,param_NEOSS)
        trans_rad = param(3);angle_rot = param(4);
        [n,m] = size(carte);[X,Y] = meshgrid(1:m,1:n);
        X=(X-mean(X(:)))/floor(m/2);Y=(Y-mean(Y(:)))/floor(m/2);
        if strcmp(param_NEOSS.SystemeCoordonnees,'polaire')
            Y = flipud(Y);
        end
        X=rho*X+param(1);Y=rho*Y+param(2);
        if strcmp(mode,'L')
            coord1 = X;
            coord2 = Y;
        end
        if strcmp(mode,'Z')
            if trans_rad ~= 0
                X = X + trans_rad;
            end
            [theta,r] = cart2pol(X,Y);
            theta = mod(theta,2*pi);
            if angle_rot ~= 0
                theta = mod(theta + angle_rot*pi/180,2*pi);
            end
            coord1 = theta;
            coord2 = r;
        end
    end


%Convertir des coefficients de Z ou L en carte.
    function [carte] = genererCarte(coord1,coord2,coefficients,lambda,mode)
        carte = zeros(size(coord1));
        for i = 1:length(coefficients)
            carte = carte + coefficients(i)*base(coord1,coord2,i-1,lambda,mode);
        end
    end


    %Contruction d'une carte à partir d'une liste de coefficients de Z ou L
    function A = base(coord1,coord2,term,lambda,mode)
        if strcmp(mode,'Z')
            A = basezernike(coord1,coord2,term,lambda);
        elseif strcmp(mode,'L')
            A = baselegendre(coord1,coord2,term,lambda);
        end
    end

%Contruction d'une carte à partir d'une liste de coefficients de Z 
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

%Cette fonction réinterpole les SSPPPS
    function CarteOutput = replacementSsppPolaire(CarteInput,ii,Param)
        CarteOutput=resizeCarte(CarteInput, Param.resolutionCS);
        CarteOutput=troncatureCercle(reinterpADO(-Param.Coord1(ii),0,CarteOutput));
        CarteOutput=rotate_image_ADO(CarteOutput,-Param.Coord2(ii));
        CarteOutput=resizeCarte(CarteOutput, Param.resolutionCS);
    end


    %Resize d'une carte
    function resizedCarte=resizeCarte(carte, resolutionFinale)
        [sz,~]=size(carte);
        if sz>resolutionFinale
            resizedCarte=raccourcirCarte(carte, resolutionFinale);
        elseif sz<resolutionFinale
            resizedCarte=elargirCarte(carte, resolutionFinale);
        else
            resizedCarte=carte;
        end
    end



%Elargissement de la taille d'une carte
    function CarteElargie=elargirCarte(carte, ResolutionFinale)
        CarteElargie=nan(ResolutionFinale,ResolutionFinale);
        [sz1,~]=size(carte);
        xA=floor((ResolutionFinale-sz1)/2);
        CarteElargie(xA+1:xA+sz1,xA+1:xA+sz1)=carte;
    end


    %Reinterpolation lors d'une translation radiale
    function [carteRecadree]=reinterpADO(TransX,TransY,carte)
        [sz,~]=size(carte);
        [Xq,Yq]=meshgrid(1:sz,1:sz);
        X=Xq+TransX;
        Y=Yq+TransY;
        carteRecadree=interp2(X,Y,carte,Xq,Yq,'cubic',nan);
    end


    %Rotation propre d'une carte.
    function carteTournee=rotate_image_ADO(imageIn,angleInDeg)
        angleInDeg=mod(angleInDeg,360);
        if angleInDeg==90
            carteTournee=rot90(imageIn,1);
        elseif angleInDeg==180
            carteTournee=rot90(imageIn,2);
        elseif angleInDeg==270
            carteTournee=rot90(imageIn,3);
        elseif angleInDeg==0
            carteTournee=imageIn;
        else
            [sz,~]=size(imageIn);
            [Xq,Yq]=meshgrid(1:sz,1:sz);
            Xq=Xq-mean(mean(Xq));
            Yq=Yq-mean(mean(Yq));
            [theta,rho]=cart2pol(Xq,Yq);
            [X,Y]=pol2cart(mod(theta-angleInDeg*pi/180,2*pi),rho);
            carteTournee=reshape(griddata(X(:),Y(:),imageIn(:),Xq(:),Yq(:),'cubic'),sz,sz);
        end
    end

    %Conversion des coordonnées fournies de pixels en mm
    function [X,Y]=CalculXY(NEOSS_Param,ii)
        X=NEOSS_Param.Coord1(ii)*NEOSS_Param.resolutionCS/(2*NEOSS_Param.RpupilleCS);
        Y=NEOSS_Param.Coord2(ii)*NEOSS_Param.resolutionCS/(2*NEOSS_Param.RpupilleCS);
    end

    %Cette fonction sert à reinterpoler une souspupille pour la déplacer en
%coordonnées IDOINEnes.
    function [carteRecadree]=reinterpSspp(TransX,TransY,SSPP, Parameters)
        carteRecadree=nan(Parameters.resolutionCS,Parameters.resolutionCS);
        Xa=Parameters.resolutionCS/2-Parameters.resolutionTP/2+1;
        Xb=Parameters.resolutionTP/2+Parameters.resolutionCS/2;
        carteRecadree(Xa:Xb,Xa:Xb)=SSPP;
        [carteRecadree]=reinterpADO(TransX,TransY,carteRecadree);
    end


%Cette fonction sert à évaluer des cartes sur un masque
    function [carteOutput1,carteOutput2]=NaningContour(carteInput1,carteInput2, carteInput3)
        carteOutput1=carteInput1;
        carteOutput2=carteInput2;
        carteInput3(isnan(carteInput3))=0;
        mask=(carteInput3==0);
        carteOutput1(mask)=0;
        carteOutput2(mask)=0;
    end