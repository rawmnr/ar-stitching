%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Code de stitching algo_IDOINE_stitching %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% A.DOCHE 20201114                        %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Certaines fonctions son hérités de la   %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Toolbox MSO.                            %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% ENTREES : Path_Param, Path_cartes :     %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Chemins vers le ficier text de paramètre, %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% chemin vers le dossier contenant les    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% cartes                                  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% SORTIES : map, mismatch                 %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Carte stitchée, carte de mismatch       %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [map, Mismatch]=algo_Stitching(Path_Param, Path_cartes)

%Initialisation des paramètres
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[NEOSS_Param,TableData]=lectureParametres(Path_Param, Path_cartes);

%Code
%Cette fonction sert à créer la carte moyenne des SSPP pour supprimer les
%HF de CS avant de réaliser le FIT MLR
[carte_random]=calculCarteRandomSmart(TableData,NEOSS_Param);

%Cette fonction réalise le FIT de TP, CS et des coefficients d'alignement.
tic
[x,carte_Instrument]=MLR(TableData,carte_random,NEOSS_Param);
toc

%Cette fonction recolle les SSPP pour réaliser la reconstruction de la
%carte finale
[map,Mismatch]=stitchingSspp(x,NEOSS_Param,TableData,carte_Instrument);

fclose all;


%%Fonctions de niveau 1

%Cette fonction lit les paramètres dans le ficheirs txt de paramtres, et
%créé la struture NEOSS_Param qui contient tous les paramètres (pour
%limiter le nombre de paramètre par fonction
    function [NEOSS_Param,TableData]=lectureParametres(Path_Param, Path_cartes)
        ParametreTxt=LectureFichierParam(Path_Param);
        NEOSS_Param.RpupilleCS=str2double(ParametreTxt{1});
        NEOSS_Param.resolutionCS=str2double(ParametreTxt{2});
        NEOSS_Param.resolutionTP=str2double(ParametreTxt{3});
        NEOSS_Param.RpupilleTP=str2double(ParametreTxt{4});
        NEOSS_Param.lambda=str2double(ParametreTxt{5});
        NEOSS_Param.nb_cartes=str2double(ParametreTxt{6});
        NEOSS_Param.sigma=str2double(ParametreTxt{7});
        NEOSS_Param.mismatch=str2double(ParametreTxt{8});
        NEOSS_Param.mode_TP=ParametreTxt{10};
        NEOSS_Param.mode_CS=ParametreTxt{11};
        NEOSS_Param.indice_alignement=str2double(ParametreTxt{12}):str2double(ParametreTxt{13});
        NEOSS_Param.indice_CS=str2double(ParametreTxt{14}):str2double(ParametreTxt{15});
        NEOSS_Param.indice_TP=str2double(ParametreTxt{16}):str2double(ParametreTxt{17});
        NEOSS_Param.limit=str2double(ParametreTxt{18});
        NEOSS_Param.supportage=str2double(ParametreTxt{19});
        NEOSS_Param.pathSupportage=ParametreTxt{20};
        NEOSS_Param.SystemeCoordonnees=ParametreTxt{21};
        
        %REmplacement Fichier Param
        if strcmp(NEOSS_Param.SystemeCoordonnees,'IDOINE')
            [kmem,l]=meshgrid(1:NEOSS_Param.resolutionTP,1:NEOSS_Param.resolutionTP);
            f1=@(x) (1/(NEOSS_Param.sigma-1))*(x-1);
            f2=@(x) (-1/(NEOSS_Param.sigma-1))*(x-128);
            Amem=min(1,min(f1(kmem),f2(kmem)));
            Bmem=min(1,min(f1(l),f2(l)));
            NEOSS_Param.cartePonderation=min(Amem,Bmem);
        else
            [k,l]=meshgrid(1:NEOSS_Param.resolutionTP,1:NEOSS_Param.resolutionTP);
            k=k-NEOSS_Param.resolutionTP/2;
            l=l-NEOSS_Param.resolutionTP/2;
            NEOSS_Param.cartePonderation=troncatureCercle(max(0.1,min(1,exp(((NEOSS_Param.resolutionTP/2-NEOSS_Param.sigma)^2-(k.^2+l.^2))/(2*NEOSS_Param.sigma^2)))));
        end
        
        %Creation Exemple
        NEOSS_Param.Path_svg=Path_cartes;
        NEOSS_Param.Path_Param=Path_Param;
        NEOSS_Param.Path_Position=[Path_cartes ParametreTxt{9}];
        NEOSS_Param.name='Carte_CS_Calculee.opd';
        [NEOSS_Param.Coord1,NEOSS_Param.Coord2]=getSsppCoordinates(NEOSS_Param.Path_Position);
        
        %ICI ce trouve la lecture des cartes dans le dossier. L'hypthèse
        %est que les cartes des SSPP sont en .Opd, avec les noms :
        %C0XXAP_TP.Opd ou XX est le numéro de la SSPP.
        for indiceSSPP=1:NEOSS_Param.nb_cartes
            fidSSPP=fopen([NEOSS_Param.Path_svg 'C' sprintf('%03i',indiceSSPP) 'AP_TP.Opd']);
            outputSSPP=readOpd(fidSSPP);
            fclose(fidSSPP);
            TableData(indiceSSPP,:)=reshape(outputSSPP.carte,1,length(outputSSPP.carte)^2);
        end
        
    end

%Cette fonction calcule intelligemment la carte random
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
        svgCarte(carteRandom,NEOSS_Param,'carteRandom.Opd');
    end

%Cette fonciton réalise le fit de CS + TP+ Coeff alignement. Le vecteur x
%contient les coefficients d'alignement et les coefficients du fit de CS.
%Carte_instrument est la carte de la TP calculée
    function [x,carte_Instrument] = MLR(TableData,carte_random,NEOSS_Param)
        nb_term_alignement = length(NEOSS_Param.indice_alignement);
        nb_term_HA =length(NEOSS_Param.indice_CS)+length(NEOSS_Param.indice_TP);
        nb_elmt_y_i = nb_term_HA + nb_term_alignement;
        M = sparse(zeros(NEOSS_Param.nb_cartes * nb_elmt_y_i, nb_term_HA + nb_term_alignement*NEOSS_Param.nb_cartes));
        y = zeros(NEOSS_Param.nb_cartes * nb_elmt_y_i,1);
        [coord1_TP_save,coord2_TP_save] = grille(carte_random,1,NEOSS_Param.mode_TP,[0 0 0 0],NEOSS_Param);
        masqueRecouvrement=calculMasquerecouvrement(TableData,NEOSS_Param);
        for numero_sspp = 1:NEOSS_Param.nb_cartes
            carte = reshape(TableData(numero_sspp,:),NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
            carte = carte - carte_random;
            param=calculParametreGrille(NEOSS_Param,numero_sspp);
            [coord1_CS,coord2_CS] =grille(carte,NEOSS_Param.resolutionTP/NEOSS_Param.resolutionCS,NEOSS_Param.mode_CS,param,NEOSS_Param);
            coord1_TP = coord1_TP_save; coord2_TP = coord2_TP_save;
            [donnees_non_masquees, carte]=masquageIntersectRecouvrement(carte,numero_sspp, masqueRecouvrement, NEOSS_Param);
            [carte,ensCoord]=masquageDonnees(donnees_non_masquees,carte,coord1_TP,coord2_TP,coord1_CS,coord2_CS);
            
            T=remplissageMatriceFit(NEOSS_Param,nb_elmt_y_i,ensCoord);
            [U,~,~] = svd(T,0);
            inv_U = pinv(U);
            y_i = inv_U*carte(:);
            M_i = inv_U*T;
            [indices]=calculIndices(numero_sspp,nb_elmt_y_i,nb_term_HA,nb_term_alignement);

            y(indices.Y) = y_i;
            M(indices.M1a,indices.M1b) = M_i(:,end-nb_term_HA+1:end);
            M(indices.M2a,indices.M2b) = M_i(:,1:nb_term_alignement);
        end
        
        x=M\y;
        
        BFResidu(NEOSS_Param.indice_TP+1)=x(1:length(NEOSS_Param.indice_TP))
        x=x(length(NEOSS_Param.indice_CS)+length(NEOSS_Param.indice_TP)+1:end);
        if NEOSS_Param.mode_TP=='L'
             carte_Instrument=carte_random+genererCarte(coord1_TP_save,coord2_TP_save,BFResidu,NEOSS_Param.lambda,NEOSS_Param.mode_TP);
        elseif NEOSS_Param.mode_TP=='Z'
             carte_Instrument=troncatureCercle(carte_random+genererCarte(coord1_TP_save,coord2_TP_save,BFResidu,NEOSS_Param.lambda,NEOSS_Param.mode_TP));
        end
        svgCarte(carte_Instrument,NEOSS_Param,'carte_Instrument.Opd');
    end

%Cette fonction recolle les SSPP en utilisant la pondération définie
    function [CarteCS,Mismatch]=stitchingSspp(x,NEOSS_Param,TableData,carte_Instrument)
        nbAlignement=numel(NEOSS_Param.indice_alignement);
        CarteCS=zeros(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS);
        ponderation=zeros(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS);
        ponderation2=ponderation;
        Mismatch2=zeros(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS);
        for i=1:NEOSS_Param.nb_cartes
            carteSSPP=reshape(TableData(i,:),NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP)-carte_Instrument;
            A=calculCarteAlignement(carteSSPP,x(nbAlignement*(i-1)+1:nbAlignement*i),NEOSS_Param);
            carteSSPP=carteSSPP-A;
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
                
            end
            fclose all;
        end
        
        CarteCS=CarteCS./ponderation;
        Mismatch = sqrt((Mismatch2./ponderation) - CarteCS.^2);
        Mismatch(Mismatch<0.001)=nan;
        
        %Soustraction carte de supportage inutile pour TALISSMAN
        if NEOSS_Param.supportage==1
            fid=fopen(NEOSS_Param.pathSupportage);
            output=readOpd(fid);
            Supportage=rotate_image_ADO(output.carte,90);
            CarteCS=CarteCS-Supportage;
            Supportage=rotate_image_ADO(CarteCS,-90);
            CarteCS=fliplr(CarteCS);
        end

        svgCarte(CarteCS,NEOSS_Param,NEOSS_Param.name);
        svgCarte(Mismatch,NEOSS_Param,'Mismatch.Opd');
        fclose all;

    end

%%Fonctions de niveau 2

%Cette fonction réinterpole les SSPPPS
    function CarteOutput = replacementSsppPolaire(CarteInput,ii,Param)
        CarteOutput=resizeCarte(CarteInput, Param.resolutionCS);
        CarteOutput=troncatureCercle(reinterpADO(-Param.Coord1(ii),0,CarteOutput));
        CarteOutput=rotate_image_ADO(CarteOutput,-Param.Coord2(ii));
        CarteOutput=resizeCarte(CarteOutput, Param.resolutionCS);
    end

%Cette fonction lit le fichier txt de paramètres
    function [ResultArray] = LectureFichierParam(filename)
        fid = fopen(filename,'r');
        tline = fgetl(fid);
        i=0;
        while ischar(tline)
            i = i+1;
            temp = strsplit(tline,{';'});
            ResultArray{i} = temp{1};
            tline = fgetl(fid);
        end
        fclose(fid);
    end

%Cette fonction lit les coordonnées des SSPP
    function [Coord1,Coord2]=getSsppCoordinates(path)
        [position_sspp] = getPositionSspp(path);
        for ii=1:length(position_sspp)
            Coord1(ii)=-position_sspp{ii,1};
            Coord2(ii)=-position_sspp{ii,2};
        end
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

%Permet de limiter le fit à la zone de recouvrement
    function masqueRecouvrement=calculMasquerecouvrement(TableData,NEOSS_Param)
        masqueRecouvrement=zeros(NEOSS_Param.resolutionCS,NEOSS_Param.resolutionCS);
        for ii=1:NEOSS_Param.nb_cartes
            carteSSPP=reshape(TableData(ii,:),NEOSS_Param.resolutionTP,NEOSS_Param.resolutionTP);
            [X,Y]=CalculXY(NEOSS_Param,ii);
            A=~isnan(reinterpSspp(X,Y,carteSSPP,NEOSS_Param));
            masqueRecouvrement=masqueRecouvrement+A;
        end
        masqueRecouvrement=masqueRecouvrement>1;
        [ii,jj]=meshgrid(1:NEOSS_Param.resolutionCS,1:NEOSS_Param.resolutionCS);
        if strcmp(NEOSS_Param.mode_CS,'Z')
            masqueRecouvrement((2*(ii-0.5*NEOSS_Param.resolutionCS)/NEOSS_Param.resolutionCS).^2+(2*(jj-0.5*NEOSS_Param.resolutionCS)/NEOSS_Param.resolutionCS).^2>1)=0;
        end
    end

%Permet de lire dans les paramètre les limites du meshgrid servant à
%évaluer les polynomes de zernike ou Legendre au niveau d'une sous-pupille
    function param=calculParametreGrille(ParametresNEOSS,ii)
        if strcmp(ParametresNEOSS.SystemeCoordonnees,'polaire')
            param = [0,0,-2*ParametresNEOSS.Coord1(ii)/ParametresNEOSS.resolutionCS,-ParametresNEOSS.Coord2(ii)];
        elseif strcmp(ParametresNEOSS.SystemeCoordonnees,'IDOINE')
            [X,Y]=CalculXY(ParametresNEOSS,ii);
            param = [2*X/ParametresNEOSS.resolutionCS,2*Y/ParametresNEOSS.resolutionCS, 0, 0];
        end
    end

%Fonction simple servant à masquer des données
    function [u,v]=masquageDonnees(booleens, carte1,carte2,carte3,carte4,carte5)
        u = carte1(booleens);
        v.coord1_TP = carte2(booleens);
        v.coord2_TP = carte3(booleens);
        v.coord1_CS = carte4(booleens);
        v.coord2_CS = carte5(booleens);
    end

%Fonction importante de MLR servant à remplir la matrice de FIT avec
%l'évaluation des polynomes de Zernike ou LEgendre.
    function T=remplissageMatriceFit(Param,length,Coordinates)
        [sz1,sz2]=size(Coordinates.coord1_TP);
        T = zeros(sz1*sz2, length);
        k = 0;
        for i = Param.indice_alignement
            if Param.mode_TP=='L'
                if i<3
                    k = k+1;
                    A=base(Coordinates.coord1_TP,Coordinates.coord2_TP,i,Param.lambda,Param.mode_TP);
                    T(:,k) = A(:);
                elseif i==3
                    k = k+1;
                    A=base(Coordinates.coord1_TP,Coordinates.coord2_TP,3,Param.lambda,Param.mode_TP)+base(Coordinates.coord1_TP,Coordinates.coord2_TP,5,Param.lambda,Param.mode_TP);
                    T(:,k) = A(:);
                elseif i==4
                    k = k+1;
                    A=base(Coordinates.coord1_TP,Coordinates.coord2_TP,4,Param.lambda,Param.mode_TP);
                    T(:,k) = A(:);
                elseif i==5
                    k = k+1;
                    A=base(Coordinates.coord1_TP,Coordinates.coord2_TP,3,Param.lambda,Param.mode_TP)-base(Coordinates.coord1_TP,Coordinates.coord2_TP,5,Param.lambda,Param.mode_TP);
                    T(:,k) = A(:);
                end
            else
                k = k+1;
                A=base(Coordinates.coord1_TP,Coordinates.coord2_TP,i,Param.lambda,Param.mode_TP);
                T(:,k) = A(:);
            end
        end
        
        for i = Param.indice_TP
            if (Param.mode_TP=='L' && i==5)
                k = k+1;
                A=base(Coordinates.coord1_TP,Coordinates.coord2_TP,3,Param.lambda,Param.mode_TP)-base(Coordinates.coord1_TP,Coordinates.coord2_TP,5,Param.lambda,Param.mode_TP);
                T(:,k) = A(:);
            else
                k = k+1;
                A=base(Coordinates.coord1_TP,Coordinates.coord2_TP,i,Param.lambda,Param.mode_TP);
                T(:,k) = A(:);
            end
        end
        
        for i = Param.indice_CS
            k = k+1;
            A=base(Coordinates.coord1_CS,Coordinates.coord2_CS,i,Param.lambda,Param.mode_CS);
            T(:,k) = A(:);
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

%Fonction simple servant à enregistrer une carte.
    function []=svgCarte(Carte,parametres,name)
        fid=fopen([parametres.Path_svg name],'w+');
        writeOpdUnit(Carte,name,fid,parametres.RpupilleTP,'lambda',parametres.lambda);
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

%CEtte fonction limite le fit au zone de recouvrement ou les pixels des
%SSPP sont définis.
    function [donnees_non_masquees,carteSSPP]=masquageIntersectRecouvrement(carteSSPP,numeroSSPP, CarteRecouvrement, Parameters)
        if strcmp(Parameters.SystemeCoordonnees,'IDOINE')
            %Create Mask
            [X,Y]=CalculXY(Parameters,numeroSSPP);
            masqueIntersection=not(isnan(reinterpSspp(X,Y,carteSSPP, Parameters))) & CarteRecouvrement~=0;
            masqueIntersection=double(masqueIntersection);
            [carteRecadree]=reinterpADO(-X,-Y,masqueIntersection);
            
            masqueIntersection=resizeCarte(carteRecadree,Parameters.resolutionTP);
            %Correct SSPP
            carteSSPP(masqueIntersection==0)=nan;
        end
        donnees_non_masquees = find(~isnan(carteSSPP));
    end


%%Fonctions de niveau 3

%Lecture des positions des SSPP
    function [pos_sspp] = getPositionSspp(filename)
        fid = fopen(filename,'r');
        tline = fgetl(fid);
        i=0;
        while ischar(tline)
            i = i+1;
            temp = strsplit(tline,{'=',';'});
            pos_sspp{i,1} = str2num(temp{2});
            pos_sspp{i,2} = str2num(temp{3});
            tline = fgetl(fid);
        end
        fclose(fid);
    end

%Conversion des coordonnées fournies de pixels en mm
    function [X,Y]=CalculXY(NEOSS_Param,ii)
        X=NEOSS_Param.Coord1(ii)*NEOSS_Param.resolutionCS/(2*NEOSS_Param.RpupilleCS);
        Y=NEOSS_Param.Coord2(ii)*NEOSS_Param.resolutionCS/(2*NEOSS_Param.RpupilleCS);
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

%Convertir des coefficients de Z ou L en carte.
    function [carte] = genererCarte(coord1,coord2,coefficients,lambda,mode)
        carte = zeros(size(coord1));
        for i = 1:length(coefficients)
            carte = carte + coefficients(i)*base(coord1,coord2,i-1,lambda,mode);
        end
    end

%Ecriture d'un tableau en fichier .Opd sur le disque
    function []=writeOpdUnit(carte,fileName,fid,Rpupille, unit,lambda )
        
        switch nargin
            case 4
                lambda=632.8;
                [fileName,pathName] = uiputfile({'*.opd';'*.Opd';'*.OPD'},'Enregistrer une carte');
                fid = fopen([pathName fileName],'w');
            case 5
                [fileName,pathName] = uiputfile({'*.opd';'*.Opd';'*.OPD'},'Enregistrer une carte');
                fid = fopen([pathName fileName],'w');
        end
        if strcmp(fileName(length(fileName)-3:length(fileName)),'.opd')
            fileName(length(fileName)-3:length(fileName))='    ';
        end
        if length(fileName)>=20
            nomCarte=fileName(1:20);
        else
            nomCarte=fileName;
            for i=1:(20-length(fileName))
                nomCarte=[nomCarte ' ']; %#ok<*AGROW>
            end
        end
        if strcmp(unit,'m')
            carte=carte*1e9/lambda;
        else
            carte=carte/lambda;
        end
        V='                    ';
        B=[V V V V V '    '];
        dateHeure=round(clock);
        year=num2str(dateHeure(1));
        month=num2str(dateHeure(2));
        day=num2str(dateHeure(3));
        hour=num2str(dateHeure(4));
        minute=num2str(dateHeure(5));
        seconds=num2str(dateHeure(6));
        if length(year)<2
            year=['0' year];
        end
        if length(month)<2
            month=['0' month];
        end
        if length(day)<2
            day=['0' day];
        end
        if length(hour)<2
            hour=['0' hour];
        end
        if length(minute)<2
            minute=['0' minute];
        end
        if length(seconds)<2
            seconds=['0' seconds];
        end
        fwrite(fid,nomCarte,'char');
        fwrite(fid,[' ' day(1:2) '/' month(1:2) '/' year(3:4) ' ' hour(1:2) ':' minute(1:2) ':' seconds(1:2) ' '],'char'); %date et heure : 19 caractères et respect de la nomenclature
        fwrite(fid,B,'char');
        fwrite(fid,'O','char');
        fwrite(fid,'                ','char'); %Origine de la carte (16 carac)
        fwrite(fid,lambda,'float32'); %Longueur d'onde de la carte : 1 réel
        fwrite(fid,0,'float32'); %Wedge Factor
        fwrite(fid,double(Rpupille),'float32'); %Rayon de la pupille
        a=single(ones(1,3));fwrite(fid,a,'float32');
        fwrite(fid,0,'float32');
        a=single(ones(1,13));fwrite(fid,a,'float32');
        S=size(carte);
        fwrite(fid,52,'int16');
        fwrite(fid,S(1),'int16');
        fwrite(fid,S(2),'int16');
        fwrite(fid,[1 1 1 1 2],'int16');
        fwrite(fid,1,'int16');
        fwrite(fid,[1 1 1 1 1 1 1 1 1 1 1],'int16');
        carteMod=flip(carte);
        carteMod(isnan(carteMod))=Inf;
        fwrite(fid,carteMod,'float32');
        fclose(fid);
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

%Lire une carte OPD (cette fonction fourni une structure dont struct.carte
%est la carte voulue.
    function output=readOpd(fid)
        global MAP
        MAP.titre = (char(fread(fid,20,'char')))';         % titre de la carte
        MAP.dummy1 = char(fread(fid,1,'char'));
        MAP.date = (char(fread(fid,17,'char')))';          % date et heure de création
        MAP.dummy2 = char(fread(fid,1,'char'));
        MAP.commentaires = (char(fread(fid,104,'char')))'; % commentaires
        MAP.dummy3 = char(fread(fid,1,'char'));            % Lettre 'O' obligatoire
        MAP.logiciel = (char(fread(fid,16,'char')))';      % Logiciel d'origine
        MAP.lambda = fread(fid,1,'float');              % longueur d'onde de la carte
        MAP.wedge = fread(fid,1,'float');               % ne sert à rien dans Warpp
        MAP.rayon = fread(fid,1,'float');               % rayon de la pupille en mm
        MAP.dummy4 = fread(fid,3,'float');              % 3 réels qui ne sont pas utilisés
        MAP.fnumber = fread(fid,1,'float');             % nombre d'ouverture
        MAP.fiducialpoints = fread(fid,8,'float');      % nombres fiduciaux
        MAP.cobsc = fread(fid,1,'float');               % rapport d'obturation centrale
        MAP.apod = fread(fid,2,'float');                % terme d'apodisation
        MAP.dummy5 = fread(fid,2,'float');              % rien
        MAP.signature = fread(fid,1,'short');           % indique la simple ou double précision
        MAP.largeur = fread(fid,1,'short');             % nombre de colonnes de la carte
        MAP.hauteur = fread(fid,1,'short');             % nombre de lignes de la carte
        MAP.aperturecode = fread(fid,1,'short');
        MAP.fringepoints = fread(fid,1,'short');
        MAP.dummy6 = fread(fid,2,'short');
        MAP.flag = fread(fid,1,'short');
        MAP.coeffmult = fread(fid,1,'short');           % facteur de précision, utilisés uniquement pour les cartes simple précision
        MAP.dummy7 = fread(fid,2,'short');
        MAP.gridsize = fread(fid,1,'short');            % ne sert à rien !!
        MAP.dummy8 = fread(fid,1,'short');
        MAP.dummy9 = fread(fid,7,'short');
        if (MAP.rayon==0)
            MAP.rayon = 100 ;
        end
        N = MAP.largeur*MAP.hauteur ;
        if (MAP.signature==18)
            MAP.rawdata = fread(fid,N,'short');
        end
        if (MAP.signature==52)
            MAP.rawdata = fread(fid,N,'float');
        end
        %gestion du cas simple précision
        if (MAP.signature==18)
            % ne garder que les données valides de la rawcarte
            masque = find(MAP.rawdata~=32767) ;
            MAP.data = MAP.rawdata(masque) ;
            % mettre en forme pour l'affichage
            masque = find(MAP.rawdata==32767) ;
            MAP.carte = reshape(MAP.rawdata,MAP.largeur,MAP.hauteur);
            MAP.carte(masque) = nan ;
            % valeurs en nm au lieu de lambda
            MAP.carte = MAP.carte/MAP.coeffmult*MAP.lambda ;
            MAP.data = MAP.data/MAP.coeffmult*MAP.lambda ;
            MAP.stat.min = min(min(MAP.data));
            MAP.stat.max = max(max(MAP.data));
        end
        %gestion du cas double précision
        if (MAP.signature==52)
            % ne garder que les données valides de la rawcarte
            masque = find(MAP.rawdata~=Inf) ;
            MAP.data = MAP.rawdata(masque) ;
            % mettre en forme pour l'affichage
            masque = find(MAP.rawdata==Inf) ;
            MAP.carte = reshape(MAP.rawdata,MAP.largeur,MAP.hauteur);
            MAP.carte(masque) = nan ;
            % valeurs en nm au lieu de lambda
            MAP.carte = MAP.carte*MAP.lambda ;
            MAP.data = MAP.data*MAP.lambda ;
            MAP.stat.min = min(min(MAP.data));
            MAP.stat.max = max(max(MAP.data));
        end
        MAP.carte = flipud(MAP.carte); % ligne qui substitue 'MAP.carte = flip(MAP.carte,1)' dans MATLAB2012
        % CALCUL DE CERTAINES PROPRIETES DE LA CARTE
        % GENERATION DES SORTIES DE LA FONCTION READOPD
        tmp = size(MAP.data) ;
        MAP.stat.n = tmp(1,1) ;
        MAP.stat.piston = mean(MAP.data) ;
        MAP.stat.ptv = MAP.stat.max - MAP.stat.min ;
        MAP.stat.rms = std(MAP.data,1) ;
        MAP.stat.piston;
        MAP.stat.ptv;
        MAP.stat.rms;
        output = MAP;
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

%Reduction taille d'un carte
    function CarteRaccourcie=raccourcirCarte(carte, ResolutionFinale)
        CarteRaccourcie=nan(ResolutionFinale,ResolutionFinale);
        [sz1,~]=size(carte);
        xA=floor((sz1-ResolutionFinale)/2);
        CarteRaccourcie=carte(xA+1:xA+ResolutionFinale,xA+1:xA+ResolutionFinale);
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

end