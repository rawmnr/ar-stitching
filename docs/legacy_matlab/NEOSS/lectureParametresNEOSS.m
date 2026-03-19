 function [NEOSS_Param,TableData]=lectureParametresNEOSS(Path_Param, Path_cartes,pos_sspp)
        
        ParametreTxt=readcell(Path_Param);
        NEOSS_Param.RpupilleCS=(ParametreTxt{1,1});
        NEOSS_Param.resolutionCS=(ParametreTxt{2,1});
        NEOSS_Param.resolutionTP=(ParametreTxt{3,1});
        NEOSS_Param.RpupilleTP=(ParametreTxt{4,1});
        NEOSS_Param.lambda=(ParametreTxt{5,1});
        NEOSS_Param.nb_cartes=(ParametreTxt{6,1});
        NEOSS_Param.sigma=(ParametreTxt{7,1});
        NEOSS_Param.mismatch=(ParametreTxt{8,1});
        NEOSS_Param.mode_TP=ParametreTxt{10,1};
        NEOSS_Param.mode_CS=ParametreTxt{11,1};
        NEOSS_Param.indice_alignement=(ParametreTxt{12,1}):(ParametreTxt{13,1});
        NEOSS_Param.indice_CS=(ParametreTxt{14,1}):(ParametreTxt{15,1});
        NEOSS_Param.indice_TP=(ParametreTxt{16,1}):(ParametreTxt{17,1});
        NEOSS_Param.limit=(ParametreTxt{18,1});
        NEOSS_Param.supportage=(ParametreTxt{19,1});
        NEOSS_Param.pathSupportage=ParametreTxt{20,1};
        NEOSS_Param.SystemeCoordonnees=ParametreTxt{21,1};
        
        %REmplacement Fichier Param
        if strcmp(NEOSS_Param.SystemeCoordonnees,'IDOINE')
            [kmem,l]=meshgrid(1:NEOSS_Param.resolutionTP,1:NEOSS_Param.resolutionTP);
            f1=@(x) (1/(NEOSS_Param.sigma-1))*(x-1);
            f2=@(x) (-1/(NEOSS_Param.sigma-1))*(x-128);
            Amem=min(1,min(f1(kmem),f2(kmem)));
            Bmem=min(1,min(f1(l),f2(l)));
            NEOSS_Param.cartePonderation=min(Amem,Bmem);        
        elseif strcmp(NEOSS_Param.SystemeCoordonnees,'IRIDE')
            [k,l]=meshgrid(1:NEOSS_Param.resolutionTP,1:NEOSS_Param.resolutionTP);
            k=k-NEOSS_Param.resolutionTP/2;
            l=l-NEOSS_Param.resolutionTP/2;
            NEOSS_Param.cartePonderation=troncatureCercle(max(0.1,min(1,exp(((NEOSS_Param.resolutionTP/2-NEOSS_Param.sigma)^2-(k.^2+l.^2))/(2*NEOSS_Param.sigma^2)))));
        else
            [k,l]=meshgrid(1:NEOSS_Param.resolutionTP,1:NEOSS_Param.resolutionTP);
            k=k-NEOSS_Param.resolutionTP/2;
            l=l-NEOSS_Param.resolutionTP/2;
            NEOSS_Param.cartePonderation=troncatureCercle(max(0.1,min(1,exp(((NEOSS_Param.resolutionTP/2-NEOSS_Param.sigma)^2-(k.^2+l.^2))/(2*NEOSS_Param.sigma^2)))));
        end
        
        %Creation Exemple
        NEOSS_Param.Path_Param=Path_Param;
        NEOSS_Param.Path_Position=pos_sspp;%Modification EMC 05.2021
        [NEOSS_Param.Coord1,NEOSS_Param.Coord2]=getSsppCoordinates(NEOSS_Param.Path_Position);
        
        %lecture des cartes
        for indiceSSPP=1:NEOSS_Param.nb_cartes
            outputSSPP=readOPD(Path_cartes{indiceSSPP});
            TableData(indiceSSPP,:)=reshape(flipud(outputSSPP.carte),1,length(outputSSPP.carte)^2);
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