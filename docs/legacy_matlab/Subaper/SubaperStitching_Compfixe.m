%%Fonction SCRIPT

%Cette fonction permet de calculer la carte stitchée a partir des autres

%Entrées : cartes à stitcher sous la forme d'un tableau de cartes opd par sous-pupilles
%compensateurs et leur type (Legendre ou Zernike)
%res=résolution de la carte finale
%Compfix= compensateurs constants dans la pupille instrument de format :
%cell de cartes opd Compfix{1,j}{1,i} j=numéro compensateur, i=numéro sous-pupilles
%Sorties : carte stitchée, carte de mismatch, moyenne et rms de la carte de mismatch
% vecteur cs des amplitudes des compensateurs

function [CarteOut,MM,moy,rms,cs,a,map2]=SubaperStitching_Compfixe(Maps, Type, Compensateurs, Cartes, res,Compfix)
    arguments
        Maps cell
        Type {mustBeMember(Type,["L","Z"])} = "L"
        Compensateurs (1,:) double = [1:3]
        Cartes double = 1
        res double = Maps{1,1}.largeur
        Compfix cell = {}
     end
    
    dim=length(Maps);
	sz=size(Compfix);
    TableData=zeros(res,res,dim);
    %On convertit les cartes au bon format
    if sz(1) ~= 0
        Compfixes=zeros(res^2,dim,sz(2));
		%boucle sur le nombre de compensateurs fixes
        for j=1:sz(2)
            %boucle sur le nombre de sous-pupilles
            for i=1:dim(1)
                temp=Compfix{1,j}{1,i};
                temp2=Maps{1,i};
                if res ~= Maps{1,1}.largeur
                    temp2=Resize(temp2,res);                    
                end
                if temp.largeur ~= res
                    temp=Resize(temp,res);
                end
                Compfixes(:,i,j)=reshape(temp.carte,[],1);
                TableData(:,:,i)=temp2.carte;
            end
        end
    else
        Compfixes=[];
        for i=1:dim(1)
            temp=Maps{1,i};
            if res ~= Maps{1,1}.largeur 
                temp=Resize(temp,res);
            end
            TableData(:,:,i)=temp.carte;
        end
    end

    %Initialise les cartes de sortie
    CarteOut=Maps{1};
    CarteOut.titre='Stiched Map';
    MM=Maps{1};
    MM.titre='Mismatch';
    if sz(1) == 0
        [CarteOut.carte,MM.carte,cs,a]=Subaper_Core_v2(TableData, Type, Compensateurs, Cartes);
        moy='non calculé';
        rms='non calculé';
    else
        [CarteOut.carte,MM.carte,moy,rms,cs,a]=Subaper_Compensateurs_fixes(TableData, Type, Compensateurs,Cartes, Compfixes);
    end
%  Le code ci-dessous marche aussi mais est plus lent dans le cas où il n'y a pas de compensateurs fixes   
% [CarteOut.carte,MM.carte,moy,rms,cs]=Subaper_Compensateurs_fixes(TableData, Type, Compensateurs,Compfixes);
CarteOut=updateProp(CarteOut);
MM=updateProp(MM);
end