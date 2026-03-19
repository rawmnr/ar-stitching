%%Fonction SCR 

%Resize Carte


%Entrées : Carte, nouvelle résolution
%Sorties : Carte 

function CarteOut=Resize(CarteIn,resol)
    
    CarteOut=CarteIn;

    m=resol; %(map size)
    dx=2/m;dy=2/m;
    [X,Y] = meshgrid(  ((1:m)-m/2-0.5)*dx   ,  ((1:m)-m/2-0.5)*dy      );
    CarteOut.grilleX=X;
    CarteOut.grilleY=Y;

    CarteOut.carte = griddata(CarteIn.grilleX,CarteIn.grilleY,CarteIn.carte,CarteOut.grilleX,CarteOut.grilleY);


    CarteOut.titre = [CarteOut.titre,'_Resize' , num2str(resol)];
%     CarteOut=updateProp(CarteOut);
end 