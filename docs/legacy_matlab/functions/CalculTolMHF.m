function  [MeanMF,B,STDMFlnr]= CalculTolMHF(M,Meantol,STDtol,nc)
%Pour ce calcul, on est obligé de réorganiser les vecteurs MeanMF et STDMF. Dans le cas d'un Monte Carlo sous Warpp, le calcul donne la
 %valeur moyenne obtenue dans une colonne de la matrice sensibilité et l'écart-type dans la colonne d'après. Afin de faire correspondre le bon écart
 %type et la bonne valeur moyenne, il faut réagencer le vecteur STDMF pour fusionner ces deux colonnes
 MeanMF=abs(M'.*Meantol); 
 STDMF=abs(M'.*STDtol);
 for k=5:nc % was 1 and nc-5  permet de considerer les sensibilités et les stats 
	if STDtol(k,1:1)== 0 
     STDMF(k,:)=STDMF(k-1,:); % was k+1
     STDMF(k-1,:)=0; % was k+1 
    end
 end
 %Dans le cas de tolérances issues d'un calcul statistique, on suppose que
 %les erreurs MHF qui en découlent suivent une loi normale repliée.
 %On calcule ici le vecteur STDMFlnr pour lesquels on va appliquer la loi normale repliée. 
 STDMFlnr=MeanMF./(MeanMF+1e-20).*STDMF;
 STDMFuni=STDMF-STDMFlnr;
 %Dans le cas d'une sensibilité simple, on suppose que les erreurs MHF suivent une distribution uniforme. 
 %On calcule la valeur max B à partir de l'écart type. Comme l'on ne considère que des tolérences centrées, la valeur moyenne est toujours
 %supposée nulle
 B=3^0.5*STDMFuni;

