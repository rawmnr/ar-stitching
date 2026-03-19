% clear all
global Mean STD;
% cd 'S:\dfs\DSOD\DPRG\REOSC\51-ELTM2\07-DFC\05_dedicated test means\02- Banc ITB M2\20- Budget derreur\WFE\'
% cd 'R:\51-ELTM2\07-DFC\05_dedicated test means\02- Banc ITB M2\20- Budget derreur\WFE\'
% [num,txt,tab]=xlsread('bilan d-erreur banc M2_20200615_matrice_asbuild.xlsx','Matrice sensibilité');
[num,txt,tab]=xlsread('ErrorBudget_EXP1000_FLAT.xlsx','Sheet1');
% T = readtable('ErrorBudget_EXP1000_FLAT.xlsx');
%Définition de la matrice de sensibilité M pour les coefficients de Zernike
%de Z4 à Z37. La première ligne correspond aux sensibilités de Z4.
 nc=15; %nombre de colonnes du bilan d'erreur contenant les différentes sensibilités
 M=num(7:41,1:nc); 
 %Définition de la matrice de sensibilité M2=M² permettant de calculer les écarts type des fonctions de répartition des Zi.
 M2=M.*M;
 Meantol=num(2,1:nc)';
 STDtol=num(3,1:nc)';
 Coeff = num(4,1:nc)';
 
 % application du coefficient 
 Meantol = Coeff.*Meantol;
 STDtol = Coeff.*STDtol;
 
 STD2=STDtol.^2;
 Meanzi=M*Meantol; % La matrice de sensibilité est calculée directement en "nm rms signés"
 STDzi=(M2*STD2).^0.5;
 
 
 %Calcul moyenne et écart type de la fonction de répartition Z5,Z6
 %q = le nombre d'événement calculé pour trouver la fonction de répartition
 %epsilonZ calcule le quantile à 95% => donne l'erreur max dans 95% des cas
 q=10000;
 
 [Zrms,meanzrms,stdzrms,epsilonZ]=FrepartitionZrms(Meanzi,STDzi,2:34,q,'norm');epsilonZ
 
 [Z4rms,meanz4rms,stdz4rms,epsilonZ4]=FrepartitionZrms(Meanzi,STDzi,1,q,'norm');epsilonZ4
 
 [Z56rms,meanz56rms,stdz56rms,epsilonZ56]=FrepartitionZrms(Meanzi,STDzi,[2,3],q,'norm');epsilonZ56;
 
 [Z78rms,meanz78rms,stdz78rms,epsilonZ78]=FrepartitionZrms(Meanzi,STDzi,[4,5],q,'norm');epsilonZ78;
 
 [Z9rms,meanz9rms,stdz9rms,epsilonZ9]=FrepartitionZrms(Meanzi,STDzi,6,q,'norm');epsilonZ9;
 
 [Z1011rms,meanz1011rms,stdz1011rms,epsilonZ1011]=FrepartitionZrms(Meanzi,STDzi,[7,8],q,'norm');epsilonZ1011;
 
 %Calcul moyenne et écart type de la fonction de répartition Z16
 [Z16rms,meanz16rms,stdz16rms,epsilonZ16]=FrepartitionZrms(Meanzi,STDzi,[13],q,'norm');
 
 %Calcul moyenne et écart type de la fonction de répartition Z12, Z13, Z17, Z18, Z19, Z20
 [Z1220rms,meanz1220rms,stdz1220rms,epsilonZ1220]=FrepartitionZrms(Meanzi,STDzi,[9,10,14,15,16,17],q,'norm');epsilonZ1220;
 
 [MeanMF,B,STDMFlnr]= CalculTolMHF(M(35,:),Meantol,STDtol,nc);
 [MFrms,meanMFrms,stdMFrms,epsilonMF]=FrepartitionZrms_MF(MeanMF,B,STDMFlnr,q,'unif');epsilonMF
 
 
figure
subplot(4,2,1)
cdfplot(Zrms);  title(['Z error @ 95% : ',num2str(epsilonZ(2),3),' nm rms']);xlabel('nm rms');grid; axis equal
subplot(4,2,2)
cdfplot(MFrms);  title(['MF error @ 95% : ',num2str(epsilonMF(2),3),' nm rms']);xlabel('nm rms');grid; axis equal
subplot(4,2,3)
cdfplot(Z4rms);  title(['Z4 error @ 95% : ',num2str(epsilonZ4(2),3),' nm rms']);xlabel('nm rms');
grid;ylim([0 1]);
subplot(4,2,4)
cdfplot(Z56rms);  title(['Z5 Z6 error @ 95% : ',num2str(epsilonZ56(2),3),' nm rms']);xlabel('nm rms');grid; axis equal
subplot(4,2,5)
cdfplot(Z9rms);  title(['Z9 error @ 95% : ',num2str(epsilonZ9(2),3),' nm rms']);xlabel('nm rms');grid; axis equal 
subplot(4,2,6)
cdfplot(Z1011rms); title(['Z10 Z11 error @ 95% : ',num2str(epsilonZ1011(2),3),' nm rms']);xlabel('nm rms');grid; axis equal
subplot(4,2,7)
cdfplot(Z16rms); title(['Z16 error @ 95% : ',num2str(epsilonZ16(2),3),' nm rms']);xlabel('nm rms');grid; axis equal
subplot(4,2,8)
cdfplot(Z1220rms); title(['Z12-13-17-18-19-20 error @ 95% : ',num2str(epsilonZ1220(2),3),' nm rms']);xlabel('nm rms');grid; axis equal 

perf_summary(1,:) = {'Aberration' 'Epsilon 50%' 'Epsilon 95%' 'Epsilon 99.7%'};
perf_summary(2:7,1) = {'Z56'; 'Z9'; 'Z1011'; 'Z1220'; 'Z16'; 'MF'};
% perf_summary(2:10,2) = num2cell(Spec)';
perf_summary(2,2:4) = num2cell([epsilonZ56]);
perf_summary(3,2:4) = num2cell([epsilonZ9]);
perf_summary(4,2:4) = num2cell([epsilonZ1011]);
perf_summary(5,2:4) = num2cell([epsilonZ1220]);
perf_summary(6,2:4) = num2cell([epsilonZ16]);
perf_summary(7,2:4) = num2cell([epsilonMF]);
 
 
 
 
 
%  %Calcul moyenne et écart type de la fonction de répartition Z9
%  indice=[6];
%  [Z9rms,meanz9rms,stdz9rms,epsilonZ9]=FrepartitionZrms(Meanzi,STDzi,indice,q,'norm');
%  %La fonction de répartition de Z16rms peut aussi facilement se calculer directement
%  %Elle vaut 0 entre -l'infini et 0 et une gaussienne entre 0 et +l'infini.
%  %Cela permet de faire une vérification croisée du calcul
% %  Mean=Meanzi(6:6);STD=STDzi(6:6);
% %  x9=0:0.1:4*STD;
% %  %y9=normpdf(x9,Mean,STD)*2;
% % %  lnr9=loinormalerepliee(x9)+0.95;
% %  lnr9=loinormalerepliee(x9,0.95)+0.95;
% %  %ddplnr9=Ddploinormalerepliee(x9);
% % %  epsilonZ9=fsolve(@loinormalerepliee,Mean)
% %  epsilonZ9=fsolve(@(x) loinormalerepliee(x,0.95),Mean)
% %  meanz9rms=2*STD/(2*pi)^0.5*exp(-(Mean/STD)^2/2)-Mean*erf(-Mean/STD/2^0.5);
% %  stdz9rms=(Mean^2+STD^2-meanz9rms^2)^0.5;
% %  %erreurmean=(meanz9rms2-meanz9rms)/(meanz9rms2+meanz9rms2)
% %  %erreurstd=(stdz9rms2-stdz9rms)/(stdz9rms2+stdz9rms)
% %  epsilon50_Z9=fsolve(@(x) loinormalerepliee(x,0.5),Mean); % perf Z9 à 50%
% %  epsilon99_Z9=fsolve(@(x) loinormalerepliee(x,0.997),Mean); % perf Z9 à 99.7%
%  
%  %Calcul moyenne et écart type de la fonction de répartition Z10,Z11
%  [Z1011rms,meanz1011rms,stdz1011rms,epsilonZ1011]=FrepartitionZrms(Meanzi,STDzi,[7,8],q,'norm');epsilonZ1011
%  
%  %Calcul moyenne et écart type de la fonction de répartition Z16
%  [Z16rms,meanz16rms,stdz16rms,epsilonZ16]=FrepartitionZrms(Meanzi,STDzi,[13],q,'norm');
%  
%  %Calcul moyenne et écart type de la fonction de répartition Z12, Z13, Z17, Z18, Z19, Z20
%  [Z1220rms,meanz1220rms,stdz1220rms,epsilonZ1220]=FrepartitionZrms(Meanzi,STDzi,[9,10,14,15,16,17],q,'norm');epsilonZ1220
%  
%  %Calcul moyenne et écart type de la fonction de répartition MF,HF,VHF200 et VHF60
%  %%%%%%%%%%%%%%%%%%%%traitement pour les MHF

%  [MeanMF,B,STDMFlnr]= CalculTolMHF(M(36,:),Meantol,STDtol,nc);
%  [HFrms,meanHFrms,stdHFrms,epsilonHF]=FrepartitionZrms_MF(MeanMF,B,STDMFlnr,q/10,'unif');
%  [MeanMF,B,STDMFlnr]= CalculTolMHF(M(37,:),Meantol,STDtol,nc);
%  [VHF200rms,meanVHF200rms,stdVHF200rms,epsilonVHF200]=FrepartitionZrms_MF(MeanMF,B,STDMFlnr,q/10,'unif');
%  [MeanMF,B,STDMFlnr]= CalculTolMHF(M(38,:),Meantol,STDtol,nc);
%  [VHF60rms,meanVHF60rms,stdVHF60rms,epsilonVHF60]=FrepartitionZrms_MF(MeanMF,B,STDMFlnr,q/10,'unif');
% 
% 
%  %Affichage des différentes fonctions de répartition
%  figure;
%  subplot(3,3,1);
%  %x = linspace(0,epsilonZ56);
%  %plot(x,cdf(Z56,x)); title(['Z5 Z6 error @ 95% : ',num2str(epsilonZ56(2),3),' nm rms']); xlabel('nm rms');grid;
%  cdfplot(Z56rms); title(['Z5 Z6 error @ 95% : ',num2str(epsilonZ56(2),3),' nm rms']); xlabel('nm rms');grid;
%  subplot(3,3,4);
%  cdfplot(Z9rms);  title(['Z9 error @ 95% : ',num2str(epsilonZ9(2),3),' nm rms']);xlabel('nm rms');grid;
%  %hold;
%  %plot(x9,lnr9); title(['Z9 error @ 95% : ',num2str(epsilonZ9,3),' nm rms']);xlabel('nm rms');grid;
%  subplot(3,3,2);
%  cdfplot(Z1011rms); title(['Z10 Z11 error @ 95% : ',num2str(epsilonZ1011(2),3),' nm rms']);xlabel('nm rms');
%  subplot(3,3,5);
%  cdfplot(Z16rms); title(['Z16 error @ 95% : ',num2str(epsilonZ16(2),3),' nm rms']);xlabel('nm rms');grid;
%  %hold;
%  %plot(x16,lnr16); title(['Z16 error @ 95% : ',num2str(epsilonZ16,3),' nm rms']);xlabel('nm rms');grid;
%  subplot(3,3,7);
%  cdfplot(Z1220rms); title(['Z12-13-17-18-19-20 error @ 95% : ',num2str(epsilonZ1220(2),3),' nm rms']);xlabel('nm rms');
%  subplot(3,3,3);
%  cdfplot(MFrms); title(['MF error @ 95% : ',num2str(epsilonMF(2),3),' nm rms']);xlabel('nm rms');
%  subplot(3,3,6);
%  cdfplot(HFrms); title(['HF error @ 95% : ',num2str(epsilonHF(2),3),' nm rms']);xlabel('nm rms');
%  subplot(3,3,8);
%  cdfplot(VHF200rms); title(['VHF200 error @ 95% : ',num2str(epsilonVHF200(2),3),' nm rms']);xlabel('nm rms');
%  subplot(3,3,9);
%  cdfplot(VHF60rms); title(['VHF60 error @ 95% : ',num2str(epsilonVHF60(2),3),' nm rms']);xlabel('nm rms');
% 
% 
%  %%%%%%%%%%%%%%%%%%%%Tableau résumant l'ensemble des performances
%  % en lignes: Z5-Z6, Z9, Z10-Z11, Z16, Z12-Z20, MF, HF, VHF200, VHF60
%  % en colonnes : perf à 50%, perf à 95%, perf à 99.7%, Critère mesure
% 

%  perf_summary(8,3:5) = num2cell([epsilonHF]);
%  perf_summary(9,3:5) = num2cell([epsilonVHF200]);
%  perf_summary(10,3:5) = num2cell([epsilonVHF60]);
%  perf_summary
 