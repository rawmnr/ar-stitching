%%% Cette feuille matlab contient un code permettant de générer
%%% aléatoirement des cartes HF similaires à une collection de miroirs
%%% définie. Les miroir à définir doivent être analysés (premier
%%% paragraphe), les chemins indiqués sont ceux du dossier contenant la
%%% feuille Matlab.
%%% Puis les cartes sont générées dans le paragraphe 2 (modifier le chemin
%%% si nécessaire. Elles sont semblables aux cartes du pool exemple.
%%% La construction est la suivante : toutes les cartes du pool sont
%%% passées dans l'espace de fourier. Les parties réelles et imaginaires
%%% sont séparées. Pour la carte de la partie réelle par exemple, la carte 
%%% contenant la moyenne pixel par pixel d'une part et l'écart type pixel 
%%% par pixel d'autre part sont calculées.
%%% Pour chaque pixel, on a donc une variable aléatoire gaussienne, de moyenne et
%%% d'écart-type déterminés ce qui permet avec cette "carte" de variable
%%% aléatoire de tirer une carte dans l'espace de fourier. Une fois la
%%% carte réelle et la carte imaginaire tirées dans l'espace de fourier, on
%%% repasse dans l'espace réel pour obtenir la carte aléatoire voulue.
%%% Ce code a été fait pour tester l'algo de NEOSS dans le cadre de
%%% l'ELT-M1 par conséquent, il créé des couples de cartes CS/TP comme pour
%%% le banc TALISMANN del'ELT-M1.
%%% Fonction créée pour les besoins de NEOSS :
%%%
%%%
%%%
%%%
%%%
%%%

function CarteHFGenere=GenerateurCartesHFFiz(HVHFLevel,resolution, rayon, main_folder)

repertoire_source=[main_folder '\EXP1000_sources'];
mirroirs=[1];
RealPart=zeros(614,614);
ImPart=zeros(614,614);
RealPart_STD=zeros(614,614);
ImPart_STD=zeros(614,614);
for i=mirroirs
    %Definition des cartes
    switch i
        case 1
            fid=fopen([repertoire_source '\PFZ-1000-170914-614.Opd']);
        case 2
            fid=fopen([repertoire_source '\FLAT1500_1.opd']);
        case 3
            fid=fopen([repertoire_source '\FLAT1500_2.opd']);
        case 4
            fid=fopen([repertoire_source '\FlatN1500_1_PH99_CPC.Opd']);
        case 5
            fid=fopen([repertoire_source '\FLAT2_N1500_CARTE PIECE_NEOSS_ZU.Opd']);
        case 6
            fid=fopen([repertoire_source '\S3.opd']);
        case 7
            fid=fopen([repertoire_source '\S4.opd']);
        case 8
            fid=fopen([repertoire_source '\carte_Instrument.Opd']);
        case 9
            fid=fopen([repertoire_source '\ELTM4_308.opd']);
        case 10
            fid=fopen([repertoire_source '\ELTM4_312.opd']);
        case 11
            fid=fopen([repertoire_source '\ELTM4_313.opd']);
    end
   
map=read_opd(fid);
map=map.carte;
%Retirer le piston, très fluctuant d'une carte à l'autre
map(isnan(map))=0;
map=map-mean(mean(map(~isnan(map))));
A=fftshift(fft2(map));
RealPart=RealPart+real(A);
ImPart=ImPart+imag(A);
fclose all;
end;
RealPart=RealPart/length(mirroirs);
ImPart=ImPart/length(mirroirs);
for i=mirroirs
    %Definition des cartes
    switch i
        case 1
            fid=fopen([repertoire_source '\PFZ-1000-170914-614.Opd']);
        case 2
            fid=fopen([repertoire_source '\FLAT1500_1.opd']);
        case 3
            fid=fopen([repertoire_source '\FLAT1500_2.opd']);
        case 4
            fid=fopen([repertoire_source '\FlatN1500_1_PH99_CPC.Opd']);
        case 5
            fid=fopen([repertoire_source '\FLAT2_N1500_CARTE PIECE_NEOSS_ZU.Opd']);
        case 6
            fid=fopen([repertoire_source '\S3.opd']);
        case 7
            fid=fopen([repertoire_source '\S4.opd']);
        case 8
            fid=fopen([repertoire_source '\carte_Instrument.Opd']);
        case 9
            fid=fopen([repertoire_source '\ELTM4_308.opd']);
        case 10
            fid=fopen([repertoire_source '\ELTM4_312.opd']);
        case 11
            fid=fopen([repertoire_source '\ELTM4_313.opd']);
    end
map=read_opd(fid);
map=map.carte;


%Retirer le piston, très fluctuant d'une carte à l'autre
map(isnan(map))=0;
map=map-mean(mean(map(~isnan(map))));
A=fftshift(fft2(map));
RealPart_STD=RealPart_STD+(real(A)-RealPart).^2;
ImPart_STD=ImPart_STD+(imag(A)-ImPart).^2;
fclose all;
end;
RealPart_STD=sqrt(RealPart_STD/length(mirroirs));
ImPart_STD=sqrt(ImPart_STD/length(mirroirs));

%Generer la carte aléatoire
Carte_Fourier=zeros(resolution,resolution);
RealPart_Bis = resizeCarte(RealPart,resolution);RealPart_Bis(isnan(RealPart_Bis))=0;
ImPart_Bis = resizeCarte(ImPart,resolution);ImPart_Bis(isnan(ImPart_Bis))=0;
RealPart_STD_Bis = resizeCarte(RealPart_STD, resolution);RealPart_STD_Bis(isnan(RealPart_STD_Bis))=0;
ImPart_STD_Bis = resizeCarte(ImPart_STD, resolution);ImPart_STD_Bis(isnan(ImPart_STD_Bis))=0;
Carte_Fourier = RealPart_Bis+1i*ImPart_Bis+(RealPart_STD_Bis+1i*ImPart_STD_Bis).*randn(resolution);
Carte_reelle=real(ifft2(fftshift(Carte_Fourier)));
CarteHFGenere=(Carte_reelle);
[a,b]=size(CarteHFGenere);
A=reshape(CarteHFGenere,1,a*b);
A=CarteHFGenere(~isnan(CarteHFGenere));
c=sqrt(sum(sum((A-mean(A)).^2))/length(A));
CarteHFGenere=(HVHFLevel/c)*CarteHFGenere;
end