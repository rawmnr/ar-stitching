function write_opd(carte,fileName,Rpupille,lambda,fid)
% Auteur : Mehdi AJROUD. Date de création : 23/10/2014
% v2.0 du 24/02/2017
%
% Syntaxe :
% write_opd(carte,fileName,Rpupille,lambda,fid)
%
% Entrées :-carte : matrice nxn que l'on veut enregistrer en carte. Les
%                   altitudes doivent être en m.
%          -fileName : string donnant le nom du fichier
%          -Rpupille : Double donnant le rayon de pupille en mm
%          -lambda : Longueur d'onde de la carte
%             (Argument facultatif, par défaut lambda=632.8nm)
%
%          -fid :  file identifier généré avec la fonction matlab fopen
%                    ex:         fileName='MaCarte';
%                                path='S:\DSOD\DPRG\REOSC\03_FABRICATION\20_MSO\Commun\99_SAS_MSO';
%                                fid = fopen([pathName fileName],'w');
%             (Argument facultatif, s'il n'est pas précisé Matlab ouvre une
%              fenêtre de dialogue permettant de choisir l'emplacement)
%             (Si cet argument est précisé, lambda doit l'être aussi)
%
% ------------ Exemple d'utilisation : ------------------------------------
%              x=linspace(-1,1,100);
%              y=linspace(-1,1,100);
%              [X,Y]=ndgrid(x,y);
%              Z=X.^2+Y.^2;
%              fileName='MaCarte.Opd';
%              pathName='';
%              fid = fopen([pathName fileName],'w');
%              write_opd(Z,fileName,100,1063,fid); %100 est le Rpupille et
%              1063 est la longueur d'onde en nm
%              
%              
%
% Sorties : Un fichier .opd est créé à l'endroit choisi
% on inclut un try catch pour éviter de perdre le fid en cas de bug
try
switch nargin
    case 3
        lambda=632.8;
        
        [fileName,pathName] = uiputfile({'*.opd';'*.Opd';'*.OPD'},'Enregistrer une carte');
        fid = fopen([pathName fileName],'w');
    case 4
        [fileName,pathName] = uiputfile({'*.opd';'*.Opd';'*.OPD'},'Enregistrer une carte');
        fid = fopen([pathName fileName],'w');
end


nomOriginalFichier=fileName;

% On enlève le '.opd' s'il y en a un.
if strcmp(fileName(length(fileName)-3:length(fileName)),'.opd')
    fileName(length(fileName)-3:length(fileName))='    ';
end

% On s'assure que le nom de la carte a la bonne taille
if length(fileName)>=20
nomCarte=fileName(1:20);
else

nomCarte=fileName;  
    for i=1:(20-length(fileName))
        nomCarte=[nomCarte ' '];
    end   
end

carte=carte*1e9/lambda; %Cette ligne permet de gérer le fait que WaRPP travail en lambda et pas en m

V='                    '; %Matrice de caractères comportant 20 espaces
B=[V V V V V '    ']; % Matrice de caractères comportant 104 espaces

%% Récupération de l'heure et mise au bon format :

dateHeure=round(clock); %dateHeure=[year,month,day,hour,minute,seconds]

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

%% Ecriture de l'en tête du fichier
fwrite(fid,nomCarte,'char'); %Nom du fichier : 20 caractères
fwrite(fid,[' ' day(1:2) '/' month(1:2) '/' year(3:4) ' ' hour(1:2) ':' minute(1:2) ':' seconds(1:2) ' '],'char'); %date et heure : 19 caractères et respect de la nomenclature

fwrite(fid,B,'char'); %Commentaires : 104 caractères
fwrite(fid,'O','char'); %Champ obligatoire pour compatibilité avec WISP
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
fwrite(fid,[1 1 1 1 2],'int16'); %Mettre [1 1 1 1 2] pour identifier la carte comme MSE, [1 1 1 1 1] pour WFE
fwrite(fid,1,'int16');
fwrite(fid,[1 1 1 1 1 1 1 1 1 1 1],'int16'); 

%% Ecriture des données de la carte

 carteMod=flip(carte); %On flip X pour compenser, dans l'écriture du fichier,
% % le fait que MatLab et WaRPP n'ont pas les mêmes références d'axes pour Y.
% % Grâce à cette manip elle est affichée dans WaRPP comme elle est affichée
% % dans MatLab

carteMod(isnan(carteMod))=Inf; % On convertit les nan en Inf (point sans données dans WaRPP)

fwrite(fid,carteMod,'float32'); % On écrit dans le fichier

fclose(fid); % On ferme le fichier

% winopen([path nomOriginalFichier]); %Ouvre le fichier enregistré dans
% WaRPP, à décommenter si l'on souhaite ouvrir la carte à la fin de l'enregistrement
% Génère une erreur si Windows n'est pas configuré pour ouvrir
% automatiquement les .opd sur WaRPP.
catch error
    disp(error)
    disp('Un erreur s''est produite dans l''execution de write_opd.')
    fclose(fid); % On ferme le fichier
end
end