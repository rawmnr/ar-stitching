function write_opd(Map,fullpath)
%A commenter
fid = -1; % Initialize fid to an invalid value for error handling
try
    %
    fid = fopen(fullpath,'w');

    %% Ecriture de l'en t�te du fichier  
    %Format titre
    if length(Map.titre)<20
        Map.titre=pad(Map.titre,20);
    elseif length(Map.titre)>20
        Map.titre=Map.titre(1:20);
    end
    Map.titre = clean_txt(Map.titre); %Gestion des caract�res speciaux
    
    if isstring(Map.titre)
        Map.titre = char(Map.titre);
    end

    fwrite(fid,Map.titre(1:20),'char*1'); %Nom du fichier : 20 caract�res
    fwrite(fid,[' ' Map.date ' '],'char'); %date et heure : 19 caract�res et respect de la nomenclature

    %Format Commentaires
    Map.commentaires = clean_txt(Map.commentaires);
    if length(Map.commentaires)<104
        Map.commentaires=pad(Map.commentaires,104);
    elseif length(Map.commentaires)>104
        Map.commentaires=Map.commentaires(1:104);
    end
    fwrite(fid,Map.commentaires,'char'); %Commentaires : 104 caract�res
    fwrite(fid,'O','char'); %Champ obligatoire pour compatibilit� avec WISP
    if isfield('logiciel',Map)
        fwrite(fid,pad(Map.logiciel,16),'char'); %Origine de la carte (16 carac)
    else
        fwrite(fid,pad('inconnu',16),'char'); %Origine de la carte (16 carac)
    end
    
    fwrite(fid,Map.lambda,'float32'); %Longueur d'onde de la carte : 1 r�el
    fwrite(fid,0,'float32'); %Wedge Factor
    fwrite(fid,double(Map.rayon),'float32'); %Rayon de la pupille
    a=single(ones(1,3));fwrite(fid,a,'float32');
    fwrite(fid,0,'float32');
    a=single(ones(1,13));fwrite(fid,a,'float32');


    S=size(Map.carte);

    fwrite(fid,52,'int16');
    fwrite(fid,S(1),'int16');
    fwrite(fid,S(2),'int16');
    fwrite(fid,[1 1 1 1 Map.flag],'int16'); %Mettre [1 1 1 1 2] pour identifier la carte comme MSE, [1 1 1 1 1] pour WFE
    fwrite(fid,1,'int16');
    fwrite(fid,[1 1 1 1 1 1 1 1 1 1 1],'int16'); 

    %% Ecriture des donn�es de la carte
    carteMod=Map.carte/Map.lambda;
    carteMod(isnan(carteMod))=Inf; % On convertit les nan en Inf (point sans donn�es dans WaRPP)

    fwrite(fid,carteMod,'float32'); % On �crit dans le fichier

    fclose(fid); % On ferme le fichier

catch error
    disp(error)
    disp('Un erreur s''est produite dans l''execution de write_opd.')
    if fid ~= -1 % Only attempt to close if it was successfully opened
        fclose(fid);
    end
end

function txt = clean_txt(txt) %Fonction de gestion des caracteres speciaux
    specials = {'"', '�', '�', '�', '�', '�'};
    replaces = {' ', 'e', 'e', 'a', 'o', 'u'};
    for i = 1:length(specials)
        txt = strrep(txt, specials{i}, replaces{i});
    end
end
end