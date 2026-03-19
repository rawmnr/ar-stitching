function [carteRandom,carteRandom_HF_RMS,CarteRandomMM]=calculCarteRandom(TableData,Coord1,Coord2,dis_random)
        ResTP=sqrt(size(TableData,2));
        idxs=(sqrt(Coord1.^2+Coord2.^2)<dis_random);
        AllMaps=TableData(idxs,:);
        for i=1:size(AllMaps,1)
           temp=GenerateMapObject(reshape(AllMaps(i,:),ResTP,ResTP),1,632.8,'');
           temp=LegendreSub(temp,1:6,1:6);
           AllMaps(i,:)=temp.carte(:);
        end
        carteRandom = reshape(mean(AllMaps,'omitnan'),[ResTP,ResTP]);
        CarteRandomMM = reshape(std(AllMaps,'omitnan'),[ResTP,ResTP]);
        carteRandom=GenerateMapObject(carteRandom,1,632.8,'');
        carteRandom=LegendreSub(carteRandom,1:6,1:6);
        carteRandomHF=LegendreSub(carteRandom,1:36,1:36);
        carteRandom=carteRandom.carte;
        carteRandom_HF_RMS=std(reshape(carteRandomHF.carte,[1, size(carteRandomHF.carte,1)*size(carteRandomHF.carte,2)]),'omitnan');
    end