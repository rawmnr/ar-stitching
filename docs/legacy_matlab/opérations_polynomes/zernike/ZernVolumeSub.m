function CPCOut=ZernVolumeSub(CPCIn,idx,OrdreAnalyse,RNorm,type)


%%
[a,Z,~,~,~,~,pup]=ZernikeDecomposeMap(CPCIn,OrdreAnalyse,RNorm,type);
%
for j=1:length(idx)
    CarteComp{j}=CPCIn.carte;CarteComp{j}(pup)=Z(:,idx(j));
end
%
MFStr="MF=@(vec)(VMMat(CPCIn.carte";
for j=1:length(idx)
    MFStr=MFStr+"+vec("+num2str(j)+")*CarteComp{"+num2str(j)+"}";
end
MFStr=MFStr+",CPCIn.rayon));";
eval(MFStr);
vecopt=fmincon(MF,zeros(length(idx),1));
%
%Version avec contrainte
%     lb=[];ub=[];
%     vecopt=fmincon(MF,zeros(length(idx),1),[],[],[],[],lb,ub,@(vec)(CS(vec,CarteComp,Threshold)));
%
CPCOut=CPCIn;
for j=1:length(idx)
    CPCOut.carte=CPCOut.carte+vecopt(j)*CarteComp{j};
end
%
end

function [c,ceq]=CS(vec,CarteComp,threshold)
    c=std(0+vec(4)*CarteComp{4}(:)+vec(5)*CarteComp{5}(:)+vec(6)*CarteComp{6}(:)+...
        vec(7)*CarteComp{7}(:)+vec(8)*CarteComp{8}(:),'omitnan')-threshold;
    ceq=0;
end