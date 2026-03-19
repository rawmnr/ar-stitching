function res = rndlnrep(Mean,STD,rd)
%Calcule l'inverse de la fonction de répartition pour la loi normale
%repliée et crée un vecteur aléatoire suivant la loi normale repliée
%de moyenne Mean et d'écart type STD
n=size(Mean);
for i=1:n
    M=Mean(i);
    S=STD(i);
    r=rd(i);
    if ~M==abs(M)
        msgbox('Danger : Moyenne négative','error')
    end
    if ~S==abs(S)
        msgbox('Danger : STD négatif','error')
    end    
    if S==0
        res(i)=0;
    else
        myfun = @(x,M,S,r) loinormalerepliee(x,M,S)-r;
        fun = @(x) myfun(x,M,S,r);
        x0=[0 M+10*S];
        res(i)=fzero(fun,x0);
    end    
end

