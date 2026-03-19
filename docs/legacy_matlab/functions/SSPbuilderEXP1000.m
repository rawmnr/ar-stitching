function [Nsave] = SSPbuilderEXP1000(Nssp,ratio,mapfile,method,radiusCour,zuexp1000)
% SSP builder sert a decouper une carte selon certain 

% distribution sunflower
% The algorithm places n points, of which the kth point is put 
% at distance sqrt(k-1/2) from the boundary (index begins with k=1), 
% and with polar angle 2*pi*k/phi^2 where phi is the golden ratio. 
% Exception: the last alpha*sqrt(n) points are placed on the outer boundary of the circle, 
% and the polar radius of other points is scaled to account for that. 
% This computation of the polar radius is done in the function radius.

% Prochaines updates
% Differentes distributions 
% Statistiques de sortie


% Nssp = 50; % nombre de souspupilles initial

ssppradius = 500;
% ratio = 0.85; 
onboundary = 0;


fid = fopen(mapfile);
m5map = read_opd(fid);
mask =~isnan(m5map.carte);
dimmask = size(mask,1); 
P = mask2poly(mask); 
rayon = m5map.rayon ;
sizemap = m5map.largeur ;
lambda = m5map.lambda;
pixelsize = 2*rayon/sizemap;
Xe = P.X - ones(1,P.Length)*floor(dimmask/2);
Ye = P.Y - ones(1,P.Length)*floor(dimmask/2);
Xe = (Xe./floor(dimmask/2)).*rayon;
Ye = (Ye./floor(dimmask/2)).*rayon;




ssppres = floor(2*ssppradius/pixelsize);
ssppsize = 0.5*ssppres*zuexp1000/(2*ssppradius);

circlesize = sizemap*zuexp1000/(2*ssppradius);

mastergrid = zeros(size(m5map.carte)); 

%% definition d'un masque ratio
Xr = Xe.*ratio;
Yr = Ye.*ratio;

%%
switch method
    case 'sunflower' 
        [Xs,Ys] = sunflower(Nssp,onboundary,1);
        Xp = Xs.*rayon; % coordonnees des centres 
        Yp = Ys.*rayon; % coordonnés des centres 

        in = inpolygon(Xp,Yp,Xe,Ye);
        inratio = inpolygon(Xp,Yp,Xr,Yr); 

        Xsave = Xp(inratio);
        Ysave = Yp(inratio); 

        fid1=fopen('position_sspp.txt', 'w');        
        for i=1:length(Xsave)
            sspname = ['C' sprintf('%03i',i) 'AP_TP.Opd'];
            fprintf(fid1,'%s=%i;%i\n',sspname,-Xsave(i),Ysave(i));
        end
        
        %% plot 
        figure; 
        hold on;
        plot(Xe,Ye,'b')
        plot(Xr,Yr,'r--')
        for i=1:Nssp
            if inratio(i)==1
                circle(Xp(i),Yp(i),circlesize);
            end
        end 

        plot(Xp(inratio),Yp(inratio),'r+')
        hold off; 
        
           
    case 'couronne'
        Xsave = zeros(1,Nssp);
        Ysave = zeros(1,Nssp);
        % center subaperture
       Xsave(1) = 0;
       Ysave(1) = 0;
       % Couronne 1
       nCouronne = Nssp - 1;
       stepAngle = 360/nCouronne;
       angle = 0;
       for ipup=1:nCouronne
           Xsave(ipup+1) = radiusCour*sin(angle*pi()/180);
           Ysave(ipup+1) = radiusCour*cos(angle*pi()/180);
           angle = angle + stepAngle; 
       end
       fid1=fopen('position_sspp.txt', 'w');        
       for i=1:length(Xsave)
           sspname = ['C' sprintf('%03i',i) 'AP_TP.Opd'];
           fprintf(fid1,'%s=%i;%i\n',sspname,-Xsave(i),Ysave(i));
       end
       %% plot 
        figure; 
        hold on;
        plot(Xe,Ye,'b')
        axis equal
%         plot(Xr,Yr,'r--')
        for i=1:Nssp
          circle(Xsave(i),Ysave(i),circlesize);
        end 

%         plot(Xp(inratio),Yp(inratio),'r+')
        hold off; 
        %% stastistics 
        coordmatrix = [Xsave(:) Ysave(:)];
        distance = pdist([coordmatrix(1,:);coordmatrix(2,:)],'euclidean');
        Ratio = overlap(distance,circlesize);
    case 'XY'    
        
end



%% Decoupage de la carte 

% pour decouper la carte on la translate de la coordonnee de la sous
% pupille et on fait une troncature cercle 

% troncature cercle de resize carte 
Nsave = size(Xsave,2)
% Xsavep = floor(Xsave./pixelsize);
Xsavep = Xsave./pixelsize;
% Ysavep = floor(Ysave./pixelsize);
Ysavep = Ysave./pixelsize;


for k=1:Nsave
    nom_SSPP=['C' sprintf('%03i',k) 'AP_TP.Opd'];
%     CarteSSP = troncatureCercle(resizeCarte(ShiftP(Ysavep(k),-Xsavep(k),m5map.carte), ssppres));
    CarteSSP = troncatureCercle(resizeCarte(reinterpADO(-Xsavep(k),Ysavep(k),m5map.carte), ssppres),ssppsize);
    fid = fopen(nom_SSPP,'w+');
    write_opd(CarteSSP*1e-9,nom_SSPP,ssppradius,lambda,fid);
end 



function [X,Y] = sunflower(n, alpha,rayon)   %  example: n=500, alpha=2
%     clf
%     hold on
    X = [];
    Y = [];
    b = round(alpha*sqrt(n));      % number of boundary points
    phi = (sqrt(5)+1)/2;           % golden ratio
    for k=1:n
        r = radius(k,n,b,rayon);
        theta = 2*pi*k/phi^2;
        X(k) = r*cos(theta);
        Y(k) = r*sin(theta);
%         plot(r*cos(theta), r*sin(theta), 'r*');
    end
end

function r = radius(k,n,b,rayon)
    if k>n-b
%         r = 1;            % put on the boundary
        r = rayon; 
    else
        r = sqrt(k-1/2)/sqrt(n-(b+1)/2);     % apply square root
    end
end

function h = circle(x,y,r)
%     hold on
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit);
    axis equal
%     hold off
end

end

