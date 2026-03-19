function [Nsave] = SSPbuilderrandom(Nssp,ratio,mapfile,method,radiusCour,resolutionCamera)
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
ssppsize = 1000; %diametre souspupilles mm 
ssppradius = floor(ssppsize/2);
% ratio = 0.85; 
onboundary = 2;


fid = fopen(mapfile);
m5map = read_opd(fid);
m5map = Resize(m5map,resolutionCamera);
mask =~isnan(m5map.carte);
dimmask = size(mask,1); 
P = mask2poly(mask); 
rayon = m5map.rayon; 
sizemap = m5map.largeur; 
lambda = m5map.lambda;
pixelsize = 2*rayon/sizemap;
Xe = P.X - ones(1,P.Length)*floor(dimmask/2);
Ye = P.Y - ones(1,P.Length)*floor(dimmask/2);
Xe = (Xe./floor(dimmask/2)).*rayon;
Ye = (Ye./floor(dimmask/2)).*rayon;

ssppres = floor(2*ssppradius/pixelsize);
% ssppres = resolutionCamera;
mastergrid = zeros(size(m5map.carte)); 

[randomfit,Z] = Zernike_Analysis(m5map.carte,0);
m5map.carte = m5map.carte - randomfit.carte_fit;

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

        fid1=fopen('position_sspp_random.txt', 'w');        
        for i=1:length(Xsave)
            sspname = ['Random' sprintf('%03i',i) '.Opd'];
            fprintf(fid1,'%s=%i;%i\n',sspname,-Xsave(i),Ysave(i));
        end
        
        %% plot 
        figure; 
        hold on;
        title('Random map measurement')
        p = plot(Xe,Ye,'k','LineWidth',2);
        plot(Xr,Yr,'r--')
        axis equal
        for i=1:Nssp
            if inratio(i)==1
                circle(Xp(i),Yp(i),ssppradius);
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
       fid1=fopen('position_sspp_random.txt', 'w');        
       for i=1:length(Xsave)
           sspname = ['Random' sprintf('%03i',i) '.Opd'];
           fprintf(fid1,'%s=%i;%i\n',sspname,-Xsave(i),Ysave(i));
       end
       %% plot 
        figure; 
        title('Random map measurement')
        hold on;
        p = plot(Xe,Ye,'k','LineWidth',2);
        axis equal
        for i=1:Nssp
          circle(Xsave(i),Ysave(i),500);
        end 
        hold off; 
        %% stastistics 
        
        
    case 'XY'    
        
end



%% Decoupage de la carte 

% pour decouper la carte on la translate de la coordonnee de la sous
% pupille et on fait une troncature cercle 

% troncature cercle de resize carte 
Nsave = size(Xsave,2)
Xsavep = floor(Xsave./pixelsize);
Ysavep = floor(Ysave./pixelsize);
for k=1:Nsave
    nom_SSPP=['Random' sprintf('%03i',k) '.Opd'];
    CarteSSP = troncatureCercle(resizeCarte(ShiftP(Ysavep(k),-Xsavep(k),m5map.carte), ssppres));
    fid = fopen(nom_SSPP,'w');
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
    h = plot(xunit, yunit,'LineWidth',0.1);
%     hold off
end

end

