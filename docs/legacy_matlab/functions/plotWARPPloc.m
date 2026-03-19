
function []=plotWARPPloc(Carte, wl, radius,figtitle,figcomment,type,logo,scale)
% chose MSE by default
if ~exist('type','var') || ~strcmp(type,'WFE')
    type='MSE';
end
% chose no logo by default
if ~exist('logo','var')
    logo = 0;
end
% chose no logo by default
if ~exist('scale','var')
%     Maxim=max(max(Carte));
%     Maxim=roundn(Maxim,numel(num2str(ceil(Maxim)))-2);
%     Minim=min(min(Carte));
%     Minim=1.2*roundn(Minim,numel(num2str(ceil(Minim)))-2);
%     dT=(Maxim-Minim)/4;
%     caxis([Minim Maxim]);
    caxis('auto')
    scaling = 0;
else
    Minim=scale(1);
    Maxim=scale(2);
    scaling = 1;
end

% Carte=wl*Carte;
% Plot the map
[sz1,sz2]=size(Carte);
fig=figure(gcf);
S1=subplot(4,5,[2:4,7:9,12:14]);
if scaling 
    imagesc(Carte,scale);
else
    imagesc(Carte);
end

A=get(S1,'Position');
cb=colorbar('westoutside');
set(S1,'Position',[A(1)+0.1 A(2) A(3)-0.1 A(4)]);
S1.XAxisLocation = 'bottom';
S1.XTick=[];S1.YTick=[];ylabel(cb,[type,' [nm]'])

radiusDisp=floor(radius);

% Display the bottom plot
S2=subplot(4,5,[17:19]);
S2.Position = [A(1)+0.1 0.125 A(3)-0.1 0.16];
dx=2*radius/sz1;
xs=((1:sz1)-sz1/2-0.5)*dx;
plot(xs,Carte(floor(sz1/2),:));
xlim([-radius,radius]);
S2.XTick = -radiusDisp:radiusDisp/2:radiusDisp;
ylabel(S2,[type,' [nm]'])
xlabel(S2,'X [mm]')

%Display the right plot
S3=subplot(4,5,5);
plot(flip(Carte(:,floor(sz2/2))),xs);
S3.Position = [0.77 A(2) 0.12 A(4)];
ylim([-radius,radius]);
S3.YTick = -radiusDisp:radiusDisp/2:radiusDisp;
S3.YAxisLocation = 'right';
S3.XAxisLocation = 'bottom';
xlabel(S3,[type,' [nm]'])
ylabel(S3,'Y [mm]')

% donnée de synthèse
data=reshape(Carte,[],1);
masque=~isnan(data);
data=data(masque);
MinMap=min(data);
MaxMap=max(data);
MeanMap=mean(data);
PtV=MaxMap-MinMap;
RmsCarte=std(data-mean(data),'omitnan');

%figure colors
set(fig,'color','w')
colormap inferno

%Canvas features
if ~exist('figtitle','var')
    figtitle='Figure';
end
annotation('textbox',...
[0.05 0.75 0.22 0.10],...
'String',['MATLAB Release R' version('-release')],...
'FontSize',8,...
'FontName','Segoe',...
'EdgeColor',[1 1 1],...
'LineWidth',1,...
'BackgroundColor',[1 1 1],...
'Color',[0 0 1]);

annotation('textbox',...
[0.05 0.65 0.22 0.10],...
'String',figtitle,...
'FontSize',12,...
'FontName','Segoe',...
'EdgeColor',[1 1 1],...
'LineWidth',1,...
'BackgroundColor',[1 1 1],...
'Color',[0 0 0]);

if ~exist('figcomment','var')
    figcomment='';
end
annotation('textbox',...
[0.05 0.45 0.22 0.2],...
'String',figcomment,...
'FontSize',12,...
'FontName','Segoe',...
'EdgeColor',[1 1 1],...
'LineWidth',1,...
'BackgroundColor',[1 1 1],...
'Color',[0 0 0]);

annotation('textbox',...
    [0.05 0.15 0.22 0.3],...
    'String',{['\lambda = ' num2str(wl) ' nm'],...
    ['Radius = ' num2str(radius) ' mm'],...
    ['Min, ',type,'= ' num2str(round(MinMap*1000)/1000) ' nm'],...
    ['Max, ',type,'= ' num2str(round(MaxMap*1000)/1000) ' nm'],...
    ['Mean, ',type,'= ' num2str(round(MeanMap*1000)/1000) ' nm'],...
    ['PV, ',type,'= ' num2str(round(PtV*1000)/1000) ' nm'],...
    ['RMS, ',type,'= ' num2str(round(RmsCarte*1000)/1000) ' nm']},...
    'FontSize',10,...
    'FontName','Segoe',...
    'EdgeColor',[1 1 1],...
    'LineWidth',1,...
    'BackgroundColor',[1 1 1],...
    'Color',[0 0 0]);
% logo
if logo
    imaxis=axes('Parent',fig,'Position',[-0.03,0.85,0.15*2.4552,0.15]);
    logoREOSC=imread('ressources\Logo_REOSC_2013.jpg');
    logoREOSC=imresize(logoREOSC,0.2);
    hImage=image(logoREOSC,'Parent',imaxis);
    
end    
    axis image
    set(imaxis,'Visible','off')
    set( gcf,'Position', [700  215 800 465.33333])
end