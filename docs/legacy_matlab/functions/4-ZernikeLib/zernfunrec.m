function Z = zernfunrec(n, m, r , t)

%Compute Zernike pols from jacobi
% see http://mathworld.wolfram.com/ZernikePolynomial.html
% n  m are list same dimension
% r t are list same dimension
r=r(:);
t=t(:);
n=n(:);
m=m(:);

ks=(n-abs(m))/2;
%compute all jacobi polynomials needed one time;
listms=unique(abs(m(ks == floor(ks))));
for ind=1:length(listms)
    v{listms(ind)+1} = j_polynomial ( length(r), max(ks), listms(ind), 0, 1-2*r.^2 );
end
Z=zeros(length(n),length(r));
%now compute zernike polynomial values
inds= find(ks == floor(ks));
for ii=1:length(inds)
    i=inds(ii);
    mt=m(i);
    Z(i,:)=(-1)^(ks(i)) .* v{abs(mt)+1}(:,ks(i)+1) .* r.^abs(mt) .* (sin(abs(mt)*t + (abs(mt)==mt).*pi/2)) ;
end
Z=Z';
end