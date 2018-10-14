tic
% A plot of the Kuramoto-Sivashinsky equation
%
% u_t = -u*u_x - u_xx - u_xxxx, periodic boundary conditions on [0,32*pi]
% computation is based on v = fft(u), so linear term is diagonal
%
% Using this program:
% u is the initial condition
% h is the time step
% N is the number of points calculated along x

% Initial condition and grid setup
N = 1024;
x = 200*(1:N)'/N; % this is used in the plotting
a = -1;
b = 1;
u = cos(x/16) % initial condition
v = fft(u);
% Precompute various scalars for ETDRK4
h = 0.25; % time step
k = [0:N/2-1 0 -N/2+1:-1]'/16; % wave numbers
L = k.^2 - k.^4; % Fourier multipliers
E = exp(h*L); E2 = exp(h*L/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% Main loop
uu = u; tt = 0;
tmax = 150; nmax = round(tmax/h); nplt = floor((tmax/100)/h);
g = -0.5i*k;
for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    if mod(n,nplt)==0
        u = real(ifft(v));
        uu = [uu,u]; tt = [tt,t];
    end
end
% Plot
surf(tt,x,uu), shading interp, lighting phong, axis tight
colormap(summer); set(gca,'zlim')
light('color',[1 1 0],'position',[-1,2,2])
material([0.30 0.60 0.60 40.00 1.00]);
title('The Kuramoto Sivashinsky equation')
xlabel('t'); ylabel('x')
colorbar
timeElapsed = toc
