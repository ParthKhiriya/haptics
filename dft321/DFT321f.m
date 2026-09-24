function y = DFT321f(u)
% Initially designed by Nils Landin, Joseph M. Romano, William McMahan, and Katherine J. Kuchenbecker
% Code cleaned by Gunhyuk Park
% If you have concerns on this code, please contact Katherine J.
% Kuchenbecker (kjk@is.mpg.de)

% u: n x 3 array
% n: the length of the signal 
% Each column stores accelleration data of x, y, and z axes, respectively


% Get the dimensions of the data buffer
[N_sampl, N_chann] = size(u);

% Check whether the signal length is even or odd 
Odd = mod(N_sampl,2);

% Declare the transformed signal
U_temp = ones(N_sampl,N_chann)*1i;
U_temp = fft(u);


% The place of the Nyquist component depends on the data buffer being odd or even
if Odd
    N_half = 0;
    N_half = (N_sampl+1)/2;
else
    N_half = 0;
    N_half = N_sampl/2 + 1;
end

% Take the real part of the signal for capturing spectral intensity
absY = real(sqrt(sum(U_temp(1:N_half,:).*conj(U_temp(1:N_half,:)),2)));

% Take the sum of three angles as the integrated angle of each frequency component
PhaseChoice = ones(N_half,1);
PhaseChoice = angle(sum(U_temp(1:N_half,:),2));

% Calculate the integrated spectrum
Y_temp = ones(N_half,1)*1i;
Y_temp = absY.*exp(1i * PhaseChoice);

if Odd
    Y = [Y_temp; conj(Y_temp(end:-1:2))];
else
    Y = [Y_temp; conj(Y_temp(end-1:-1:2))];
end

% Reconstruct a temporal signal from the integrated spectrum
y = real(ifft(Y));
