import numpy as np

def apply_dft321(u: np.ndarray) -> np.ndarray:
    """
    Python implementation of the DFT321 algorithm to convert 3D acceleration to 1D.
    Directly translates the MATLAB logic designed by Nils Landin, Joseph M. Romano, 
    William McMahan, and Katherine J. Kuchenbecker.
    """
    # Get dimensions
    N_sampl, N_chann = u.shape
    
    if N_chann != 3:
        raise ValueError(f"DFT321 expects exactly 3 channels, but got {N_chann}")

    # Check whether the signal length is even or odd
    is_odd = (N_sampl % 2 != 0)

    # The place of the Nyquist component depends on the data buffer being odd or even
    if is_odd:
        N_half = int((N_sampl + 1) / 2)
    else:
        N_half = int(N_sampl / 2 + 1)

    # Perform Fast Fourier Transform
    U_temp = np.fft.fft(u, axis=0)
    U_half = U_temp[:N_half, :]

    # Take the real part of the signal for capturing spectral intensity
    # (equivalent to L2 norm of the complex magnitudes across channels)
    absY = np.sqrt(np.sum(np.abs(U_half)**2, axis=1))

    # Take the sum of three angles as the integrated angle of each frequency component
    phase_choice = np.angle(np.sum(U_half, axis=1))

    # Calculate the integrated spectrum
    Y_temp = absY * np.exp(1j * phase_choice)

    # Reconstruct a temporal signal from the integrated spectrum
    if is_odd:
        # Append conjugated reversed spectrum (matching MATLAB: conj(Y_temp(end:-1:2)))
        Y = np.concatenate([Y_temp, np.conj(Y_temp[-1:0:-1])])
    else:
        # Append conjugated reversed spectrum (matching MATLAB: conj(Y_temp(end-1:-1:2)))
        Y = np.concatenate([Y_temp, np.conj(Y_temp[-2:0:-1])])

    # Transform back to the time domain
    y = np.real(np.fft.ifft(Y))
    
    # Return as column vector (N, 1)
    return y.reshape(-1, 1)