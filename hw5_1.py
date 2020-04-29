from matplotlib import pyplot as plt
from scipy.optimize import root, fsolve, minimize, newton
from scipy.special import jv
from tqdm import tqdm
import numpy as np
import cmath as cm
import sys


def terminal_size():
    import fcntl, termios, struct
    h, w, hp, wp = struct.unpack('HHHH',
            fcntl.ioctl(0, termios.TIOCGWINSZ,
            struct.pack('HHHH', 0, 0, 0, 0)))
    return w, h


def generate_rect_pulse(delta_t=1e-8):
    t = np.arange(0, 0.006, delta_t); f = np.zeros(len(t));
    f[len(t)/2 - len(t)/4:len(t)/2 + len(t)/4] = 1.E-3
    return np.vstack([t, f])


def to_freq(data, delta_t=1e-4):
    f_hat = np.fft.fft(data[1,:])
    w = np.fft.fftfreq(len(data[1,:]), delta_t)
    return np.vstack([w, f_hat])


def freq_shift(data, dist=1., init_guess=0.):  # m
    f_hat_sorted = data[1,np.argsort(data[0,:])]
    frequencies_sorted = np.sort(data[0,:])

    corrected = np.zeros(len(data[1,:]), dtype=complex)
    ksi_hist = np.zeros(len(data[1,:]))


    for i in tqdm(np.arange(len(f_hat_sorted)), ncols=terminal_size()[0]):
        if i >= 2:
            init_guess = ksi_hist[i-1] + (frequencies_sorted[i]-frequencies_sorted[i-1]) * \
                    (ksi_hist[i-1]-ksi_hist[i-2])/(frequencies_sorted[i-1]-frequencies_sorted[i-2])


        ksi_info = ksi(frequencies_sorted[i], init_guess)
        ksi_hist[i] = np.real(ksi_info[0])  + 0.j

        corrected[i] = np.exp(1j*ksi_hist[i]*dist)*f_hat_sorted[i]


    return np.vstack([np.roll(frequencies_sorted, len(frequencies_sorted)//2), np.roll(corrected, len(corrected)//2)])


def to_time(orig_data, data):
    f = np.fft.ifft(data[1,:])
    t = orig_data[0,:]
    return np.vstack([t, f])


def ksi(w, init_guess):
    residual_handle = get_residual(np.real(w))
    soln = root(residual_handle, init_guess, method='anderson', options={'maxiter':1000} )
    print '\nOMEGA = ', w
    print '\nINIT GUESS = ', init_guess
    print '\n', soln, '\n\n'
    return [soln.x, soln.success]



def get_residual(w, cd =  5932., cs =  3170., a=0.0127): # m/s, m/s, m 
    p = lambda k: np.sqrt( (w/cd)**2 - k**2  )
    q = lambda k: np.sqrt( (w/cs)**2 - k**2  )

    first_term = lambda k: -(2*p(k)*w**2)/(a*cs**2) * jv(1, p(k)*a) * jv(1, q(k)*a)  
    second_term = lambda k: 4 * k**2 * p(k) * q(k) * jv(1, p(k)*a) * jv(0, q(k)*a)  
    third_term = lambda k:  ( 2 * k**2 - (w/cs)**2 )**2 * jv(0, p(k)*a) * jv(1, q(k)*a)  

    return lambda k: (first_term( k ) + second_term( k ) + third_term( k ))**2



#######################################

# Rectangular Pulse
print '\nRectangular Pulse...\n'
pulse = generate_rect_pulse(delta_t=1.E-5)
pulse_hat = to_freq(pulse, delta_t=1.E-5)
new_pulse_hat = freq_shift(pulse_hat, dist=1., init_guess=500.+0j)
new_pulse = np.real(to_time(pulse, new_pulse_hat))


# Plots
print '\n\nPlotting...\n'
plt.figure()
plt.plot(pulse[0,:], pulse[1,:], label='Original')
plt.plot(new_pulse[0,:], new_pulse[1,:], label='Dispersion After 1 meter')
plt.xlabel(r'Time, $t$'); plt.ylabel(r'Strain, $\epsilon$'); plt.legend()
plt.title('Dispersion of a Rectangular Wave along a Cylindrical Rod')
plt.show()

print 'Done !\n'
#
