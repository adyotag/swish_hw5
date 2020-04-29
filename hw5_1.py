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


def volt_to_strain(data, UI=30., GF=2.04, delta_t=1e-6):
    print data.shape
    data = data[:,::int(delta_t/1e-8)]
    print data.shape
    strain = 2.*data[1,:]/(GF*UI) 
    return np.vstack([data[0,:], strain])


def to_freq(data, delta_t=1e-4):
    f_hat = np.fft.fft(data[1,:])
    w = np.fft.fftfreq(len(data[1,:]), delta_t)
    return np.vstack([w, f_hat])


def freq_shift(data, dist=1., init_guess=1.+1.j):  # m
    f_hat_sorted = data[1,np.argsort(data[0,:])]
    frequencies_sorted = np.sort(data[0,:])

    corrected = np.zeros(len(data[1,:]), dtype=complex)
    fun_hist = np.zeros(len(data[1,:]), dtype=complex)
    ksi_hist = np.zeros(len(data[1,:]))


    for i in tqdm(np.arange(len(f_hat_sorted)), ncols=terminal_size()[0]):
        if i >= 2:
            init_guess = ksi_hist[i-1] + (frequencies_sorted[i]-frequencies_sorted[i-1]) * \
                    (ksi_hist[i-1]-ksi_hist[i-2])/(frequencies_sorted[i-1]-frequencies_sorted[i-2])


        ksi_info = ksi(frequencies_sorted[i], init_guess)
        ksi_hist[i] = np.real(ksi_info[0])  + 0.j
        fun_hist[i] = ksi_info[2]
        corrected[i] = np.exp(1j*ksi_hist[i]*dist)*f_hat_sorted[i]


    plt.plot(fun_hist)

    return np.vstack([np.roll(frequencies_sorted, len(frequencies_sorted)//2), np.roll(corrected, len(corrected)//2)])


def to_time(orig_data, data):
    f = np.fft.ifft(data[1,:])
    t = orig_data[0,:]
    return np.vstack([t, f])


def ksi(w, init_guess):
    residual_handle = get_residual(np.real(w))
    soln = root(residual_handle, init_guess, method='anderson', options={'maxiter':1000} )
    return [soln.x, soln.success, soln.fun]



def get_residual(w, cd =  5932., cs =  3170., a=0.0127): # m/s, m/s, m 
    p = lambda k: np.sqrt( (w/cd)**2 - k**2  )
    q = lambda k: np.sqrt( (w/cs)**2 - k**2  )

    first_term = lambda k: -(2*p(k)*w**2)/(a*cs**2) * jv(1, p(k)*a) * jv(1, q(k)*a)  
    second_term = lambda k: 4 * k**2 * p(k) * q(k) * jv(1, p(k)*a) * jv(0, q(k)*a)  
    third_term = lambda k:  ( 2 * k**2 - (w/cs)**2 )**2 * jv(0, p(k)*a) * jv(1, q(k)*a)  

    return lambda k: first_term( k ) + second_term( k ) + third_term( k )



#######################################

# Data from Alex
print '\nExperimental Data...\n'
text_data = np.loadtxt('CKB_Example.txt', skiprows=2).T
b3 = text_data[:2, :]; b4 = text_data[0:3:2, :]

b3 = volt_to_strain(b3, delta_t=1E-6); b4 = volt_to_strain(b4, delta_t=1E-6)
b3_hat = to_freq(b3, 1E-6); b4_hat = to_freq(b4, 1E-6)

print 'Channel B3:\n'; new_b3_hat = freq_shift(b3_hat, 0.61, 1500.+0.1j)
print '\nChannel B4:\n'; new_b4_hat = freq_shift(b4_hat, -0.76, 2.)
new_b3 = np.real(to_time(b3, new_b3_hat)); new_b4 = np.real(to_time(b4, new_b4_hat))



# Plots
print '\n\nPlotting...\n'

plt.figure()
plt.plot(b3[0,:], b3[1,:], label='Strain Gauge')
plt.plot(new_b3[0,:], new_b3[1,:], label='Start of Sample')
plt.xlabel(r'Time, $t$'); plt.ylabel(r'Strain, $\epsilon$'); plt.legend()
plt.title('Strain Evolution at the Strain Gauge and at the Beginning of the Sample')

plt.figure()
plt.plot(b4[0,:], b4[1,:], label='Strain Gauge')
plt.plot(new_b4[0,:], new_b4[1,:], label='End of Sample')
plt.xlabel(r'Time, $t$'); plt.ylabel(r'Strain, $\epsilon$'); plt.legend()
plt.title('Strain Evolution at the Strain Gauge and at the End of the Sample')
plt.show()



print 'Done !\n'
#
