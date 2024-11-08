import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as scp
from pylab import *

def gaussian_noise(p,mu,std):
    # p is the pulse
    # mu is the mean
    # std is the standard deviation
    noise = np.random.normal(mu, std, size = p.shape)
    noisy_p = p + noise
    return noisy_p 

def derive_filter(signal, filter_coeffs):
    # Apply the filter using the 'lfilter' function
    filtered_signal = scp.lfilter(filter_coeffs, 1.0, signal)
    return filtered_signal

def find_zero_crossings(t, signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return t[zero_crossings]

fs = 64.0 
ts = 1 / fs
amplitude = 1
duration = 1
sample_count = int(duration/ts)
timestep = [ts * i for i in range(sample_count)]

# A and B Values are ajusted to the double exponential last around 1ns with a quick rising time 
A = 11
B = 10

tmax = math.log(B/A)/(B-A)
current_max = math.exp(-A*tmax)-math.exp(-B*tmax)
C = amplitude/current_max

x = []
for i in range(sample_count):
    x.append(C*(math.exp(-A*timestep[i])-math.exp(-B*timestep[i])))
x = np.asarray(x)
t = np.asarray(timestep)
x = x/x.max() # normalize
x[x<(amplitude/1000)] = 0

# Add Gaussian noise
xg= gaussian_noise(x,0.0,0.02) 

# Create a FIR filter and apply it to x:
# The Nyquist rate of the signal.
nyq_rate = fs / 2.0

# The desired width of the transition from pass to stop, relative to the Nyquist rate.  We'll design the filter with a 5 Hz transition width.
width = 5/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = scp.kaiserord(ripple_db, width)
print(N, beta)

# The cutoff frequency of the filter.
cutoff_hz = 17.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = scp.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = scp.lfilter(taps, 1.0, xg) 

# Plot the FIR filter coefficients.
fig, axs = plt.subplots(1,1, figsize=(20, 15))
figure(1)
plot(taps, 'bo-', linewidth=2)
print(taps) #we need them for FIR/FPGA implementation(DSP)
print('we need them for FIR/FPGA implementation(DSP)')
title('Filter Coefficients (%d taps)' % N)
grid(True)

# Plot the magnitude response of the filter.
figure(2)
clf()
w, h = scp.freqz(taps, worN=8000)
fig, axs = plt.subplots(1,1, figsize=(22, 15))  
plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
xlabel('Frequency (GHz)')
ylabel('Gain')
title('Frequency Response')
grid(True)

# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / fs
figure(3)

# Plot the original signal.
fig, axs = plt.subplots(1,1, figsize=(22, 15))
xlabel('Time (ns)')
ylabel('Amplitude')  
plot(t, x,'-g')

#Plot the noisy signal.
xlabel('Time (ns)')
ylabel('Amplitude')  
plot(t, xg, '-b') 

# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t-delay, filtered_x, 'r-')

# Plot just the "good" part of the filtered signal.  The first N-1 samples are "corrupted" by the initial conditions.
xlabel('Time(ns)')
grid(True)
show()

# Define the filter coefficients for differentiation (1st order derivative)
filter_coeffs = np.array([1, 0, -1])

# Derive the filtered signal
dfirx = derive_filter(filtered_x, filter_coeffs)

# Plot the derivated signal
fig, axs = plt.subplots(1,1, figsize=(22, 15))
plot(t, filtered_x,'r-',label='Filtered Signal')
plot(t, dfirx, label='Derivated Filtered Signal (1st Order Derivative)')
xlabel('Time(ns)')
ylabel('Amplitude')
legend()
grid(True)
show()

#Apply the zero-crossing method to the derivated-filtered signal :
# Find zero crossings
zero_crossings = find_zero_crossings(t, dfirx)

# Plot the biexponential signal and zero crossings
fig, axs = plt.subplots(1,1, figsize=(22, 15))
plot(t,dfirx,'b-', label='Derivated Filtered Signal')
scatter(zero_crossings, np.zeros_like(zero_crossings), color='red', marker='o', label='Zero Crossings')
xlabel('Time(ns)')
ylabel('Amplitude')
legend()
grid(True)
print(zero_crossings)
show()