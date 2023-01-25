from pickle import FALSE
import numpy as np
import matplotlib.pyplot as plot

# sin

frequency = 30;
frequency2 = 5;
w = 2 * np.pi * frequency;
w2 = 2 * np.pi * frequency2;
time_interval = 1.8;
samples= 800;
time = np.linspace(0, time_interval, samples);
amplitude = np.sin(w*time);
amplitude += np.sin(w2*time);

plot.plot(time, amplitude);
plot.title("Sine wave");
plot.xlabel("Time");
plot.ylabel("Amplitude");
plot.grid(True, which="both");
plot.axhline(y=0, color="k");
plot.show(block=False);

# window

newAmplitude = amplitude * np.hamming(samples)

plot.plot(time, newAmplitude);
plot.title("Sine wave");
plot.xlabel("Time");
plot.ylabel("Amplitude");
plot.grid(True, which="both");
plot.axhline(y=0, color="k");
plot.show();

plot.plot(time, np.hamming(samples))
plot.show();

# fft

def fft(amp):
  fourierTransform = np.fft.fft(amp)  / len(amp)
  fourierTransform = fourierTransform[range(int(len(amp)/2))]

  tpCount = len(amp)
  values = np.arange(int(tpCount/2))
  timePeriod = tpCount/(samples/time_interval)
  frequencies = values/timePeriod

  plot.plot(frequencies[:15], abs(fourierTransform)[:15])
  plot.show(block=False)
  
def fft2(amp):
  n = int(samples/time_interval)
  freqs = np.fft.fftfreq(n)
  mask = freqs > 0
  fft_vals = np.fft.fft(amp)
  fft_theo = 2.0*np.abs(fft_vals/n)
  
  plot.figure(1)
  plot.title("OS")
  plot.plot(time, amp, color="xkcd:salmon", label="original")
  plot.legend()
  
  plot.figure(2)
  plot.plot(freqs[mask]*260, fft_theo[:len(mask)][mask], "ro-", label="true fft values")
  plot.title("True FFT values")
  plot.show(block=False)
  
  
# fft(amplitude);
# fft(newAmplitude);
fft2(amplitude);
fft2(newAmplitude)
plot.show()