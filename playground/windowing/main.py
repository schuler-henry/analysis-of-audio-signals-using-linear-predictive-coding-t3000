from pickle import FALSE
import numpy as np
import matplotlib.pyplot as plot
import librosa

# sin

frequency = 3;
frequency2 = 5;
w = 2 * np.pi * frequency;
w2 = 2 * np.pi * frequency2;
time_interval = 1.2;
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
  mask = freqs >= 0
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
  
def fft3(amp):
  freqs = np.fft.fftfreq(samples)
  mask = freqs >= 0
  fft = abs(np.fft.fft(amp))[mask]
  print(freqs.shape)
  print(amp.shape)
  print(freqs[0])
  plot.figure(1)
  plot.plot(np.linspace(0, freqs.size, freqs.size), freqs)
  plot.figure(2)
  plot.plot(np.linspace(0, fft.size, fft.size) / time_interval, fft, "o-")
  plot.show(block=False)
  
  
  
# fft(amplitude);
# fft(newAmplitude);
# fft2(amplitude);
# fft2(newAmplitude)
fft3(amplitude);
fft3(newAmplitude);
plot.show()

y, sr = librosa.load("C:\\Users\\SCU8BH\\Downloads\\Casio-MT-45-Piano-C4.wav")
plot.plot(np.linspace(0, y.size/sr, y.size), y)
plot.show()

time_interval = y.size/sr; # sec
samples = y.size;
fft3(y);
fft3(y * np.hanning(samples))
plot.show()

coefficient = 0.1/time_interval # brings everything to 0.1 sec

samples = int(y.size * coefficient);
y = y[:samples];
time_interval = time_interval * coefficient;

plot.plot(np.linspace(0, samples/sr, samples), y)
plot.show()

fft3(y)
fft3(y * np.hanning(samples))
fft3(y * np.hamming(samples))
plot.show()