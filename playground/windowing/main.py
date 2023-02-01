import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy

from fileHandler import FileHandler 

# sin

def test1():

  frequency = 3;
  frequency2 = 5;
  w = 2 * np.pi * frequency;
  w2 = 2 * np.pi * frequency2;
  time_interval = 1.2;
  samples= 800;
  time = np.linspace(0, time_interval, samples);
  amplitude = np.sin(w*time);
  amplitude += np.sin(w2*time);

  plt.plot(time, amplitude);
  plt.title("Sine wave");
  plt.xlabel("Time");
  plt.ylabel("Amplitude");
  plt.grid(True, which="both");
  plt.axhline(y=0, color="k");
  plt.show(block=False);

  # window

  newAmplitude = amplitude * np.hamming(samples)

  plt.plot(time, newAmplitude);
  plt.title("Sine wave");
  plt.xlabel("Time");
  plt.ylabel("Amplitude");
  plt.grid(True, which="both");
  plt.axhline(y=0, color="k");
  plt.show();

  plt.plot(time, np.hamming(samples))
  plt.show();

  # fft

  def fft(amp):
    fourierTransform = np.fft.fft(amp)  / len(amp)
    fourierTransform = fourierTransform[range(int(len(amp)/2))]

    tpCount = len(amp)
    values = np.arange(int(tpCount/2))
    timePeriod = tpCount/(samples/time_interval)
    frequencies = values/timePeriod

    plt.plot(frequencies[:15], abs(fourierTransform)[:15])
    plt.show(block=False)
    
  def fft2(amp):
    n = int(samples/time_interval)
    freqs = np.fft.fftfreq(n)
    mask = freqs >= 0
    fft_vals = np.fft.fft(amp)
    fft_theo = 2.0*np.abs(fft_vals/n)
    
    plt.figure(1)
    plt.title("OS")
    plt.plot(time, amp, color="xkcd:salmon", label="original")
    plt.legend()
    
    plt.figure(2)
    plt.plot(freqs[mask]*260, fft_theo[:len(mask)][mask], "ro-", label="true fft values")
    plt.title("True FFT values")
    plt.show(block=False)
    
  def fft3(amp):
    freqs = np.fft.fftfreq(samples)
    mask = freqs >= 0
    fft = abs(np.fft.fft(amp))[mask]
    print(freqs.shape)
    print(amp.shape)
    print(freqs[0])
    plt.figure(1)
    plt.plot(np.linspace(0, freqs.size, freqs.size), freqs)
    plt.figure(2)
    plt.plot(np.linspace(0, fft.size, fft.size) / time_interval, fft, "o-")
    plt.show(block=False)
    
    
    
  # fft(amplitude);
  # fft(newAmplitude);
  # fft2(amplitude);
  # fft2(newAmplitude)
  # fft3(amplitude);
  # fft3(newAmplitude);
  # plot.show()

  y, sr = librosa.load("C:\\Users\\SCU8BH\\Downloads\\Casio-MT-45-Piano-C4.wav")
  y_second, sr_second = librosa.load("C:\\Users\\SCU8BH\\Downloads\\1980s-Casio-Harpsichord-C5.wav")
  plt.plot(np.linspace(0, y.size/sr, y.size), y)
  plt.show()

  time_interval = y.size/sr; # sec
  samples = y.size;
  # fft3(y);
  # fft3(y * np.hanning(samples))
  # plot.show()

  coefficient = 0.1/time_interval # brings everything to 0.1 sec

  y_old = y

  samples = int(y.size * coefficient);
  y = y[:samples];
  time_interval = time_interval * coefficient;

  plt.plot(np.linspace(0, samples/sr, samples), y)
  plt.show()

  # fft3(y)
  # fft3(y * np.hanning(samples))
  # fft3(y * np.hamming(samples))
  # plot.show()

  # Tests with lpc TODO: Get the meaning of what I am doing here.

  # Add a signal to the given tone
  y_n = y_second[samples:(2*samples)] # + np.sin(w * np.pi * 200 * np.linspace(0, time_interval, samples)) + np.cos(w * np.pi * 20 * np.linspace(0, time_interval, samples))

  print(librosa.lpc(y, order=2))
  a = librosa.lpc(y * np.hanning(samples), order=20)
  print(a[1:])
  b = np.hstack([[0], -1 * a[1:]])
  y_hat = scipy.signal.lfilter(b, [1], (y_n * np.hanning(samples)))
  fig, ax = plot.subplots()
  ax.plot(y_n * np.hanning(samples))
  ax.plot(y_hat, linestyle='--')
  ax.legend(['y', 'y_hat'])
  ax.set_title('LP Model Forward Prediction')
  plt.show()
  
def test2():
  file = FileHandler("C:/Users/SCU8BH/Documents/T3000/Studienarbeit/Data/50_speakers_audio_data/Speaker_0000/Speaker_0000_00000.wav")
  # file.view()
  print(file.autocorrelate(0.1)[0:20])
  y, frames = file.get_frame(0.1, 3200)
  plt.plot(np.linspace(0, 30, 30), y[0:30])
  # plt.show()
  lpc = np.flip(file.get_lpc(0.1))
  # print(lpc)
  new_y = y[0:20]
  # print(np.pad(new_y[-20:], (0,(20-new_y.size)), mode='constant'))
  # print(np.flip(new_y))
  # print(np.dot(np.pad(new_y[-20:], (0,(20-new_y.size)), mode='constant'), lpc))
  
  while len(new_y) < frames:
    # get last 20 values of new_y
    new_y = np.append(new_y, np.array([np.dot(new_y[-20:], lpc)]))
    
  plt.plot(np.linspace(0, 30, 30), new_y[0:30])
  plt.show()
  

# test2()

print(librosa.lpc(np.array([13.77, 13.6, 13.11, 12.38, 11.48, 10.45]), order=1))
print(librosa.lpc(np.array([13.77, 13.6, 13.11, 12.38, 11.48, 10.45]), order=2))