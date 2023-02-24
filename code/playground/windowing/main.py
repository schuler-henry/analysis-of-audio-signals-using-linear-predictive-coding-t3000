import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy, scipy.io, scipy.io.wavfile, scipy.signal
import soundfile as sf
import IPython
from numpy.polynomial import polynomial as P

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
  
def test3():
  # spectral leakage 8hz, 9hz
  f = 8 # Hz
  w = 2 * np.pi * f
  time_interval = 1 # s
  samples = 16000
  time = np.linspace(0, time_interval, samples)
  
  f_cos = np.cos(w*time)
  
  # plt.plot(time, f_cos)
  # plt.show()
  
  # plt.plot(np.fft.fftfreq(samples, time_interval/samples), np.abs(np.fft.fft(f_cos)), label="FFT")
  # window = 0.8 # s
  # plt.plot(np.fft.fftfreq(int(window*samples), time_interval/samples), np.abs(np.fft.fft(f_cos[:int(window*samples)])), label="FFT 0.8s")
  # plt.plot(np.fft.fftfreq(int(window*samples), time_interval/samples), np.abs(np.fft.fft(f_cos[:int(window*samples)] * np.hanning(int(window*samples)))), label="FFT 0.8s Hann")
  # plt.plot(np.fft.fftfreq(int(window*samples), time_interval/samples), np.abs(np.fft.fft(f_cos[:int(window*samples)] * np.hamming(int(window*samples)))), label="FFT 0.8s Hamming")
  # plt.legend()
  # plt.show()
  
  f2 = 102 # Hz
  w2 = 2 * np.pi * f2
  f_cos = f_cos + np.cos(w2*time)
  
  # plt.plot(time, f_cos)
  # plt.show()
    
  # plt.plot(np.fft.fftfreq(samples, time_interval/samples), np.abs(np.fft.fft(f_cos)), label="FFT")
  # window = 0.8 # s
  # plt.plot(np.fft.fftfreq(int(window*samples), time_interval/samples), np.abs(np.fft.fft(f_cos[:int(window*samples)])), label="FFT 0.8s")
  # plt.plot(np.fft.fftfreq(int(window*samples), time_interval/samples), np.abs(np.fft.fft(f_cos[:int(window*samples)] * np.hanning(int(window*samples)))), label="FFT 0.8s Hann")
  # plt.plot(np.fft.fftfreq(int(window*samples), time_interval/samples), np.abs(np.fft.fft(f_cos[:int(window*samples)] * np.hamming(int(window*samples)))), label="FFT 0.8s Hamming")
  # plt.legend()
  # plt.show()
  
  window = 0.7
  
  fft_hann = np.abs(np.fft.fft(f_cos[:int(window*samples)] * np.hanning(int(window*samples))))
  fft_freq = np.fft.fftfreq(int(window*samples), time_interval/samples)
  half_size = int(fft_hann.shape[0] / 2)
  
  plt.plot(fft_freq[:half_size], fft_hann[:half_size])
  plt.show()
  
  fft_max = np.amax(fft_hann)
  new_f = 0
  
  for i in range(1, half_size):
    new_f = new_f + np.cos(2*np.pi*fft_freq[i]*time) * (1 if fft_hann[i] > (fft_max/2) else 0)
    if fft_hann[i] > (fft_max/2):
      print(fft_freq[i])
    
  plt.plot(time, f_cos, label="Original function")
  plt.plot(time, new_f, label="Reconstructed function")
  plt.legend()
  plt.show()
  
  cepst = np.fft.ifft(np.log(fft_hann))

  plt.plot(fft_freq, cepst)
  plt.show()

def burg_s12530(N, X):  
  def e(m, n):
    # print(f"e({m}, {n})")
    if m == 0:
      return X[n]
    else:
      return e(m-1, n) + pi(m)*b(m-1, n-1)
  
  def b(m, n):
    # print(f"b({m}, {n})")
    if m == 0:
      return X[n]
    else:
      return b(m-1, n-1) + pi(m)*e(m-1, n)
  
  def pi(m):
    # print(f"pi({m})")
    return (sum(b(m-1, n-1)*e(m-1, n) for n in range (m, N))) / (sum((e(m-1, n)**2 + b(m-1, n-1)**2) for n in range(m, N)))
  
  def sig_squared(m):
    # print(f"sig_squared({m})")
    if m == 0:
      return 1/N * sum((X[n] ** 2) for n in range(0, N))
    else:
      return (1 - (abs(pi(m)) ** 2)) * sig_squared(m - 1)
    
  def a(m, k):
    # print(f"a({m}, {k})")
    if m > 1:
      return a(m-1, k) + pi(m)*a(m-1, m-k)
    elif m == 1 and k == 1:
      return pi(1)
    else:
      return 0
    
  print(a(1, 1))
  print(a(2, 1))
  print(a(3, 0))
  print(a(3, 1))
  print(a(3, 2))
  print(librosa.lpc(np.array(X), order=3))

def burg_marple(N, X):
  def x(i):
    # print(f"x({i})")
    return X[i-1]  
  # Forward propagation error
  # 1 <= k <= N-M
  def f(M, k):
    return x(k+M) + sum(a(M, i)*x(k+M-i) for i in range(1, M+1))
    
  # Backwards propagation error
  # 1 <= k <= N-M
  def b(M, k):
    return sum(np.conj(a(M, i))*x(k+i) for i in range(0, M+1))
    
  # always known: a(M-1, 1) to a(M-1, M-1)
  def a(M, k):
    if k == 0:
      return 1.0
    if M == k:
      return - (2 * sum(np.conj(b(M-1, k))*f(M-1, k+1) for k in range(1, N-M + 1))) / (DEN(M))
    if k >= 1 and k <= M-1:
      return a(M-1, k) + a(M, M)*np.conj(a(M-1, M-k))
    print("Error")
    return 0 
    
  def e(M):
    return sum(abs(f(M,k))**2 for k in range (1, N-M + 1)) + sum(abs(b(M, k))**2 for k in range(1, N-M+1))

  def DEN(M):
    return sum((abs(b(M-1, k))**2 + abs(f(M-1, k+1))**2) for k in range (1, N-M + 1))

  print(a(2, 0))
  print(a(2, 1))
  print(a(2, 2))
  print(librosa.lpc(np.array(X), order=2))

def test4():
  multiplyer = 5
  file = FileHandler("C:/Users/SCU8BH/Documents/T3000/Studienarbeit/Data/50_speakers_audio_data/Speaker_0000/Speaker_0000_00000.wav")
  print(file.get_lpc(multiplyer, 20))
  
  def lpc_noise_synthesize(lpc, samples):
    phase = np.random.uniform(-0.5, 0.5, (samples))
    plt.plot(np.linspace(0, samples, samples), phase, label="phase")
    signal = scipy.signal.lfilter([1.], lpc, phase)
    signal *= 0.005
    plt.plot(np.linspace(0, samples, samples), file.get_frame(multiplyer, 3200)[0], label="original")
    plt.plot(np.linspace(0, samples, samples), signal, label="signal")
    plt.legend()
    plt.show()
    return signal
  
  sf.write("orig.wav", file.get_frame(multiplyer, 3200)[0], file.get_sampling_rate(), subtype="PCM_24")
  sf.write("test.wav", lpc_noise_synthesize(file.get_lpc(multiplyer, 20), int(multiplyer*file.get_sampling_rate())), file.get_sampling_rate(), subtype="PCM_24")
  
def test5():
  def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k
  def play_sound(sound, rate=44100):
    """Play a mono 44Khz sound file in the browser"""
    return IPython.display.display(IPython.display.Audio(sound,rate=rate))
  def lsp_to_lpc(lsp):  
    """Convert line spectral pairs to LPC"""
    ps = np.concatenate((lsp[:,0], -lsp[::-1,0], [np.pi]))
    qs = np.concatenate((lsp[:,1], [0], -lsp[::-1,1]))
    
    p = np.cos(ps) - np.sin(ps)*1.0j
    q = np.cos(qs) - np.sin(qs)*1.0j
    
    p = np.real(P.polyfromroots(p))
    q = -np.real(P.polyfromroots(q))
    
    a = 0.5 * (p+q)
    return a[:-1]

    
  def lpc_noise_synthesize(lpc, samples=10000):
      """Apply LPC coefficients to white noise"""
      phase = np.random.uniform(0,0.5,(samples))
      signal= scipy.signal.lfilter([1.], lpc, phase)        
      return signal
      
  def lpc_buzz_synthesize(lpc, f, sr, samples=10000):       
      """Apply LPC coefficients to a sawtooth with the given frequency and sample rate"""
      phase = scipy.signal.sawtooth(2*np.pi*f*np.arange(samples)/(sr))
      signal= scipy.signal.lfilter([1.], lpc, phase)        
      return signal

  def lpc_to_lsp(lpc):    
      """Convert LPC to line spectral pairs"""
      l = len(lpc)+1
      a = np.zeros((l,))        
      a[0:-1] = lpc
      p = np.zeros((l,))
      q = np.zeros((l,))    
      for i in range(l):
          j = l-i-1
          p[i] = a[i] + a[j]
          q[i] = a[i] - a[j]
      
      ps = np.sort(np.angle(np.roots(p)))
      qs = np.sort(np.angle(np.roots(q)))            
      lsp = np.vstack([ps[:len(ps)//2],qs[:len(qs)//2]]).T    
      return lsp

  def lpc_to_formants(lpc, sr):    
      """Convert LPC to formants    
      """
          
      # extract roots, get angle and radius
      roots = np.roots(lpc)
      
      pos_roots = roots[np.imag(roots)>=0]
      if len(pos_roots)<len(roots)//2:
          pos_roots = list(pos_roots) + [0] * (len(roots)//2 - len(pos_roots))
      if len(pos_roots)>len(roots)//2:
          pos_roots = pos_roots[:len(roots)//2]
      
      w = np.angle(pos_roots)
      a = np.abs(pos_roots)
      
      order = np.argsort(w)
      w = w[order]
      a = a[order]
      
      freqs = w * (sr/(2*np.pi))
      bws =  -0.5 * (sr/(2*np.pi)) * np.log(a)    
      
      # exclude DC and sr/2 frequencies
      return freqs, bws
  def load_wave(fname):
    """Load a 16 bit wave file and return normalised in 0,1 range"""
    # load and return a wave file
    sr, wave = scipy.io.wavfile.read(fname)
    return wave/32768.0
  def lpc(wave, order):
    """Compute LPC of the waveform. 
    a: the LPC coefficients
    e: the total error
    k: the reflection coefficients
    
    Typically only a is required.
    """    
    # only use right half of autocorrelation, normalised by total length
    autocorr = scipy.signal.correlate(wave, wave)[len(wave)-1:]/len(wave)
    a, e, k  = levinson_1d(autocorr, order)
    return a,e,k
  
  def modfm_buzz(samples, f, sr, k):
    """Generate a pulse train using modfm:
        y(t) = cos(x(t)) * exp(cos(x(t))*k - k)
        
        samples: number of samples to generate
        f: base frequency (Hz)
        sr: sample rate (Hz)
        k: modulation depth; higher has more harmonics but increases risk of aliasing
        (e.g. k=1000 for f=50, k=100 for f=200, k=2 for f=4000)        
    
    """
    t = np.arange(samples)
    phase = (f*2*np.pi * (t/float(sr)))
    # simple pulse oscillator (ModFM)
    buzz = np.cos(phase) * np.exp(np.cos(phase)*k-k)    
   
    return buzz

  def noise(samples):
      """Generate white noise in range [-1,1]
      
      samples: number of samples to generate
      """
      return np.random.uniform(-1,1,size=samples)
  
  def lpc_vocode(wave, frame_len, order, carrier, residual_amp=0.0, vocode_amp=1.0, env=False, 
               freq_shift=1.0):    
    """
    Apply LPC vocoding to a pair of signals using 50% overlap-add Hamming window resynthesis
    The modulator `wave` is applied to the carrier `imposed`
    
    Parameters:
    ---
    wave: modulator wave
    frame_len: length of frames
    order: LPC order (typically 2-30)
    carrier: carrier signal; should be at least as long as wave
    residual_amp: amplitude of LPC residual to include in output
    vocode_amp: amplitude of vocoded signal 
    env: if True, the original volume envelope of wave is imposed on the output
          otherwise, no volume modulation is applied
    freq_shift: (default 1.0) shift the frequency of the resonances by the given scale factor. Warning :
        values >1.1 are usually unstable, and values <0.5 likewise.
    """
    
    # precompute the hamming window
    window = scipy.signal.hann(frame_len)
    t = np.arange(frame_len)
    #allocate the array for the output
    vocode = np.zeros(len(wave+frame_len))    
    last = np.zeros(order)
    # 50% window steps for overlap-add
    for i in range(0,len(wave),frame_len//2):
        # slice the wave
        wave_slice = wave[i:i+frame_len]
        carrier_slice = carrier[i:i+frame_len]
        if len(wave_slice)==frame_len:                        
            # compute LPC
            a,error,reflection = lpc(wave_slice, order)           
            
            # apply shifting in LSP space
            lsp = lpc_to_lsp(a)
            lsp = (lsp * freq_shift+np.pi) % (np.pi) -np.pi     
            a = lsp_to_lpc(lsp)
            
            # compute the LPC residual         
            residual = scipy.signal.lfilter(a, 1., wave_slice)           
            # filter, using LPC as the *IIR* component
            #vocoded, last = scipy.signal.lfilter([1.], a, carrier_slice, zi=last)             
            vocoded = scipy.signal.lfilter([1.], a, carrier_slice)             
            
            # match RMS of original signal
            if env:
                voc_amp = 1e-5+np.sqrt(np.mean(vocoded**2))
                wave_amp = 1e-5+np.sqrt(np.mean(wave_slice**2))
                vocoded = vocoded * (wave_amp/voc_amp)
          
            # Hann window 50%-overlap-add to remove clicking
            vocode[i:i+frame_len] +=  (vocoded * vocode_amp + residual * residual_amp) * window

    return vocode[:len(wave)]
  
  modulator = load_wave("C:\\Users\\SCU8BH\\Downloads\\1.wav")
  # modulator = load_wave("C:\\Users\\SCU8BH\\Documents\\T3000\\orig.wav")

  # ModFM pulse train, with exp. decreasing modulation depth (lowpass filter effect)
  carrier = modfm_buzz(len(modulator), f=40*np.floor(np.linspace(1,6,len(modulator)))**0.25,
                      sr=44100, k=10**np.linspace(4,2,len(modulator)))


  vocoded = lpc_vocode(modulator, frame_len=500, order=48, carrier=carrier, 
                      residual_amp=0, vocode_amp=1, env=True, freq_shift=1)                        
  play_sound(modulator)
  play_sound(carrier)
  play_sound(vocoded)
  sf.write("testout.wav", vocoded, 44100, subtype="PCM_24")
  
def test6():
  y, samplingrate = librosa.load("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0000\\Speaker_0000_00000.wav")
  f_samples = int(0.45)
  f_samples = int(0.5 * samplingrate)
  f_time = np.linspace(0, 50, f_samples)
  
  f = 800 # Hz
  w = 2 * np.pi * f
  time_interval = 1 # s
  samples = 16000
  time = np.linspace(0, time_interval, samples)
  
  f_cos = np.cos(w*time)
  
  f2 = 1000 # Hz
  w2 = 2 * np.pi * f2
  f_cos = f_cos + np.cos(w2*time)
  
  fft = np.abs(np.fft.fft(f_cos))
  fft_freq = np.fft.fftfreq(samples, time_interval/samples)
  
  # cep = np.abs(np.fft.ifft(np.log(fft**2)))**2
  cep = np.fft.ifft(np.log(np.fft.fft(f_cos)))
  
  fig, axs = plt.subplots(3)
  axs[0].plot(f_time, y[:f_samples])
  axs[1].plot(fft_freq, fft)
  axs[2].plot(np.linspace(0, samples, samples), cep)
  plt.show()
  
def test7(src, frame_time, start_time):
  y, samplingrate = librosa.load(src)
  start_sample = int(start_time * samplingrate)
  samples = int(frame_time * samplingrate)
  end_sample = start_sample + samples
  f_time = np.linspace(0, 1000*frame_time, samples)

  fig, axs = plt.subplots(20)
  # fig_all, axs_all = plt.subplots(1)
  
  max_value = np.zeros(20)
  sum_value = np.zeros(20)
  center = np.zeros(20)
  
  for i in range(0, 20):
    print(f"Round: {i}")
    
    axs[i].plot(f_time, y[start_sample:end_sample])
    
    lpc_values = librosa.lpc(y[start_sample:end_sample], order=12) # a1 bis a12 sind die lpc werte; a0 ist standardmäßig 1
    print(lpc_values)
    axs2[i].plot(np.linspace(0, 13, 13), lpc_values, label="test")
    axs3[i].plot(np.linspace(0, 13, 13), lpc_to_lpcc(lpc_values), label="lpcc")
    # axs4[i].plot(np.linspace(0, 13, 13), lpcc_to_wlpcc(lpc_to_lpcc(lpc_values)), label="wlpcc")
    # axs_all.plot(np.linspace(0, 13, 13), lpc_values, label=f"Round: {i}")
    
    if max_value[i] < np.abs(np.max(lpc_values)):
      max_value[i] = np.abs(np.max(lpc_values))
      
    for j in range(0, 13):
      sum_value[i] += np.abs(lpc_values[j])
      center[i] += j * np.abs(lpc_values[j])
    center[i] = center[i] / sum_value[i]
    
    start_sample += samples
    end_sample+= samples
    
  # fig_overall, axs_overall = plt.subplots(1)
  # axs_overall.plot(np.linspace(0, 20, 20), max_value, label="max")
  # axs_center.plot(np.linspace(0, 20, 20), center, label=f"center {frame_time}")
  # axs_center.legend()
  # axs_overall.legend()
  
  plt.ylim(-1000, 1000)

def plotAudio(src):
  y, samplingrate = librosa.load(src)
  
  plt.plot(np.linspace(0, y.shape[0]/samplingrate, y.shape[0]), y)
  plt.show()
  
def lpc_to_lpcc(lpc):
  lpcc = np.zeros(lpc.shape)
  lpcc[0] = lpc[0]
  for i in range(2, lpc.shape[0] + 1):
    lpcc[i-1] = sum((1-k/i)* lpc[k-1] * lpcc[i-k-1] for k in range(1, i)) + lpc[i-1]
  # lpcc[1] = lpc[1] 
  # for i in range(2, lpc.shape[0]):
    # lpcc[i] = sum((1-k/i)* lpc[k] * lpcc[i-k] for k in range(1, i)) + lpc[i]
  return lpcc

def lpcc_to_wlpcc(lpcc):
  Q = lpcc.shape[0] - 1
  def w(m):
    return 1 + (Q/2) * np.sin((np.pi * m)/Q)
  
  wlpcc = np.zeros(lpcc.shape)
  
  for i in range(1, Q + 1):
    wlpcc[i] = w(i) * lpcc[i]
    
  return wlpcc
    
  
# test2()
# test3()
# burg_marple(4, [1.0,2.0,3.0, 4.0])

# plotAudio("C:\\Users\\SCU8BH\\Documents\\T3000\\Hallo, das ist ein Test.wav")
# plotAudio("C:\\Users\\SCU8BH\\Documents\\T3000\\Account einloggen.wav")

# fig_x, axs_center = plt.subplots(1)
fig2, axs2 = plt.subplots(20)
fig3, axs3 = plt.subplots(20)
fig4, axs4 = plt.subplots(20)
test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Hallo, das ist ein Test.wav", 0.1, 1.44)
test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Hallo, das ist ein Test.wav", 0.05, 1.44)
test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Hallo, das ist ein Test.wav", 0.01, 1.44)
# fig_x, axs_center = plt.subplots(1)
fig2, axs2 = plt.subplots(20)
fig3, axs3 = plt.subplots(20)
fig4, axs4 = plt.subplots(20)
test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Account einloggen.wav", 0.1, 0.89)
test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Account einloggen.wav", 0.05, 0.89)
test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Account einloggen.wav", 0.01, 0.89)
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0000\\Speaker_0000_00000.wav")
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0000\\Speaker_0000_00001.wav")
# fig2, axs2 = plt.subplots(20)
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0001\\Speaker_0001_00000.wav")
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0001\\Speaker_0001_00001.wav")
# fig2, axs2 = plt.subplots(20)
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0002\\Speaker_0002_00000.wav")
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0002\\Speaker_0002_00001.wav")
# fig2, axs2 = plt.subplots(20)
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0003\\Speaker_0003_00000.wav")
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0003\\Speaker_0003_00001.wav")
# fig2, axs2 = plt.subplots(20)
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0004\\Speaker_0004_00000.wav")
# test7("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0004\\Speaker_0004_00001.wav")
plt.legend()
plt.show()


# print(librosa.lpc(np.array([13.77, 13.6, 13.11, 12.38, 11.48, 10.45]), order=1))
# print(librosa.lpc(np.array([13.77, 13.6, 13.11, 12.38, 11.48, 10.45]), order=2))

