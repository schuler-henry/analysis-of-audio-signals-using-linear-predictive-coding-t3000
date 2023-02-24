from tracemalloc import start
import librosa
import numpy as np
import matplotlib.pyplot as plt

class FileHandler:
  def __init__(self, filepath):
    self.y, self.sampling_rate = librosa.load(filepath)
    self.total_time = self.y.size / self.sampling_rate
    
    print(self.total_time)
    
  def get_sampling_rate(self):
    return self.sampling_rate  

  def get_frame(self, frame_time, start_frame):
    frame_frames = int(self.sampling_rate * frame_time)
    return self.y[start_frame:(start_frame + frame_frames)], frame_frames
    
  def view(self):
    plt.plot(np.linspace(0, self.y.size, self.y.size), self.y)
    plt.show()
    
  def autocorrelate(self, frame_size):
    frame_frames = int(self.sampling_rate * frame_size)
    frame_y = self.y[3200:(3200 + frame_frames)]
    Fr = np.fft.fft(frame_y)
    S = Fr * np.conjugate(Fr)
    print(Fr)
    
    print(abs(np.fft.ifft(S))[:10])
    print(abs(np.fft.ifft(S)).size)
    
    print(librosa.autocorrelate(frame_y)[:10])
    print(librosa.autocorrelate(frame_y).size)
    
    plt.plot(np.linspace(0, frame_frames, frame_frames), frame_y)
    plt.show()
    plt.plot(np.linspace(0, frame_frames, frame_frames), np.fft.ifft(S))
    plt.plot(np.linspace(0, frame_frames, frame_frames), librosa.autocorrelate(frame_y))
    plt.show()
    return librosa.autocorrelate(frame_y * np.hanning(frame_frames))
    
  def get_lpc(self, frame_time, order):
    frame_y, frame_frames = self.get_frame(frame_time, 3200)

    return librosa.lpc(frame_y * np.hanning(frame_frames), order=order)