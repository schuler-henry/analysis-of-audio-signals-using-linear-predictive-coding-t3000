from AudioPreprocessor.AudioPreprocessor import AudioPreprocessor
from FeatureExtractor.FeatureExtractor import FeatureExtractor, Feature

import librosa
import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def main():
    filePath = "C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0003\\Speaker_0003_00000.wav"
    
    # Load audio file
    y, sr = librosa.load(filePath)

    # Preprocess audio file
    y = AudioPreprocessor.remove_noise(y=y, sr=sr)
    y = AudioPreprocessor.remove_silence(y=y)
    # frame-duration: 0.2 s, overlap: 0.1 s
    frames = AudioPreprocessor.create_frames(y=y, frame_size=int(sr / 5), overlap=int(sr / 10))
    windowed_frames = AudioPreprocessor.window_frames(frames=frames)

    # Extract features
    feature_extractor = FeatureExtractor(windowed_frames, sr)
    # Create LPC features with 13 coefficients per frame and no derivatives
    extraction_pattern = [
        [Feature.LPC, 13, []]
    ]
    features = feature_extractor.extract_features(extraction_pattern)
    
if __name__ == "__main__":
    main()