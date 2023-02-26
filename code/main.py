from preprocessing.AudioPreprocessor import AudioPreprocessor
from feature_extraction.LPCExtractor import LPCExtractor

def main():
    frames = AudioPreprocessor.load_preprocessed_frames("C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0003\\Speaker_0003_00000.wav")
    lpccs = LPCExtractor.get_lpcc_from_frames(frames=frames, order=12)
    print(len(lpccs))
    
if __name__ == "__main__":
    main()