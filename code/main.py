from preprocessing.AudioPreprocessor import AudioPreprocessor

def main():
    frames = AudioPreprocessor.load_preprocessed_frames("./audio.wav")
    print(frames)
    
if __name__ == "__main__":
    main()