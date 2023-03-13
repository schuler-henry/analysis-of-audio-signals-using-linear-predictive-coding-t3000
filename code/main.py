from AudioPreprocessor.AudioPreprocessor import AudioPreprocessor
from FeatureExtractor.FeatureExtractor import FeatureExtractor, Feature
from FeatureEvaluator.FeatureEvaluator import FeatureEvaluator

import librosa
import numpy as np

def main():
    evaluator = FeatureEvaluator("/home/henry/Downloads/archive/50_speakers_audio_data")

    X, y = evaluator.create_dataset([0, 1, 2, 3, 4], [[Feature.LPC, 13, []]], 10, 200, 4410, 2205, np.hanning, start_at_file_index=0)
    evaluator.set_model_dataset(X, y)

    X, y = evaluator.create_dataset([0, 1, 2, 3, 4], [[Feature.LPC, 13, []]], 10, 200, 4410, 2205, np.hanning, start_at_file_index=12)
    evaluator.set_evaluation_dataset(X, y)

    evaluator.create_nn_model(epochs=1000)

    evaluator.evaluate_model()
    
    
if __name__ == "__main__":
    main()