import librosa

class Utils:
    @staticmethod
    def load_file(file_path):
        return librosa.load(file_path)
