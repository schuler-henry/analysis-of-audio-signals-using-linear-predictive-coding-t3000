from utils.utils import Utils
from preprocessing.AudioPreprocessor import AudioPreprocessor
from feature_extraction.LPCExtractor import LPCExtractor
import numpy as np
import tensorflow as tf
from tensorflow import keras

def calc_average_lpc(lpcc_list):
    print(len(lpcc_list))
    print(lpcc_list[20])
    print(lpcc_list[0].dtype)
    
    for index in range(1, lpcc_list[0].shape[0]):
        average = sum(lpcc_list[n][index] for n in range(0, len(lpcc_list))) / len(lpcc_list)
        print(average)
        
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
        
def get_data_set(count):
    third = int(count/3)
    X = np.zeros((count, 12*10))
    y = np.zeros(count, dtype='uint8')
    
    coefficients_per_speaker = third * 10
    
    all_speakers = [[], [], []]
    for i in range(0, 3):
        index = 0
        while (len(all_speakers[i]) < coefficients_per_speaker):
            print(index)
            y_, sr = Utils.load_file(f"C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_{i:04}\\Speaker_{i:04}_{index:05}.wav")
        
            y_ = AudioPreprocessor.remove_noise(y=y_, sr=sr)
            y_ = AudioPreprocessor.remove_silence(y=y_)
            frames = AudioPreprocessor.create_frames(y=y_, frame_size=500, overlap=100)
            frames = AudioPreprocessor.window_frames(frames=frames)
            
            lpc = LPCExtractor.lpc(frames=frames, order=12)
            lpcc = LPCExtractor.lpcc(lpc_list=lpc, order=12)
            
            all_speakers[i] += lpcc
            
            index += 1
    
    for i in range(0, 3):
        for j in range(0, third):
            X[i*third + j] = np.concatenate((all_speakers[i][10*j][1:13], 
                                           all_speakers[i][10*j+1][1:13], 
                                           all_speakers[i][10*j+2][1:13],
                                           all_speakers[i][10*j+3][1:13],
                                           all_speakers[i][10*j+4][1:13],
                                           all_speakers[i][10*j+5][1:13],
                                           all_speakers[i][10*j+6][1:13],
                                           all_speakers[i][10*j+7][1:13],
                                           all_speakers[i][10*j+8][1:13],
                                           all_speakers[i][10*j+9][1:13]
                                         ))
            y[i*third + j] = i
            
    return X, y
    
def main():
    count = 2400
    X, y = get_data_set(count)
    print(y)
    X, y = unison_shuffled_copies(X, y)
    print(y)
    # model takes 10 frames a 12 coefficients
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=[12*10]),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X[int(5*count/6):], y[int(5*count/6):], epochs=1000)
    
    test_loss, test_acc = model.evaluate(X[-int(count/6):], y[-int(count/6):])
    
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")
    
    
    y_, sr = Utils.load_file(f"C:\\Users\\SCU8BH\\Documents\\T3000\\Studienarbeit\\Data\\50_speakers_audio_data\\Speaker_0000\\Speaker_0000_00002.wav")
        
    y_ = AudioPreprocessor.remove_noise(y=y_, sr=sr)
    y_ = AudioPreprocessor.remove_silence(y=y_)
    frames = AudioPreprocessor.create_frames(y=y_, frame_size=500, overlap=100)
    frames = AudioPreprocessor.window_frames(frames=frames)
    
    lpc = LPCExtractor.lpc(frames=frames, order=12)
    lpcc = LPCExtractor.lpcc(lpc_list=lpc, order=12)
    
    X = np.zeros((int(len(lpcc)/10), 12*10))
    
    for j in range(0, int(len(lpcc)/10)):
        X[j] = np.concatenate((lpcc[10*j][1:13], 
                                lpcc[10*j+1][1:13], 
                                lpcc[10*j+2][1:13],
                                lpcc[10*j+3][1:13],
                                lpcc[10*j+4][1:13],
                                lpcc[10*j+5][1:13],
                                lpcc[10*j+6][1:13],
                                lpcc[10*j+7][1:13],
                                lpcc[10*j+8][1:13],
                                lpcc[10*j+9][1:13]
                                ))
    if X.shape[0] > 100:
        X_2 = X[-100:]
        print(X[0])
        print(X[99])
    pred = model.predict(X)
    print(np.argmax(pred, axis=1))
    print(np.count_nonzero(np.argmax(pred, axis=1) == 0))
    print(np.count_nonzero(np.argmax(pred, axis=1) == 1))
    print(np.count_nonzero(np.argmax(pred, axis=1) == 2))
    # print(y[-100:])
    
    
if __name__ == "__main__":
    main()