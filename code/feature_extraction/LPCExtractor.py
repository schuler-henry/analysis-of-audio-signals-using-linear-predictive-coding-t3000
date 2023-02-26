import numpy as np
import librosa

class LPCExtractor:
    @staticmethod
    def lpc(frames, order=12):
        lpc_coefficients = []
        
        for frame in frames:
            lpc_coefficients.append(librosa.lpc(y=frame, order=order))
            
        return lpc_coefficients
    
    @staticmethod
    def lpcc(lpc_list, order=12):
        # atm max lpcc order is lpc order, if lpcc oder is higher, zero filling in performed
        lpcc_coefficients = []
        
        for lpc in lpc_list:
            lpcc = np.zeros(order+1)
            lpcc[0] = lpc[0] # normally the power of the signal is used here
            lpcc[1] = lpc[1]
            
            for n in range(2, order+1):
                if n < lpc.shape[0]:
                    lpcc[n] = sum((1-k/n) * lpc[k] * lpcc[n-k] for k in range(1, n)) + lpc[n]
                else:
                    pass # there is a function in literature but it does not make any sense
            
            lpcc_coefficients.append(lpcc)
        
        return lpcc_coefficients

    @staticmethod
    def get_lpcc_from_frames(frames, order=12):
        return LPCExtractor.lpcc(lpc_list=LPCExtractor.lpc(frames=frames, order=order), order=order)