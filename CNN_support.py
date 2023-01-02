import os
import sys
import librosa
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import normalize

class SoundPreprocessing:
    """
    Parameters
    ----------
    
    sr (int): sampling rate
    max_size (iterable): resulting shape of the tensor
    n_fft (int): number related to FFT
    n_mfcc (int): number of MFCC
    
    """
    
    
    def __init__(self, *, sr, max_size, n_fft, n_mfcc = 60, hop_length = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.max_size = max_size
        self.hop_length = hop_length
        
        
    def padding(self, array, xx, yy):
        """
        Parameters
        ----------
            array: numpy array
            xx: desired height
            yy: desirex width
        
        Returns: padded array
        """
        self.array = array
        self.xx = xx
        self.yy = yy
        
        h = array.shape[0]
        w = array.shape[1]
        a = max((xx - h) // 2,0)
        aa = max(0,xx - a - h)
        b = max(0,(yy - w) // 2)
        bb = max(yy - b - w,0)

        return np.pad(array, pad_width = ((a, aa), (b, bb)),
                      mode = "constant")
    
    
    def generate_features(self, y_cut, sr, max_size, n_fft, n_mfcc, hop_length):
        self.y_cut = y_cut
        
        # Numeri -2 divisibili per 14
        condition = np.arange(2, 1000)[np.where((np.arange(2, 1000) - 2)%14 == 0)]
        
        global shape_changed
        shape_changed = False
        
        if max_size[0] not in condition:
            # Get closest number to 'max_size' that respects 'condition'
            new_max0 = sorted(condition, key = lambda v: abs(v - max_size[0]))[0]
            shape_changed = True
            max_size = (new_max0, max_size[1])
        
        stft = self.padding(np.abs(librosa.stft(y = y_cut, n_fft = n_fft,
                                   hop_length = 512)), max_size[0], max_size[1])
        
        if max_size[0] < stft.shape[0]:
            new_max0 = sorted(condition[condition >= stft.shape[0]],
                              key = lambda v: abs(v - stft.shape[0]))[0]
            max_size = (new_max0, max_size[1])
            shape_changed = True
        
        stft = self.padding(np.abs(librosa.stft(y = y_cut, n_fft = n_fft,
                                   hop_length = 512)), max_size[0], max_size[1])
        
        MFCCs = self.padding(librosa.feature.mfcc(y = y_cut, n_fft = n_fft, sr = sr,
                                                  hop_length = hop_length, n_mfcc = n_mfcc),
                             max_size[0], max_size[1])
        
        spec_centroid = librosa.feature.spectral_centroid(y = y_cut, sr = sr)
        chroma_stft = librosa.feature.chroma_stft(y = y_cut, sr = sr)
        spec_bw = librosa.feature.spectral_bandwidth(y = y_cut, sr = sr)
    
        #Now the padding part
        image = np.array([self.padding(normalize(spec_bw), 1, max_size[1])]).reshape(1, max_size[1])
        image = np.append(image, self.padding(normalize(spec_centroid), 1, max_size[1]), axis = 0)
        
        #repeat the padded spec_bw,spec_centroid and chroma stft until they are stft and MFCC-sized        
        for i in range( int((max_size[0]-2)/14) ):
            image = np.append(image, self.padding(normalize(spec_bw), 1, max_size[1]), axis = 0)
            image = np.append(image, self.padding(normalize(spec_centroid), 1, max_size[1]), axis = 0)
            image = np.append(image, self.padding(normalize(chroma_stft), 12, max_size[1]), axis = 0)
        
        image = np.dstack((image, np.abs(stft)))
        image = np.dstack((image, MFCCs))
        
        return image

    
    def get_features(self, df, filepath):
        self.df = df
        self.filepath = filepath
        
        # Get data for CNN
        X = []
        y = np.zeros(shape = (len(df), 1))

        for i in df.index:

            sr_i, aud = wavfile.read("{}\\{}".format(filepath, df.loc[i, "filename"]))
            aud = aud.astype(np.float16)
            
            X += [self.generate_features(y_cut = aud, sr = sr_i,
                                         n_fft = self.n_fft,
                                         n_mfcc = self.n_mfcc,
                                         max_size = self.max_size,
                                         hop_length = self.hop_length)]

            y[i] = df.loc[i, "target"]
        
        
        if shape_changed == True:
            print(f"New max_size is {max_size}")
            
        X = np.array(X)
        
        return X, y
        
        
        
        
        
        
