
import numpy as np
from scipy.signal import butter, lfilter

class EMGPreprocessor:
    def __init__(self, fs=100, num_classes=13):
        self.fs = fs
        self.num_classes = num_classes

    def bandpass_filter(self, data, lowcut=20, highcut=45, order=4):
        nyquist = 0.5 * self.fs
        if highcut >= nyquist:
            highcut = nyquist - 0.1
            
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=0)

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-10)

    def filter_and_normalize(self, emg_data):
        filtered = self.bandpass_filter(emg_data, lowcut=20, highcut=45)
        normalized = self.normalize(filtered)
        return normalized

    def segment_data(self, emg_data, labels, window_size_ms=200, overlap_percent=0.5):
        window_samples = int(self.fs * (window_size_ms / 1000))
        step_samples = int(window_samples * (1 - overlap_percent))
        
        X = []
        y = []
        
        length = len(emg_data)
        
        for i in range(0, length - window_samples, step_samples):
            window_data = emg_data[i : i + window_samples]
            window_labels = labels[i : i + window_samples]
            
            most_common_label = np.bincount(window_labels.flatten()).argmax()
            
            if most_common_label < self.num_classes:
                X.append(window_data)
                y.append(most_common_label)
                
        return np.array(X), np.array(y)
