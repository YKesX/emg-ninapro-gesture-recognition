
import os
import requests
import scipy.io

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_nina_data(file_path):
    if not os.path.exists(file_path):
        print(f"HATA: Dosya bulunamadı -> {file_path}")
        return None, None

    try:
        mat = scipy.io.loadmat(file_path)
        
        if 'emg' in mat:
            emg = mat['emg']
        else:
            return None, None
            
        if 'restimulus' in mat:
            labels = mat['restimulus']
        elif 'stimulus' in mat:
            labels = mat['stimulus']
        else:
            labels = None
            
        return emg, labels
        
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return None, None
