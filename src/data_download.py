import scipy.io
import os
import sys

def load_nina_data(file_path):
    """
    Belirtilen yoldaki .mat dosyasini scipy.io ile yukler.
    
    Args:
        file_path (str): .mat dosyasinin tam yolu.
        
    Returns:
        dict: 'emg' ve 'stimulus' verilerini iceren sozluk.
    """
    if not os.path.exists(file_path):
        print(f"HATA: Dosya bulunamadi -> {file_path}")
        return None

    try:
        print(f"Dosya okunuyor: {file_path} ...")
        mat = scipy.io.loadmat(file_path)
        
        data = {}
        
        # EMG verisini al
        if 'emg' in mat:
            data['emg'] = mat['emg']
            print(f"Basarili! EMG Verisi Boyutu: {data['emg'].shape}")
        else:
            print("UYARI: Dosya icinde 'emg' anahtari bulunamadi.")
            
        # Etiket verisini al
        if 'restimulus' in mat:
            data['stimulus'] = mat['restimulus']
            print(f"Basarili! Etiket (restimulus) Boyutu: {data['stimulus'].shape}")
        elif 'stimulus' in mat:
            data['stimulus'] = mat['stimulus']
            print(f"Basarili! Etiket (stimulus) Boyutu: {data['stimulus'].shape}")
        else:
            data['stimulus'] = None
            print("Bilgi: Etiket verisi bulunamadi.")
            
        return data

    except Exception as e:
        print(f"Veri yukleme hatasi: {e}")
        return None


if __name__ == "__main__":
    # Bu blok sadece dosya dogrudan calistirildiginda devreye girer.
    print("--- NinaPro Veri Yukleyici Test Modu ---")
    
    # Google Colab ortaminda miyiz kontrol eder
    try:
        from google.colab import drive
        print("Google Colab algilandi. Drive baglaniyor...")
        drive.mount('/content/drive')
        
        # Drive icindeki olasi dosya yollari (S1_A1_E1)
        test_paths = [
            '/content/drive/MyDrive/S1_A1_E1.mat',
            '/content/drive/MyDrive/Ninapro_DB1.csv'
        ]
        
        found = False
        for path in test_paths:
            if os.path.exists(path):
                print(f"Test dosyasi bulundu: {path}")
                load_nina_data(path)
                found = True
                break
        
        if not found:
            print("HATA: Test edilecek dosya Drive ana dizininde bulunamadi.")
            print("Lutfen 'S1_A1_E1.mat' dosyasini Drive'iniza yukleyin.")

    except ImportError:
        # Colab degilse (Lokal bilgisayar)
        print("Lokal ortam algilandi.")
        test_file = "S1_A1_E1.mat" # Ayni klasorde var sayiyoruz
        if os.path.exists(test_file):
            load_nina_data(test_file)
        else:
            print(f"Test icin '{test_file}' dosyasi bulunamadi.")
