import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
import os
import sys

# data_download.py dosyasından veri yükleyiciyi çağırıyoruz
# Eğer hata verirse dosya yapısına göre path eklemesi yaparız
try:
    from data_download import load_nina_data
except ImportError:
    # Eğer src klasörünün içindeysek ve çalışmıyorsa:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from src.data_download import load_nina_data
    except ImportError:
        print("UYARI: data_download modülü bulunamadı. Lütfen aynı klasörde olduğundan emin olun.")

class EMGPreprocessor:
    def __init__(self, fs=100, num_classes=13):
        self.fs = fs
        self.num_classes = num_classes

    def bandpass_filter(self, data, lowcut=20, highcut=45, order=4):
        nyquist = 0.5 * self.fs
        # Kesim frekansı Nyquist'ten büyük olamaz (Hata önleyici)
        if highcut >= nyquist:
            highcut = nyquist - 0.1
            
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=0)

    def normalize(self, data):
        # Standart Sapma (Z-score) Normalizasyonu
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-10) # 0'a bölme hatasını önlemek için 1e-10

    def filter_and_normalize(self, emg_data):
        # 1. Filtrele (Bandpass 20-45Hz)
        filtered = self.bandpass_filter(emg_data, lowcut=20, highcut=45)
        # 2. Normalize Et
        normalized = self.normalize(filtered)
        return normalized

    def segment_data(self, emg_data, labels, window_size_ms=200, overlap_percent=0.5):
        # Pencere boyutunu örnek sayısına çevir
        window_samples = int(self.fs * (window_size_ms / 1000))
        # Adım sayısını (Step) hesapla
        step_samples = int(window_samples * (1 - overlap_percent))
        
        X = []
        y = []
        
        length = len(emg_data)
        
        # Etiketler bazen (N,1) formatında gelir, düzleştirelim ve integer yapalım
        labels = labels.flatten().astype(int)
        
        for i in range(0, length - window_samples, step_samples):
            window_data = emg_data[i : i + window_samples]
            window_labels = labels[i : i + window_samples]
            
            # Pencere içindeki etiketlerin dağılımını say
            counts = np.bincount(window_labels)
            
            # En çok tekrar eden etiketi bul
            most_common_label = counts.argmax()
            
            # --- YENİ EKLENEN GÜVENLİK (PURITY) KONTROLÜ ---
            # Bu etiketin pencere içindeki oranı ne? (Örn: 200 verinin 180'i aynı mı?)
            purity = counts[most_common_label] / len(window_labels)
            
            # Eğer pencerenin %70'i aynı hareket değilse, bu bir geçiş anıdır.
            # Veriyi kirletmemek için bu pencereyi EĞİTİME ALMA (Atla).
            if purity < 0.70: 
                continue 
            # -----------------------------------------------

            # Sadece hedeflediğimiz hareketleri al (0-12 arası)
            if most_common_label < self.num_classes:
                X.append(window_data)
                y.append(most_common_label)
                
        return np.array(X), np.array(y)

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print(" Preprocess (Veri İşleme) Başlatıldı...")
    
    # 1. Klasör ve Dosya Ayarları
    input_file = "data/raw/S1_A1_E1.mat" 
    output_dir = "data/processed"
    
    # Klasör yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f" Klasör oluşturuldu: {output_dir}")

    # Dosya kontrolü (Esnek yapı)
    if not os.path.exists(input_file):
        if os.path.exists("S1_A1_E1.mat"): # Belki ana dizindedir
            input_file = "S1_A1_E1.mat"
        elif os.path.exists("/content/drive/MyDrive/S1_A1_E1.mat"): # Belki Colab Drive'dadır
            input_file = "/content/drive/MyDrive/S1_A1_E1.mat"
        else:
            print(f" HATA: '{input_file}' bulunamadı. Lütfen dosyayı indirdiğinden emin ol.")
            sys.exit()

    # 2. Veriyi Yükle
    print(f" Veri okunuyor: {input_file}")
    
    # Veriyi yükle (Sözlük döner)
    data_dict = load_nina_data(input_file)
    
    if data_dict and 'emg' in data_dict:
        emg = data_dict['emg']
        # Etiket ismini güvenli çek
        labels = data_dict.get('stimulus', data_dict.get('restimulus'))

        if labels is not None:
            # 3. İşleme Başla
            print(" Filtreleme ve Pencereleme yapılıyor...")
            
            # Sınıfı başlat
            processor = EMGPreprocessor(fs=100, num_classes=13)
            
            # Veriyi temizle
            emg_clean = processor.filter_and_normalize(emg)
            
            # Veriyi parçala (X, y oluştur)
            X, y = processor.segment_data(emg_clean, labels)
            
            print(f" İŞLEM TAMAM! Oluşan Veri Seti:")
            print(f"   X_train shape: {X.shape} (Örnek, Pencere, Kanal)")
            print(f"   y_train shape: {y.shape}")
            
            # 4. Kaydet
            np.save(os.path.join(output_dir, 'X_train.npy'), X)
            np.save(os.path.join(output_dir, 'y_train.npy'), y)
            print(f" Dosyalar '{output_dir}' klasörüne kaydedildi.")
            
        else:
            print(" UYARI: Dosyada etiket (stimulus) verisi bulunamadı.")
    else:
        print(" HATA: EMG verisi okunamadı.")

