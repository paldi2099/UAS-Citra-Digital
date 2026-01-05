import cv2
import numpy as np
import pandas as pd
import os
from skimage import feature
from skimage.util import img_as_ubyte
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. BAGIAN HELPER FEATURES
# ==========================================

def color_features(bgr, nbins=8):
    res = {}
    chans = cv2.split(bgr)
    colors = ["B", "G", "R"]
    for i, ch in enumerate(chans):
        res[f"{colors[i]}_mean"] = float(ch.mean())
        res[f"{colors[i]}_std"] = float(ch.std())
        hist = cv2.calcHist([ch], [0], None, [nbins], [0, 256]).flatten()
        if hist.sum() > 0: hist /= hist.sum()
        for j, v in enumerate(hist):
            res[f"{colors[i]}_hist_{j}"] = float(v)
            
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    for arr, name in zip([h, s, v], ["H", "S", "V"]):
        res[f"{name}_mean"] = float(arr.mean())
        res[f"{name}_std"] = float(arr.std())
        hist = cv2.calcHist([arr], [0], None, [nbins], [0, 256]).flatten()
        if hist.sum() > 0: hist /= hist.sum()
        for j, val in enumerate(hist):
            res[f"{name}_hist_{j}"] = float(val)
    return res

def glcm_texture_features(gray, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    if gray.dtype != np.uint8:
        gray = img_as_ubyte(gray)
    try:
        glcm = feature.graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    except Exception:
        return {f"glcm_{p}": 0.0 for p in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]}

    props = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = feature.graycoprops(glcm, prop)
        props[f"glcm_{prop}"] = float(np.nanmean(vals))
    return props

def lbp_features(gray, P=8, R=1, n_bins=10):
    lbp = feature.local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    if hist.sum() > 0:
        hist = hist.astype(float) / hist.sum()
    res = {}
    for i, v in enumerate(hist):
        res[f"lbp_hist_{i}"] = float(v)
    return res

def extract_features_from_frame(img):
    # Fungsi pembungkus untuk memproses satu frame gambar
    feat = {}
    
    # 1. Color
    cf = color_features(img, nbins=8)
    feat.update(cf)
    
    # Pre-process Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. GLCM
    glcm = glcm_texture_features(gray)
    feat.update(glcm)
    
    # 3. LBP
    lbp = lbp_features(gray)
    feat.update(lbp)
    
    return feat

# ==========================================
# 2. FUNGSI TRAINING MODEL
# ==========================================
def train_model(csv_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("ERROR: File CSV tidak ditemukan. Harap jalankan kode ekstraksi fitur dulu!")
        return None, None, None

    # Pisahkan Fitur (X) dan Label (y)
    # Kita buang kolom filename jika ada, dan label sebagai target
    drop_cols = ["label", "filename"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]

    print(f"Melatih model dengan {len(df)} data gambar...")
    
    # Encode label (misal: "Parang" -> 0, "Kawung" -> 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Gunakan Random Forest (Cukup kuat untuk data tekstur/tabular)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_encoded)
    
    print("Model selesai dilatih!")
    return clf, le, X.columns # Kembalikan nama kolom agar urutan fitur konsisten

# ==========================================
# 3. FUNGSI UTAMA (WEBCAM)
# ==========================================
def run_webcam_recognition(csv_path):
    # 1. Latih Model Dulu
    model, label_encoder, feature_columns = train_model(csv_path)
    if model is None:
        return

    # 2. Buka Kamera
    cap = cv2.VideoCapture(0) # 0 biasanya ID webcam default
    if not cap.isOpened():
        print("Tidak dapat membuka kamera.")
        return

    # Ukuran kotak deteksi (ROI - Region of Interest)
    # Kita hanya akan memproses bagian tengah agar lebih akurat dan ringan
    box_size = 300 

    print("\n--- KAMERA AKTIF ---")
    print("Arahkan kain batik ke dalam kotak hijau.")
    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Mirror effect agar natural
        h, w, _ = frame.shape
        
        # Tentukan koordinat kotak di tengah layar
        y1 = (h - box_size) // 2
        x1 = (w - box_size) // 2
        y2 = y1 + box_size
        x2 = x1 + box_size

        # Ambil gambar hanya di dalam kotak (ROI)
        roi = frame[y1:y2, x1:x2]
        
        # Tampilkan kotak hijau di layar
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # PROSES PREDIKSI
        try:
            # Ekstrak fitur dari ROI
            features = extract_features_from_frame(roi)
            
            # Ubah ke DataFrame agar sesuai format training
            df_feat = pd.DataFrame([features])
            
            # Pastikan urutan kolom SAMA PERSIS dengan saat training
            # Jika ada kolom hilang (misal karena variance 0), isi dengan 0
            df_feat = df_feat.reindex(columns=feature_columns, fill_value=0)
            
            # Prediksi
            prediction_idx = model.predict(df_feat)[0]
            prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
            confidence = np.max(model.predict_proba(df_feat)) * 100
            
            # Tampilkan Teks Hasil
            text = f"{prediction_label} ({confidence:.1f}%)"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error prediksi: {e}")

        cv2.imshow("Batik Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ganti dengan path file CSV hasil training Anda yang sudah ada
    PATH_CSV = r"fitur_testing.csv"
    
    run_webcam_recognition(PATH_CSV)