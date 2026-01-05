

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage.util import img_as_ubyte
from tqdm import tqdm

# -----------------------------
# Helper feature functions
# -----------------------------
def read_image(path):
    # Membaca gambar, mengatasi error path Windows
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = img_as_ubyte(img)
    return img

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

def extract_features_for_image(path):
    img = read_image(path)
    if img is None:
        raise IOError(f"Cannot read image: {path}")

    feat = {}
    feat["filename"] = str(path)
    feat["label"] = Path(path).parent.name 
    
    cf = color_features(img, nbins=8)
    feat.update(cf)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    glcm = glcm_texture_features(gray)
    feat.update(glcm)
    
    lbp = lbp_features(gray)
    feat.update(lbp)
    
    return feat

def process_dataset(root_dir, out_csv, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    root = Path(root_dir)
    rows = []
    
    print(f"Mencari gambar di folder: {root}")
    image_paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    print(f"Ditemukan {len(image_paths)} gambar.")
    
    if len(image_paths) == 0:
        print("GAGAL: Tidak ada gambar ditemukan. Cek kembali path folder Anda.")
        return None

    for p in tqdm(image_paths):
        try:
            feat = extract_features_for_image(p)
            rows.append(feat)
        except Exception as e:
            print(f"Error pada file {p}: {e}")
            
    if not rows:
        print("Tidak ada fitur yang diekstrak.")
        return None
        
    df = pd.DataFrame(rows)
    cols = list(df.columns)
    if "filename" in cols: cols.remove("filename")
    if "label" in cols: cols.remove("label")
    final_cols = ["filename"] + cols + ["label"]
    df = df[final_cols]
    
    df.to_csv(out_csv, index=False)
    print(f"SUKSES! Data fitur disimpan ke: {out_csv}")
    return df



if __name__ == "__main__":
    # ==========================================
    # PENGATURAN FOLDER (EDIT BAGIAN INI)
    # ==========================================
    
    # Ganti path di bawah sesuai folder yang mau diproses ('training' atau 'testing')
    # Gunakan r"..." agar path Windows terbaca benar
    
    FOLDER_INPUT = r"d:\batik\testing"   
    FILE_OUTPUT  = r"fitur_testing.csv"
    
    # ==========================================
    
    print("--- Memulai Program Ekstraksi Fitur Batik ---")
    
    # Cek apakah folder ada
    if not os.path.exists(FOLDER_INPUT):
        print(f"ERROR: Folder tidak ditemukan: {FOLDER_INPUT}")
        print("Pastikan Anda sudah mengedit bagian FOLDER_INPUT di dalam kodingan.")
    else:
        process_dataset(FOLDER_INPUT, FILE_OUTPUT)