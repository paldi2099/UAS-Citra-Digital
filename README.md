# Batik Pattern Recognition System 🇮🇩
**Projek Tugas Akhir (UAS) - Mata Kuliah Pengolahan Citra Digital**

Sistem identifikasi jenis motif batik secara real-time berbasis Computer Vision dan Machine Learning. Projek ini dikembangkan untuk memenuhi tugas ujian akhir semester yang dikerjakan oleh kelompok yang beranggotakan 5 orang.

---

## Anggota Kelompok
1. 231011403404  -  Aditya Alfianto
2. 231011400524  -  Nabil Radina
3. 231011403045  -  Raihan Fathir M.
4. 231011403152  -  Revaldi Ridwan
5. 231011400511  -  Sadam Sofyan

---

## Deskripsi Projek
Projek ini bertujuan untuk mengklasifikasikan jenis motif batik (contoh: Parang, Kawung, dll.) dengan mengekstraksi karakteristik unik dari gambar batik. Sistem bekerja dengan cara mempelajari pola warna dan tekstur dari dataset *training*, menyimpannya dalam bentuk fitur numerik, lalu melakukan prediksi pada input kamera (webcam) secara langsung.

### Metodologi Pengolahan Citra
Sistem ini menggunakan penggabungan tiga metode ekstraksi fitur utama untuk mendapatkan representasi data yang akurat:

1. **Fitur Warna (Color Features)**:
   - Mengekstraksi nilai **Mean** dan **Standard Deviation** pada ruang warna **BGR** dan **HSV**.
   - Menggunakan **Color Histogram** dengan 8 bin untuk menangkap distribusi intensitas warna pada setiap saluran.

2. **Fitur Tekstur GLCM (Gray Level Co-occurrence Matrix)**:
   - Menganalisis hubungan spasial antar piksel pada aras keabuan (grayscale).
   - Parameter yang diambil meliputi: *Contrast, Dissimilarity, Homogeneity, Energy, Correlation,* dan *ASM*.

3. **Fitur Tekstur LBP (Local Binary Pattern)**:
   - Digunakan untuk mengenali tekstur lokal yang invarian terhadap perubahan pencahayaan.
   - Fitur ini sangat efektif untuk membedakan detail halus pada motif batik yang repetitif.

### Klasifikasi
Fitur-fitur yang telah diekstrak kemudian diklasifikasikan menggunakan algoritma **Random Forest Classifier**. Model ini dipilih karena kemampuannya menangani data tabular hasil ekstraksi fitur dengan tingkat akurasi yang stabil.

---

## Komponen Program
Projek ini terdiri dari dua modul utama:

* **`fiturbatik.py` (Feature Extraction)**: 
    Script ini berfungsi untuk memproses seluruh dataset gambar di folder *training* dan *testing*. Hasil ekstraksi fitur (warna, GLCM, dan LBP) akan disimpan ke dalam file format `.csv` sebagai basis data pengetahuan model.

* **`runbatik.py` (Recognition & Training)**: 
    Script utama yang melatih model Random Forest menggunakan data CSV. Script ini mengaktifkan webcam dan menggunakan **Region of Interest (ROI)** berbentuk kotak di tengah layar untuk mendeteksi kain batik secara real-time.

---

## Alur Kerja Sistem
1. **Preprocessing**: Gambar diubah ke skala keabuan untuk tekstur dan ruang warna HSV untuk analisis warna.
2. **Feature Extraction**: Menghitung karakteristik matematis dari gambar batik.
3. **Training**: Model mempelajari pola dari file CSV yang dihasilkan.
4. **Real-time Detection**: Kamera menangkap frame, mengekstrak fitur secara instan, dan menampilkan label prediksi serta tingkat kepercayaan (*confidence*) di layar.
