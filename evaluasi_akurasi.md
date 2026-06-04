# Evaluasi Akurasi Pengukuran Citra
## Sistem Pengukuran Antropometri Bayi Berbasis Kamera
### Tinggi Badan & Lingkar Kepala — Posylike V2

---

## SLIDE 1 — Metode Pengukuran Tinggi Badan

**Alur Pengukuran:**
1. Foto bayi dari **samping** menggunakan kamera HP
2. Sistem deteksi **ArUco Marker** (10×10 cm, DICT_4X4_50) sebagai referensi skala
3. Hitung kalibrasi skala:

$$pixels\_per\_cm = \frac{ukuran\_marker_{px}}{10\ cm}$$

4. Pengguna tap titik **kepala** dan **kaki** di layar
5. Hitung tinggi badan:

$$tinggi = \frac{\sqrt{(x_{kaki}-x_{kepala})^2 + (y_{kaki}-y_{kepala})^2}}{pixels\_per\_cm}$$

---

## SLIDE 2 — Metode Pengukuran Lingkar Kepala

**Alur Pengukuran:**
1. Foto kepala bayi dari **atas (top-down)** menggunakan kamera HP
2. Deteksi ArUco Marker → kalibrasi skala
3. **Segmentasi kulit** di ruang warna YCrCb
4. Kontur terbesar = kepala → **Fitting Ellipse** → Rumus Ramanujan:

$$C \approx \pi(a+b)\left[1 + \frac{3h}{10+\sqrt{4-3h}}\right], \quad h = \frac{(a-b)^2}{(a+b)^2}$$

- $a$ = semi-axis panjang ellipse (cm)
- $b$ = semi-axis pendek ellipse (cm)

---

## SLIDE 3 — Metrik Evaluasi Akurasi

| Metrik | Rumus | Keterangan |
|--------|-------|------------|
| MAE | $\frac{1}{n}\sum\|y_{pred}-y_{true}\|$ | Rata-rata selisih (cm) |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_{pred}-y_{true})^2}$ | Sensitif terhadap outlier |
| MAPE | $\frac{1}{n}\sum\left\|\frac{y_{true}-y_{pred}}{y_{true}}\right\|\times 100\%$ | Persentase error |

**Standar diterima untuk posyandu:** MAE ≤ 1 cm, MAPE ≤ 2%

---

## SLIDE 4 — Prosedur Pengujian

1. Ukur tinggi & lingkar kepala bayi menggunakan **alat manual** (stadiometer / pita ukur) → catat sebagai *ground truth*
2. Ukur bayi yang **sama** menggunakan aplikasi Posylike → catat hasil sistem
3. Lakukan pada **10 sampel** (anak berbeda)
4. Hitung MAE, RMSE, MAPE menggunakan script `evaluate.py`

---

## SLIDE 5 — Hasil Evaluasi: Tinggi Badan *(Data Contoh)*

> ⚠️ Data di bawah adalah **contoh** — ganti dengan hasil pengujian nyata

| No | Ground Truth | Sistem | Selisih |
|----|-------------|--------|---------|
| 1  | 65.0 cm | 64.5 cm | -0.5 cm |
| 2  | 70.0 cm | 70.8 cm | +0.8 cm |
| 3  | 75.0 cm | 74.3 cm | -0.7 cm |
| 4  | 80.0 cm | 80.5 cm | +0.5 cm |
| 5  | 55.0 cm | 55.2 cm | +0.2 cm |
| 6  | 60.0 cm | 59.4 cm | -0.6 cm |
| 7  | 72.0 cm | 72.9 cm | +0.9 cm |
| 8  | 68.0 cm | 67.5 cm | -0.5 cm |
| 9  | 78.0 cm | 78.6 cm | +0.6 cm |
| 10 | 85.0 cm | 84.2 cm | -0.8 cm |

| Metrik | Nilai |
|--------|-------|
| Jumlah Sampel | 10 |
| MAE | **0.61 cm** |
| RMSE | **0.64 cm** |
| MAPE | **0.85%** |
| Error Terbesar | 0.90 cm |
| Error Terkecil | 0.20 cm |
| **Penilaian** | ✅ **SANGAT BAIK (MAPE ≤ 1%)** |

---

## SLIDE 6 — Hasil Evaluasi: Lingkar Kepala *(Data Contoh)*

> ⚠️ Data di bawah adalah **contoh** — ganti dengan hasil pengujian nyata

| No | Ground Truth | Sistem | Selisih |
|----|-------------|--------|---------|
| 1  | 40.0 cm | 40.3 cm | +0.3 cm |
| 2  | 42.0 cm | 41.6 cm | -0.4 cm |
| 3  | 38.0 cm | 38.5 cm | +0.5 cm |
| 4  | 44.0 cm | 43.7 cm | -0.3 cm |
| 5  | 39.0 cm | 39.4 cm | +0.4 cm |
| 6  | 41.0 cm | 41.8 cm | +0.8 cm |
| 7  | 43.0 cm | 42.5 cm | -0.5 cm |
| 8  | 37.0 cm | 37.2 cm | +0.2 cm |
| 9  | 45.0 cm | 44.6 cm | -0.4 cm |
| 10 | 36.0 cm | 36.8 cm | +0.8 cm |

| Metrik | Nilai |
|--------|-------|
| Jumlah Sampel | 10 |
| MAE | **0.46 cm** |
| RMSE | **0.50 cm** |
| MAPE | **1.15%** |
| Error Terbesar | 0.80 cm |
| Error Terkecil | 0.20 cm |
| **Penilaian** | ✅ **BAIK (MAPE ≤ 2%)** |

---

## SLIDE 7 — Faktor yang Mempengaruhi Akurasi

| Faktor | Tinggi Badan | Lingkar Kepala |
|--------|-------------|----------------|
| Deteksi ArUco gagal | Tidak bisa hitung skala | Tidak bisa hitung skala |
| Sudut kamera | Harus dari samping lurus | Harus dari tepat atas |
| Pencahayaan | Marker harus terlihat jelas | Mempengaruhi segmentasi kulit |
| Ketepatan tap titik | Error manual saat tap | Tidak relevan (otomatis) |
| Resolusi kamera | Semakin tinggi, lebih akurat | Semakin tinggi, lebih akurat |

---

## SLIDE 8 — Kesimpulan

| Parameter | MAE | RMSE | MAPE | Kategori |
|-----------|-----|------|------|----------|
| Tinggi Badan | 0.61 cm | 0.64 cm | 0.85% | ✅ Sangat Baik |
| Lingkar Kepala | 0.46 cm | 0.50 cm | 1.15% | ✅ Baik |

- Kedua pengukuran memenuhi standar akurasi posyandu (MAPE ≤ 2%)
- Sistem kamera ArUco layak digunakan sebagai alternatif pengukuran
- Data akan diperbarui setelah pengujian langsung dilakukan

---

*Catatan: Jalankan `python backend/evaluate.py` untuk menghitung ulang metrik setelah data pengujian nyata dimasukkan.*
