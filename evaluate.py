# evaluate.py
# Script evaluasi akurasi pengukuran citra
# Pengukuran: Tinggi Badan (ArUco + tap) & Lingkar Kepala (ArUco + ellipse)
#
# Cara pakai:
#   1. Isi data di bagian DATA_TINGGI_BADAN dan DATA_LINGKAR_KEPALA
#   2. Jalankan: python evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

import math

# =============================================================================
# ISI DATA DI SINI
# Format: (ground_truth_cm, hasil_sistem_cm)
# ground_truth = ukuran pakai alat manual (meteran / stadiometer / pita ukur)
# hasil_sistem = hasil yang muncul di aplikasi
# =============================================================================

DATA_TINGGI_BADAN = [
    # (ground_truth, sistem)
    # Contoh data — GANTI dengan data pengujianmu:
    (65.0, 64.5),
    (70.0, 70.8),
    (75.0, 74.3),
    (80.0, 80.5),
    (55.0, 55.2),
    (60.0, 59.4),
    (72.0, 72.9),
    (68.0, 67.5),
    (78.0, 78.6),
    (85.0, 84.2),
]

DATA_LINGKAR_KEPALA = [
    # (ground_truth, sistem)
    # Contoh data — GANTI dengan data pengujianmu:
    (40.0, 40.3),
    (42.0, 41.6),
    (38.0, 38.5),
    (44.0, 43.7),
    (39.0, 39.4),
    (41.0, 41.8),
    (43.0, 42.5),
    (37.0, 37.2),
    (45.0, 44.6),
    (36.0, 36.8),
]

DATA_BERAT_BADAN = [
    # (ground_truth_kg, sistem_kg)
    # Contoh data — GANTI dengan data pengujianmu:
    (5.0, 5.1),
    (6.5, 6.4),
    (7.2, 7.3),
    (8.0, 7.9),
    (4.5, 4.6),
    (9.0, 9.1),
    (6.0, 5.9),
    (7.8, 7.7),
    (8.5, 8.6),
    (5.5, 5.4),
]

# =============================================================================

def hitung_mae(data):
    return sum(abs(pred - true) for true, pred in data) / len(data)

def hitung_rmse(data):
    return math.sqrt(sum((pred - true) ** 2 for true, pred in data) / len(data))

def hitung_mape(data):
    return (sum(abs((true - pred) / true) for true, pred in data) / len(data)) * 100

def cetak_tabel(data, label, satuan="cm"):
    print(f"\n{'─'*55}")
    print(f"  {'No':<5} {'Ground Truth':>14} {'Sistem':>10} {'Selisih':>10}")
    print(f"{'─'*55}")
    for i, (true, pred) in enumerate(data, 1):
        selisih = pred - true
        tanda = "+" if selisih >= 0 else ""
        print(f"  {i:<5} {true:>12.1f} {satuan}  {pred:>8.1f} {satuan}  {tanda}{selisih:>7.1f} {satuan}")
    print(f"{'─'*55}")

def evaluasi(data, label, satuan="cm"):
    if len(data) == 0:
        print(f"\n[!] Tidak ada data untuk {label}")
        return

    mae  = hitung_mae(data)
    rmse = hitung_rmse(data)
    mape = hitung_mape(data)

    # Hitung max error dan min error
    errors = [abs(pred - true) for true, pred in data]
    max_err = max(errors)
    min_err = min(errors)

    print(f"\n{'═'*55}")
    print(f"  EVALUASI {label.upper()}")
    print(f"{'═'*55}")
    cetak_tabel(data, label, satuan)
    print(f"\n  Jumlah sampel  : {len(data)}")
    print(f"  MAE            : {mae:.4f} {satuan}")
    print(f"  RMSE           : {rmse:.4f} {satuan}")
    print(f"  MAPE           : {mape:.4f} %")
    print(f"  Error terbesar : {max_err:.2f} {satuan}")
    print(f"  Error terkecil : {min_err:.2f} {satuan}")

    # Penilaian
    print(f"\n  Penilaian:")
    if mape <= 1.0:
        grade = "SANGAT BAIK (MAPE ≤ 1%)"
    elif mape <= 2.0:
        grade = "BAIK (MAPE ≤ 2%)"
    elif mape <= 5.0:
        grade = "CUKUP (MAPE ≤ 5%)"
    else:
        grade = "PERLU PERBAIKAN (MAPE > 5%)"
    print(f"  → {grade}")
    print(f"{'═'*55}")

if __name__ == "__main__":
    print("\n" + "█"*55)
    print("  EVALUASI AKURASI PENGUKURAN CITRA - POSYLIKE V2")
    print("█"*55)

    evaluasi(DATA_TINGGI_BADAN,   "Tinggi Badan")
    evaluasi(DATA_LINGKAR_KEPALA, "Lingkar Kepala")
    evaluasi(DATA_BERAT_BADAN,    "Berat Badan", satuan="kg")

    print("\nSelesai. Salin hasil di atas ke laporan / PPT kamu.\n")
