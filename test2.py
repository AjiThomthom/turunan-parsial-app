
import streamlit as st
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Aplikasi Matematika Terapan", layout="centered")

st.title("📊 Aplikasi Matematika Terapan di Dunia Teknik")

# Tabs utama
tab1, tab2, tab3 = st.tabs(["📈 Antrian M/M/1", "🏭 Optimasi Produksi (LP)", "📦 Model Persediaan (EOQ)"])

# ========================
# Tab 1: Antrian M/M/1
# ========================
with tab1:
    st.header("Model Antrian M/M/1")

    λ = st.number_input("Laju kedatangan (λ)", min_value=0.1, value=4.0)
    μ = st.number_input("Laju pelayanan (μ)", min_value=0.1, value=6.0)

    if λ >= μ:
        st.error("Laju kedatangan harus lebih kecil dari laju pelayanan agar sistem stabil.")
    else:
        ρ = λ / μ
        L = λ / (μ - λ)
        W = 1 / (μ - λ)
        P0 = 1 - ρ

        st.write(f"**Utilisasi server (ρ)**: {ρ:.2f}")
        st.write(f"**Rata-rata jumlah dalam sistem (L)**: {L:.2f}")
        st.write(f"**Rata-rata waktu dalam sistem (W)**: {W:.2f} jam")
        st.write(f"**Probabilitas sistem kosong (P₀)**: {P0:.2f}")

# ========================
# Tab 2: Optimasi Produksi (LP)
# ========================
with tab2:
    st.header("Optimasi Produksi (Linear Programming)")

    st.write("Kasus: Sebuah pabrik memproduksi dua jenis produk A dan B. Setiap produk menggunakan dua jenis sumber daya terbatas.")

    c1 = st.number_input("Keuntungan per unit produk A", value=40)
    c2 = st.number_input("Keuntungan per unit produk B", value=30)

    a1 = st.number_input("Jam kerja mesin per unit A", value=2)
    b1 = st.number_input("Jam kerja mesin per unit B", value=1)
    mesin = st.number_input("Total jam kerja mesin tersedia", value=100)

    a2 = st.number_input("Bahan baku per unit A", value=1)
    b2 = st.number_input("Bahan baku per unit B", value=1)
    bahan = st.number_input("Total bahan baku tersedia", value=80)

    if st.button("Hitung Optimasi"):
        c = [-c1, -c2]  # fungsi objektif negatif (karena minimisasi)
        A = [[a1, b1], [a2, b2]]
        b = [mesin, bahan]
        bounds = [(0, None), (0, None)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if res.success:
            st.success("Solusi ditemukan!")
            st.write(f"Produksi optimal produk A: {res.x[0]:.2f} unit")
            st.write(f"Produksi optimal produk B: {res.x[1]:.2f} unit")
            st.write(f"Total keuntungan maksimum: {-res.fun:.2f}")
        else:
            st.error("Tidak ditemukan solusi optimal.")

# ========================
# Tab 3: Model Persediaan (EOQ)
# ========================
with tab3:
    st.header("Model Persediaan EOQ (Economic Order Quantity)")

    D = st.number_input("Permintaan tahunan (D)", value=1200)
    S = st.number_input("Biaya pemesanan per pesanan (S)", value=50)
    H = st.number_input("Biaya penyimpanan per unit per tahun (H)", value=2)

    if D > 0 and S > 0 and H > 0:
        EOQ = np.sqrt((2 * D * S) / H)
        TBO = EOQ / D  # waktu antara pemesanan (tahun)

        st.write(f"**Jumlah pemesanan ekonomis (EOQ)**: {EOQ:.2f} unit")
        st.write(f"**Waktu antar pemesanan (TBO)**: {TBO:.2f} tahun")
    else:
        st.warning("Masukkan semua nilai yang lebih dari nol.")
