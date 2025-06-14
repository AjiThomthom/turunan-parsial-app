import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# Judul aplikasi
st.title("Aplikasi Model Matematika Industri")
st.write("""
Aplikasi ini mencakup implementasi berbagai model matematika yang digunakan dalam industri:
1. Optimasi Produksi (Linear Programming)
2. Model Persediaan (EOQ)
3. Model Antrian (M/M/1)
4. Model Pertumbuhan Eksponensial
""")

# Sidebar untuk navigasi
menu = st.sidebar.selectbox(
    "Pilih Model:",
    ["Optimasi Produksi", "Model Persediaan", "Model Antrian", "Model Pertumbuhan Eksponensial"]
)

# Tab 1: Optimasi Produksi
if menu == "Optimasi Produksi":
    st.header("Optimasi Produksi (Linear Programming)")
    
    st.subheader("Masukkan Parameter")
    col1, col2 = st.columns(2)
    
    with col1:
        profit_a = st.number_input("Keuntungan per Unit Produk A (Rp)", min_value=0, value=40000)
        time_a = st.number_input("Waktu Mesin per Unit Produk A (jam)", min_value=0.0, value=2.0)
    
    with col2:
        profit_b = st.number_input("Keuntungan per Unit Produk B (Rp)", min_value=0, value=60000)
        time_b = st.number_input("Waktu Mesin per Unit Produk B (jam)", min_value=0.0, value=3.0)
    
    total_time = st.number_input("Total Waktu Mesin Tersedia (jam)", min_value=0.0, value=100.0)
    
    # Hitung solusi optimal
    if st.button("Hitung Solusi Optimal"):
        # Hitung titik potong
        x_max = total_time / time_a
        y_max = total_time / time_b
        
        # Hitung keuntungan di setiap titik ekstrim
        profit_x = profit_a * x_max
        profit_y = profit_b * y_max
        
        # Tentukan solusi optimal
        if profit_x > profit_y:
            optimal_x = x_max
            optimal_y = 0
            max_profit = profit_x
        else:
            optimal_x = 0
            optimal_y = y_max
            max_profit = profit_y
        
        # Tampilkan hasil
        st.subheader("Hasil Optimasi")
        st.write(f"Solusi Optimal: Produksi {optimal_x:.2f} unit Produk A dan {optimal_y:.2f} unit Produk B")
        st.write(f"Keuntungan Maksimal: Rp{max_profit:,.2f}")
        
        # Visualisasi
        fig, ax = plt.subplots()
        x = np.linspace(0, x_max, 100)
        y = (total_time - time_a * x) / time_b
        
        ax.plot(x, y, label='Kendala Waktu Mesin')
        ax.fill_between(x, 0, y, alpha=0.1)
        ax.scatter([optimal_x], [optimal_y], color='red', label='Solusi Optimal')
        
        ax.set_xlabel('Jumlah Produk A')
        ax.set_ylabel('Jumlah Produk B')
        ax.set_title('Daerah Solusi Layak')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)

# Tab 2: Model Persediaan
elif menu == "Model Persediaan":
    st.header("Model Persediaan (Economic Order Quantity - EOQ)")
    
    st.subheader("Masukkan Parameter")
    D = st.number_input("Permintaan Tahunan (unit)", min_value=1, value=10000)
    S = st.number_input("Biaya Pemesanan per Pesanan (Rp)", min_value=1, value=50000)
    H = st.number_input("Biaya Penyimpanan per Unit per Tahun (Rp)", min_value=1, value=2000)
    
    if st.button("Hitung EOQ"):
        eoq = math.sqrt((2 * D * S) / H)
        total_orders = D / eoq
        total_holding_cost = (eoq / 2) * H
        total_ordering_cost = (D / eoq) * S
        total_cost = total_holding_cost + total_ordering_cost
        
        st.subheader("Hasil Perhitungan EOQ")
        st.write(f"Jumlah Pemesanan Optimal (EOQ): {eoq:.2f} unit")
        st.write(f"Jumlah Pemesanan per Tahun: {total_orders:.2f} kali")
        st.write(f"Total Biaya Penyimpanan: Rp{total_holding_cost:,.2f}")
        st.write(f"Total Biaya Pemesanan: Rp{total_ordering_cost:,.2f}")
        st.write(f"Total Biaya Persediaan Tahunan: Rp{total_cost:,.2f}")
        
        # Visualisasi hubungan EOQ dan biaya
        q_values = np.linspace(100, 2*eoq, 50)
        holding_costs = (q_values / 2) * H
        ordering_costs = (D / q_values) * S
        total_costs = holding_costs + ordering_costs
        
        fig, ax = plt.subplots()
        ax.plot(q_values, holding_costs, label='Biaya Penyimpanan')
        ax.plot(q_values, ordering_costs, label='Biaya Pemesanan')
        ax.plot(q_values, total_costs, label='Total Biaya')
        ax.axvline(eoq, color='red', linestyle='--', label='EOQ')
        
        ax.set_xlabel('Jumlah Pesanan (Q)')
        ax.set_ylabel('Biaya (Rp)')
        ax.set_title('Hubungan Jumlah Pesanan dan Biaya')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)

# Tab 3: Model Antrian
elif menu == "Model Antrian":
    st.header("Model Antrian (M/M/1)")
    
    st.subheader("Masukkan Parameter")
    col1, col2 = st.columns(2)
    
    with col1:
        arrival_rate = st.number_input("Tingkat Kedatangan (λ, pelanggan/jam)", min_value=0.1, value=10.0)
    
    with col2:
        service_rate = st.number_input("Tingkat Pelayanan (μ, pelanggan/jam)", min_value=0.1, value=12.0)
    
    if arrival_rate >= service_rate:
        st.warning("Warning: Tingkat kedatangan harus lebih kecil dari tingkat pelayanan untuk sistem stabil.")
    
    if st.button("Hitung Parameter Antrian"):
        rho = arrival_rate / service_rate
        L = arrival_rate / (service_rate - arrival_rate)
        W = 1 / (service_rate - arrival_rate)
        Wq = arrival_rate / (service_rate * (service_rate - arrival_rate))
        Lq = arrival_rate * Wq
        
        st.subheader("Hasil Perhitungan Antrian")
        st.write(f"Tingkat Utilisasi Server (ρ): {rho:.2%}")
        st.write(f"Rata-rata Jumlah Pelanggan dalam Sistem (L): {L:.2f}")
        st.write(f"Rata-rata Waktu dalam Sistem (W): {W:.2f} jam ({W*60:.0f} menit)")
        st.write(f"Rata-rata Waktu Menunggu dalam Antrian (Wq): {Wq:.2f} jam ({Wq*60:.0f} menit)")
        st.write(f"Rata-rata Jumlah Pelanggan dalam Antrian (Lq): {Lq:.2f}")
        
        # Visualisasi distribusi
        time_points = np.linspace(0, 2, 100)
        prob_system = 1 - np.exp(-(service_rate - arrival_rate) * time_points)
        
        fig, ax = plt.subplots()
        ax.plot(time_points, prob_system, label='Probabilitas Sistem')
        ax.axvline(W, color='red', linestyle='--', label='Waktu Rata-rata dalam Sistem')
        
        ax.set_xlabel('Waktu (jam)')
        ax.set_ylabel('Probabilitas')
        ax.set_title('Distribusi Waktu dalam Sistem')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)

# Tab 4: Model Pertumbuhan Eksponensial
else:
    st.header("Model Pertumbuhan Eksponensial")
    
    st.subheader("Masukkan Parameter")
    col1, col2 = st.columns(2)
    
    with col1:
        initial_value = st.number_input("Nilai Awal", min_value=1.0, value=100.0)
        growth_rate = st.number_input("Tingkat Pertumbuhan (% per periode)", min_value=0.1, value=5.0) / 100
    
    with col2:
        periods = st.number_input("Jumlah Periode", min_value=1, value=10)
        time_unit = st.selectbox("Satuan Waktu", ["hari", "minggu", "bulan", "tahun"])
    
    if st.button("Proyeksi Pertumbuhan"):
        time_points = np.arange(0, periods + 1)
        values = initial_value * np.exp(growth_rate * time_points)
        
        st.subheader("Hasil Proyeksi")
        st.write(f"Nilai awal: {initial_value}")
        st.write(f"Nilai setelah {periods} {time_unit}: {values[-1]:.2f}")
        
        # Visualisasi pertumbuhan
        fig, ax = plt.subplots()
        ax.plot(time_points, values, marker='o')
        
        ax.set_xlabel(f'Waktu ({time_unit})')
        ax.set_ylabel('Nilai')
        ax.set_title('Pertumbuhan Eksponensial')
        ax.grid(True)
        
        st.pyplot(fig)

# Informasi tambahan di sidebar
st.sidebar.markdown("""
**Informasi Aplikasi:**
- Dibuat untuk memenuhi tugas mata kuliah Matematika Terapan
- Framework: Streamlit
- Bahasa: Python
""")
