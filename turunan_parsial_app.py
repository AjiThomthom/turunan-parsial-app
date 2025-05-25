# Mengimport semua library yang dibutukan untuk menjalankan aplikasi ini
import streamlit as st # Library untuk membuat aplikasi web interaktif
import sympy as sp    # Library untuk matematika simbolik (kalkulus, aljabar)
import numpy as np    # Library untuk operasi numerik pada array (digunakan untuk grid plot)
import matplotlib.pyplot as plt # Library untuk membuat plot dan grafik

# Pengaturan Halaman Aplikasi Streamlit
st.set_page_config(layout="wide")

# Judul Utama Aplikasi
st.title("Aplikasi Kalkulator & Visualisasi Turunan Parsial")

# Bagian 1: Input Fungsi Matematika
st.header("1. Masukkan Fungsi f(x, y)")
fungsi_str = st.text_input(
    "Masukkan fungsi f(x, y) (gunakan 'x' dan 'y' sebagai variabel, '**' untuk pangkat, '*' untuk perkalian):",
    "x**2 + y**3"
)

# Mendefinisikan simbol matematika 'x' dan 'y' menggunakan SymPy agar dapat diproses secara simbolik.
x, y = sp.symbols('x y')

# Penanganan Error (Try-Except Block)
try:
    # Mengubah string input dari pengguna menjadi objek ekspresi matematika SymPy yang bisa dihitung.
    f = sp.sympify(fungsi_str)

    # Validasi: Memastikan fungsi yang dimasukkan hanya mengandung variabel 'x' dan 'y'.
    if not f.free_symbols.issubset({x, y}):
        st.error("Error: Fungsi hanya boleh mengandung variabel 'x' dan 'y'.")
    else:
        # Bagian 2: Perhitungan Turunan Parsial
        # Menghitung turunan parsial dari fungsi 'f' terhadap 'x' dan 'y' menggunakan SymPy.
        fx = sp.diff(f, x)
        fy = sp.diff(f, y)

        st.header("2. Hasil Turunan Parsial")
        # Menampilkan fungsi asli dan hasil turunan parsial dalam format LaTeX untuk tampilan yang rapi.
        st.latex(f"f(x, y) = {sp.latex(f)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(fx)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(fy)}")

        # Bagian 3: Evaluasi Fungsi dan Gradien di Titik Tertentu
        # Pengguna memasukkan nilai x₀ dan y₀ untuk evaluasi.
        st.header("3. Evaluasi di Titik (x₀, y₀)")
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input("Nilai x₀:", value=1.0, key="x0_input")
        with col2:
            y0 = st.number_input("Nilai y₀:", value=2.0, key="y0_input")

        # Mengevaluasi nilai fungsi dan kedua turunan parsialnya (fx dan fy) di titik (x₀, y₀).
        f_val = f.subs({x: x0, y: y0})
        fx_val = fx.subs({x: x0, y: y0})
        fy_val = fy.subs({x: x0, y: y0})

        # Menampilkan hasil evaluasi dan vektor gradien di titik tersebut.
        st.write(f"Nilai fungsi $f({x0}, {y0})$: **{f_val:.4f}**")
        st.write(f"Gradien di titik $({x0}, {y0})$: $ \\nabla f = \\left( {fx_val:.4f}, {fy_val:.4f} \\right) $")

        # Bagian 4: Visualisasi Permukaan 3D dan Bidang Singgung
        # Membuat plot 3D dari permukaan fungsi f(x, y) dan bidang singgungnya di titik (x₀, y₀).
        st.header("4. Visualisasi Permukaan & Bidang Singgung")

        # Menyiapkan grid nilai x dan y untuk plot, terpusat di sekitar (x₀, y₀).
        plot_range = 2
        x_vals = np.linspace(x0 - plot_range, x0 + plot_range, 50)
        y_vals = np.linspace(y0 - plot_range, y0 + plot_range, 50)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Mengkonversi ekspresi SymPy menjadi fungsi yang dapat dievaluasi oleh NumPy untuk plotting.
        Z_func = sp.lambdify((x, y), f, 'sympy')
        Z = Z_func(X, Y)

        # Mengkonversi nilai-nilai evaluasi menjadi float untuk operasi NumPy.
        f_val_float = float(f_val)
        fx_val_float = float(fx_val)
        fy_val_float = float(fy_val)

        # Menghitung nilai Z untuk bidang singgung menggunakan rumus:
        # Z_tangent = f(x₀,y₀) + fx(x₀,y₀)*(x - x₀) + fy(x₀,y₀)*(y - y₀)
        Z_tangent = f_val_float + fx_val_float * (X - x0) + fy_val_float * (Y - y0)

        # Membuat objek figure dan axes untuk plot 3D menggunakan Matplotlib.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Memplot permukaan fungsi f(x, y).
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')

        # Memplot bidang singgung (ditampilkan dengan warna merah).
        ax.plot_surface(X, Y, Z_tangent, alpha=0.5, color='red', label='Bidang Singgung')

        # Menambahkan penanda titik (x₀, y₀, f(x₀, y₀)) di permukaan.
        ax.scatter([x0], [y0], [f_val_float], color='blue', s=100, label='Titik Evaluasi', depthshade=True)

        # Menambahkan judul, label sumbu, legenda, dan mengatur sudut pandang awal plot.
        ax.set_title("Permukaan f(x, y) dan Bidang Singgung")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        ax.view_init(elev=30, azim=45)

        # Menampilkan plot yang sudah dibuat di aplikasi Streamlit.
        st.pyplot(fig)

# Penanganan Kesalahan Umum
# Menampilkan pesan kesalahan jika ada exception yang terjadi di blok try,
# membantu pengguna memahami apa yang salah.
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}. Pastikan format fungsi benar dan hanya menggunakan variabel 'x' dan 'y'. Contoh: x**2 + y**3")
