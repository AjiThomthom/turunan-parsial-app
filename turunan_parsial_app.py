import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide") # Memaksimalkan lebar halaman untuk grafik

st.title("Aplikasi Kalkulator & Visualisasi Turunan Parsial")

# --- 1. Input Fungsi ---
st.header("1. Masukkan Fungsi f(x, y)")
fungsi_str = st.text_input("Masukkan fungsi f(x, y) (gunakan 'x' dan 'y' sebagai variabel, '**' untuk pangkat):", "x**2 + y**3")

# Definisikan simbol x dan y
x, y = sp.symbols('x y')

try:
    # --- 2. Hitung Turunan Parsial ---
    f = sp.sympify(fungsi_str) # Konversi string ke ekspresi SymPy

    # Pastikan ekspresi hanya mengandung x dan y
    # Ini penting untuk mencegah error jika user input variabel lain
    if not f.free_symbols.issubset({x, y}):
        st.error("Error: Fungsi hanya boleh mengandung variabel 'x' dan 'y'.")
    else:
        fx = sp.diff(f, x) # Turunan parsial terhadap x
        fy = sp.diff(f, y) # Turunan parsial terhadap y

        st.header("2. Hasil Turunan Parsial")
        st.latex(f"f(x, y) = {sp.latex(f)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(fx)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(fy)}")

        # --- 3. Evaluasi di Titik Tertentu ---
        st.header("3. Evaluasi di Titik (x₀, y₀)")
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input("Nilai x₀:", value=1.0, key="x0_input")
        with col2:
            y0 = st.number_input("Nilai y₀:", value=2.0, key="y0_input")

        # Evaluasi fungsi dan turunannya di titik (x0, y0)
        f_val = f.subs({x: x0, y: y0})
        fx_val = fx.subs({x: x0, y: y0}) # Perbaikan: fx_val harus disubs dengan y:y0 juga
        fy_val = fy.subs({x: x0, y: y0}) # Perbaikan: fy_val harus disubs dengan x:x0 juga

        st.write(f"Nilai fungsi $f({x0}, {y0})$: **{f_val:.4f}**")
        st.write(f"Gradien di titik $({x0}, {y0})$: $ \\nabla f = \\left( {fx_val:.4f}, {fy_val:.4f} \\right) $")

        # --- 4. Visualisasi Permukaan & Bidang Singgung ---
        st.header("4. Visualisasi Permukaan & Bidang Singgung")

        # Buat grid nilai x dan y untuk plot
        # Rentang plot disesuaikan agar titik (x0, y0) berada di tengah
        plot_range = 2 # Atur rentang plot di sekitar x0, y0
        x_vals = np.linspace(x0 - plot_range, x0 + plot_range, 50)
        y_vals = np.linspace(y0 - plot_range, y0 + plot_range, 50)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Ubah ekspresi SymPy menjadi fungsi NumPy untuk plot
        # Gunakan 'sympy' sebagai argumen lambdify untuk handle fungsi yang lebih kompleks
        Z_func = sp.lambdify((x, y), f, 'sympy')
        Z = Z_func(X, Y)

        # Pastikan f_val, fx_val, fy_val adalah float untuk operasi NumPy
        f_val_float = float(f_val)
        fx_val_float = float(fx_val)
        fy_val_float = float(fy_val)

        # Persamaan bidang singgung: Z_tangent = f(x0,y0) + fx(x0,y0)*(x - x0) + fy(x0,y0)*(y - y0)
        Z_tangent = f_val_float + fx_val_float * (X - x0) + fy_val_float * (Y - y0)

        fig = plt.figure(figsize=(10, 8)) # Ukuran plot lebih besar
        ax = fig.add_subplot(111, projection='3d')

        # Plot permukaan f(x, y)
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')

        # Plot bidang singgung
        ax.plot_surface(X, Y, Z_tangent, alpha=0.5, color='red', label='Bidang Singgung')

        # Tambahkan titik (x0, y0, f_val) di permukaan
        ax.scatter([x0], [y0], [f_val_float], color='blue', s=100, label='Titik Evaluasi', depthshade=True)

        ax.set_title("Permukaan f(x, y) dan Bidang Singgung")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend() # Menampilkan label
        ax.view_init(elev=30, azim=45) # Sudut pandang awal yang lebih baik

        st.pyplot(fig)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}. Pastikan format fungsi benar dan hanya menggunakan variabel 'x' dan 'y'. Contoh: x**2 + y**3")
