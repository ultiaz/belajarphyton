import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model yang sudah disimpan
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# UI Streamlit
st.title("🚀 Fraud Detection App")
st.markdown("Aplikasi ini mendeteksi apakah transaksi mencurigakan atau tidak berdasarkan jumlah transaksi dan lokasi.")

# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Data Transaksi")
amount = st.sidebar.number_input("💰 Jumlah Transaksi (IDR)", min_value=0, value=500)
location = st.sidebar.selectbox("📍 Pilih Lokasi Transaksi", ["A", "B", "C"])

# Konversi lokasi ke angka
location_mapping = {"A": 0, "B": 1, "C": 2}
location = location_mapping[location]

# Prediksi fraud
if st.sidebar.button("🔍 Cek Transaksi"):
    input_data = np.array([[amount, location]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Transaksi mencurigakan (Fraud) ❌")
    else:
        st.success("✅ Transaksi normal")

# **📊 Tambahkan Grafik untuk Visualisasi**
st.subheader("📊 Distribusi Jumlah Transaksi")
data_sample = pd.DataFrame({
    "Jumlah Transaksi": [100, 500, 2000, 150, 700, 50, 3000, 250, 800, 1000],
    "Kategori": ["Normal", "Normal", "Fraud", "Normal", "Normal", "Normal", "Fraud", "Normal", "Normal", "Fraud"]
})

fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(data_sample["Jumlah Transaksi"], bins=5, color="skyblue", edgecolor="black")
plt.xlabel("Jumlah Transaksi (IDR)")
plt.ylabel("Frekuensi")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("📌 *Aplikasi ini menggunakan model Machine Learning sederhana dengan Decision Tree.*")
