import streamlit as st
import pandas as pd
import joblib

# Load model & encoder
model = joblib.load("fraud_detection.pkl")
le = joblib.load("label_encoder.pkl")

# Judul aplikasi
st.title("Fraud Detection App 🚀")

# Input dari user
amount = st.number_input("Masukkan jumlah transaksi:", min_value=1, value=100)

location = st.selectbox("Pilih lokasi transaksi:", ["A", "B", "C"])
location_encoded = le.transform([location])[0]  # Convert lokasi ke angka

# Prediksi ketika tombol ditekan
if st.button("Cek Transaksi"):
    new_data = pd.DataFrame({'amount': [amount], 'location': [location_encoded]})
    prediction = model.predict(new_data)
    
    if prediction[0] == 1:
        st.error("🚨 Transaksi mencurigakan (Fraud)!")
    else:
        st.success("✅ Transaksi normal.")
