import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# === Judul Aplikasi ===
st.title("🔩 Prediksi Penurunan Fondasi Tiang Bor (SVM)")

# === Load Model, Scaler, dan LabelEncoder ===
model_qp = joblib.load("svr_model_terbaik_Qp.pkl")
scaler_qp = joblib.load("scaler_svr_Qp.pkl")

model_qs = joblib.load("svr_model_terbaik_Qs.pkl")
scaler_qs = joblib.load("scaler_svr_Qs.pkl")

model_set = joblib.load("svr_model_terbaik_set.pkl")
scaler_set = joblib.load("scaler_svr_set.pkl")

le = joblib.load("label_encoder_jenis_tanah.pkl")  # pastikan file ini tersedia

# === Input Manual ===
st.header("1️⃣ Prediksi Manual")

jenis_tanah = st.selectbox("Jenis Tanah", le.classes_)  # pilihan otomatis dari encoder
jenis_tanah_num = le.transform([jenis_tanah])[0]

n_spt_p = st.number_input("Nilai N-SPT(p)", min_value=0.0)
n_spt_s = st.number_input("Nilai N-SPT(s)", min_value=0.0)
L = st.number_input("Panjang tiang (L)", min_value=0.0, format="%.2f")
st.caption("m")
D = st.number_input("Diameter tiang (D)", min_value=0.0, format="%.2f")
st.caption("m")
Q_beban = st.number_input("Beban kerja (Q)", min_value=0.0, format="%.2f")
st.caption("kN")

if st.button("🔍 Prediksi Manual"):
    try:
        # Prediksi Qp
        X_qp = pd.DataFrame([{
            'Jenis_Tanah': jenis_tanah_num, 'N-SPT(p)': n_spt_p, 'L': L, 'D': D
        }])
        qp_scaled = scaler_qp.transform(X_qp)
        Qp_pred = model_qp.predict(qp_scaled)[0]

        # Prediksi Qs
        X_qs = pd.DataFrame([{
            'Jenis_Tanah': jenis_tanah_num, 'N-SPT(s)': n_spt_s, 'L': L, 'D': D
        }])
        qs_scaled = scaler_qs.transform(X_qs)
        Qs_pred = model_qs.predict(qs_scaled)[0]

        # Prediksi Penurunan
        X_set = pd.DataFrame([{
            'L': L, 'Qp_pred': Qp_pred, 'Qs_pred': Qs_pred, 'D': D, 'Q(beban)': Q_beban
        }])
        set_scaled = scaler_set.transform(X_set)
        penurunan = model_set.predict(set_scaled)[0]

        st.success(f"Hasil Prediksi Penurunan: {penurunan:.2f} cm")
        st.write("📌 Rincian Prediksi:")
        st.write(f"- Qp Prediksi: {Qp_pred:.2f} kN")
        st.write(f"- Qs Prediksi: {Qs_pred:.2f} kN")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# === Upload Excel ===
st.header("2️⃣ Prediksi dari File Excel")

# Template download
st.subheader("📥 Download Template Excel")
template_data = {
    "Jenis_Tanah": [le.classes_[0]],
    "N-SPT(p)": [0],
    "N-SPT(s)": [0],
    "L [m]": [0],
    "D [m]": [0],
    "Q(beban) [kN]": [0]
}
template_df = pd.DataFrame(template_data)

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
    return output.getvalue()

excel_bytes = to_excel_bytes(template_df)

st.download_button(
    label="📄 Unduh Template Excel",
    data=excel_bytes,
    file_name="template_input_penurunan.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded = st.file_uploader("Upload File Excel (.xlsx)", type=["xlsx"])

if uploaded:
    try:
        df = pd.read_excel(uploaded)

        required_cols = ['Jenis_Tanah', 'N-SPT(p)', 'N-SPT(s)', 'L', 'D', 'Q(beban)']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Format kolom salah. Harus ada: {', '.join(required_cols)}")
            st.stop()

        # Validasi nilai Jenis_Tanah
        if not set(df['Jenis_Tanah']).issubset(set(le.classes_)):
            st.error(f"❌ Jenis Tanah harus salah satu dari: {', '.join(le.classes_)}")
            st.stop()

        df['Jenis_Tanah'] = le.transform(df['Jenis_Tanah'])

        # Prediksi Qp
        X_qp = df[['Jenis_Tanah', 'N-SPT(p)', 'L', 'D']]
        df['Qp_pred'] = model_qp.predict(scaler_qp.transform(X_qp))

        # Prediksi Qs
        X_qs = df[['Jenis_Tanah', 'N-SPT(s)', 'L', 'D']]
        df['Qs_pred'] = model_qs.predict(scaler_qs.transform(X_qs))

        # Prediksi Penurunan
        X_set = df[['L', 'Qp_pred', 'Qs_pred', 'D', 'Q(beban)']]
        df['Prediksi Penurunan (mm)'] = model_set.predict(scaler_set.transform(X_set))

        st.success("✅ Prediksi berhasil dihitung!")
        st.dataframe(df)

        hasil_excel = "hasil_prediksi_penurunan_batch.xlsx"
        df.to_excel(hasil_excel, index=False)
        st.download_button(
            label="⬇️ Unduh Hasil Prediksi",
            data=open(hasil_excel, 'rb').read(),
            file_name=hasil_excel,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"⚠️ Terjadi error: {e}")
