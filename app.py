import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="COVID-19 Smart Diagnosis",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
}
.sub-title {
    text-align: center;
    color: #6c757d;
    margin-bottom: 30px;
}
.card {
    background-color: #f8f9fa;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.footer {
    text-align: center;
    font-size: 12px;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_excel("Dataset_COVID_DecisionTree_Umur.xlsx")

encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("Diagnosa", axis=1)
y = df["Diagnosa"]

model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
model.fit(X, y)

st.markdown("<div class='main-title'>🧬 Smart COVID-19 Diagnosis System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>"
    "</div>",
    unsafe_allow_html=True
)

st.info(
    "📌 **Petunjuk Penggunaan**\n\n"
    "1. Masukkan umur pasien\n"
    "2. Pilih kondisi gejala\n"
    "3. Klik tombol **Diagnosa** untuk melihat hasil"
)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧑‍⚕️ Data Pasien")

col1, col2 = st.columns(2)

with col1:
    umur = st.number_input("Umur (tahun)", min_value=1, max_value=100, step=1)
    demam = st.radio("Demam", ["Ya", "Tidak"])

with col2:
    batuk = st.radio("Batuk", ["Ya", "Tidak"])
    sesak = st.radio("Sesak Napas", ["Ya", "Tidak"])
    penciuman = st.radio("Kehilangan Penciuman", ["Ya", "Tidak"])

st.markdown("</div>", unsafe_allow_html=True)

def kategori_umur(u):
    if u < 30:
        return 0
    elif 30 <= u <= 50:
        return 1
    else:
        return 2

def ya_tidak(val):
    return 1 if val == "Ya" else 0

if st.button("🧪 Mulai Diagnosa", use_container_width=True):

    input_data = [[
        0,
        kategori_umur(umur),
        ya_tidak(demam),
        ya_tidak(batuk),
        ya_tidak(sesak),
        ya_tidak(penciuman)
    ]]

    hasil = model.predict(input_data)

    hasil_map = {
        0: "NEGATIF",
        1: "POSITIF",
        2: "SUSPEK"
    }

    hasil_text = hasil_map[hasil[0]]

    st.markdown("---")

    if hasil_text == "POSITIF":
        st.markdown(
            "<div class='result-box' style='background:#f8d7da;color:#842029;'>"
            "🟥 HASIL DIAGNOSA: POSITIF COVID-19"
            "</div>",
            unsafe_allow_html=True
        )
        st.warning("Segera lakukan pemeriksaan lanjutan dan konsultasi tenaga medis.")

    elif hasil_text == "SUSPEK":
        st.markdown(
            "<div class='result-box' style='background:#fff3cd;color:#664d03;'>"
            "🟨 HASIL DIAGNOSA: SUSPEK COVID-19"
            "</div>",
            unsafe_allow_html=True
        )
        st.info("Disarankan isolasi mandiri dan pemantauan kondisi kesehatan.")

    else:
        st.markdown(
            "<div class='result-box' style='background:#d1e7dd;color:#0f5132;'>"
            "🟩 HASIL DIAGNOSA: NEGATIF COVID-19"
            "</div>",
            unsafe_allow_html=True
        )
        st.success("Tetap jaga kesehatan dan patuhi protokol.")

st.markdown(
    "<div class='footer'>"
    "UAS Struktur Data • Decision Tree • Tio Kati Nuansya"
    "</div>",
    unsafe_allow_html=True
)
