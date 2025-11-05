import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📊 Prediksi Churn Pelanggan</h1>
    <p style='text-align: center;'>Gunakan algoritma Machine Learning pilihan Anda untuk memprediksi apakah pelanggan akan berhenti berlangganan (churn).</p>
    <hr>
""", unsafe_allow_html=True)

# Upload file CSV
uploaded_file = st.file_uploader("📁 Upload dataset pelanggan (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- INFORMASI DATA ---
    st.markdown("## 🧾 Informasi Data")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**🔹 Dimensi Data:**", df.shape)
        st.write("**🔹 Kolom Data:**", list(df.columns))
        st.write("**🔹 Tipe Data:**")
        st.dataframe(df.dtypes.rename("Tipe Data"))

    with col2:
        st.write("**🔹 Jumlah Nilai Kosong (Missing Values):**")
        st.dataframe(df.isnull().sum())

    st.markdown("### 📋 5 Data Teratas (Data Asli)")
    st.dataframe(df.head())

    # --- VISUALISASI DATA SEBELUM PREPROCESSING ---
    st.markdown("## 📊 Visualisasi Awal")

    if "Churn" in df.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            churn_counts = df["Churn"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%", startangle=90, colors=["#4CAF50", "#F44336"])
            st.pyplot(fig)
            st.caption("Proporsi pelanggan churn vs tidak churn")

        with col_b:
            if "tenure" in df.columns:
                fig, ax = plt.subplots()
                sns.histplot(df, x="tenure", hue="Churn", multiple="stack", ax=ax)
                st.pyplot(fig)
                st.caption("Distribusi lama berlangganan (tenure) berdasarkan status churn")
    else:
        st.warning("⚠️ Kolom 'Churn' tidak ditemukan. Beberapa visualisasi tidak dapat ditampilkan.")

    # --- PREPROCESSING ---
    st.markdown("## ⚙️ Tahapan Pra-Pemrosesan Data")

    # 1️⃣ Hapus kolom ID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
        st.success("✅ Kolom 'customerID' dihapus karena bukan fitur penting untuk prediksi.")

    # 2️⃣ Ubah kolom TotalCharges ke numerik
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        before = len(df)
        df.dropna(inplace=True)
        dropped = before - len(df)
        st.info(f"💡 {dropped} baris dihapus karena berisi nilai kosong pada kolom 'TotalCharges'.")

    # 3️⃣ Encoding kolom kategorikal
    st.markdown("### 🔡 Encoding Fitur Kategorikal")
    label_encoders = {}
    label_cols = df.select_dtypes(include=["object"]).columns

    if len(label_cols) > 0:
        for col in label_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        if "Churn" in label_encoders:
            del label_encoders["Churn"]

        st.success(f"✅ {len(label_cols)} kolom kategorikal telah diubah ke bentuk numerik menggunakan LabelEncoder.")
        st.write("**📋 Kolom yang di-encode:**", list(label_cols))

        # 🔍 Tampilkan contoh hasil encoding
        st.markdown("### 🔢 Contoh Data Setelah Encoding")
        st.dataframe(df.head())
    else:
        st.info("ℹ️ Tidak ada kolom bertipe kategorikal yang perlu di-encode.")

    # --- Visualisasi Korelasi ---
    st.markdown("### 🔍 Korelasi Antar Fitur")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --- Split Data ---
    if "Churn" not in df.columns:
        st.error("❌ Kolom 'Churn' tidak ditemukan dalam dataset.")
        st.stop()

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- PILIH ALGORITMA ---
    st.markdown("## 🧠 Pilih Algoritma Machine Learning")

    algoritma = st.selectbox(
        "Pilih algoritma untuk digunakan dalam model:",
        ["Random Forest", "Logistic Regression", "Decision Tree", "Support Vector Machine (SVM)"]
    )

    if algoritma == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif algoritma == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif algoritma == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = SVC(probability=True, random_state=42)

    # --- TRAINING MODEL ---
    st.markdown("## 🏋️‍♂️ Proses Pelatihan Model")
    with st.spinner(f"Model {algoritma} sedang dilatih..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # --- EVALUASI MODEL ---
    st.markdown("## 📈 Evaluasi Model")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="🎯 Akurasi Model", value=f"{acc*100:.2f}%")

    st.text("📊 Laporan Klasifikasi:")
    st.code(classification_report(y_test, y_pred), language="text")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=["Tidak Churn", "Churn"],
                yticklabels=["Tidak Churn", "Churn"], ax=ax)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig)

    # --- Feature Importance ---
    if hasattr(model, "feature_importances_"):
        st.markdown("### 📌 Pengaruh Fitur terhadap Churn (Feature Importance)")
        fi = pd.DataFrame({
            "Fitur": X.columns,
            "Pentingnya": model.feature_importances_
        }).sort_values(by="Pentingnya", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Pentingnya", y="Fitur", data=fi, palette="viridis", ax=ax)
        st.pyplot(fig)

    # --- FORM INPUT PREDIKSI ---
    st.markdown("## 🧪 Prediksi Pelanggan Baru")

    st.markdown("""
    <div style='background-color:#f6f6f6;padding:10px;border-radius:8px;'>
    <b>📋 Petunjuk:</b> Masukkan data pelanggan baru sesuai format dataset asli. 
    Kolom bertipe kategori akan muncul sebagai pilihan (dropdown).
    </div>
    <br>
    """, unsafe_allow_html=True)

    input_data = {}
    col1, col2 = st.columns(2)

    # 🧍 Form input pelanggan baru (bahasa Indonesia)
    for i, col in enumerate(X.columns):
        if i % 2 == 0:
            target_col = col1
        else:
            target_col = col2

        if col in label_encoders:
            le = label_encoders[col]
            options = list(le.classes_)
            input_data[col] = target_col.selectbox(f"Pilih nilai untuk kolom **{col}**:", options)
        else:
            mean_value = float(df[col].mean())
            input_data[col] = target_col.number_input(f"Masukkan nilai untuk kolom **{col}**:", value=mean_value)

    if st.button("🔮 Lakukan Prediksi", use_container_width=True):
        input_df = pd.DataFrame([input_data])

        # Encoding input baru
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        result = "❌ Pelanggan kemungkinan *Churn*" if prediction == 1 else "✅ Pelanggan kemungkinan *Tidak Churn*"

        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.success(f"Hasil Prediksi: {result}")
        with col_res2:
            st.metric("Probabilitas Churn", f"{probability*100:.2f}%")

else:
    st.info("📤 Silakan upload dataset terlebih dahulu untuk memulai analisis.")
