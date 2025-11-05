import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import ChurnModel, DataProcessor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==================== SESSION STATE ====================
if 'data' not in st.session_state:
    st.session_state.data = DataProcessor.generate_sample_data()
if 'model' not in st.session_state:
    st.session_state.model = ChurnModel()
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# ==================== SIDEBAR ====================
st.sidebar.title("🎛️ Kontrol Aplikasi")
page = st.sidebar.radio("Pilih Halaman:", 
                        ["Dashboard", "Prediksi Churn", "Filter & Segmentasi", 
                         "Analisis Fitur", "Manajemen Data", "Model & Evaluasi"])

# ==================== PAGE: DASHBOARD ====================
if page == "Dashboard":
    st.title("📊 Dashboard Ringkasan Churn")
    
    df = st.session_state.data
    stats = DataProcessor.get_churn_statistics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pelanggan", f"{stats['total_customers']:,}")
    with col2:
        st.metric("Churn Rate", f"{stats['churn_rate']:.2f}%", 
                 delta=f"{stats['churn_customers']} pelanggan")
    with col3:
        st.metric("Retention Rate", f"{stats['retention_rate']:.2f}%")
    with col4:
        st.metric("Pelanggan Aktif", f"{stats['active_customers']:,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Tren Churn Berdasarkan Tenure")
        tenure_churn = DataProcessor.get_tenure_churn(df)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tenure_churn.index, tenure_churn['churn_rate'], marker='o', linewidth=2)
        ax.fill_between(tenure_churn.index, tenure_churn['churn_rate'], alpha=0.3)
        ax.set_xlabel("Tenure (Bulan)")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Tren Churn Berdasarkan Tenure")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🌐 Segmentasi Pelanggan Berdasarkan Internet Service")
        internet_dist = df['InternetService'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette("husl", len(internet_dist))
        ax.pie(internet_dist, labels=internet_dist.index, autopct='%1.1f%%', colors=colors)
        ax.set_title("Distribusi Internet Service")
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Churn berdasarkan Jenis Kontrak")
        contract_churn = DataProcessor.get_contract_churn(df)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        contract_churn.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_xlabel("Jenis Kontrak")
        ax.set_ylabel("Persentase (%)")
        ax.set_title("Churn Rate Berdasarkan Jenis Kontrak")
        ax.legend(['Aktif', 'Churn'])
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("👥 Status Pelanggan")
        status_data = {
            'Senior Citizen': df['SeniorCitizen'].sum(),
            'Memiliki Partner': (df['Partner'] == 'Yes').sum(),
            'Memiliki Dependents': (df['Dependents'] == 'Yes').sum(),
            'Phone Service': (df['PhoneService'] == 'Yes').sum()
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(status_data.keys()), list(status_data.values()), color='#3498db')
        ax.set_xlabel("Jumlah Pelanggan")
        st.pyplot(fig)
        plt.close()

# ==================== PAGE: PREDIKSI CHURN ====================
elif page == "Prediksi Churn":
    st.title("🔮 Prediksi Churn Otomatis")
    
    available_models = st.session_state.model.get_available_models()
    
    if not available_models:
        st.warning("⚠️ Model belum dilatih. Silakan ke halaman 'Model & Evaluasi' terlebih dahulu.")
    else:
        selected_model = st.selectbox("Pilih Model:", available_models)
        
        st.subheader("📝 Input Data Pelanggan Baru")
        
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (bulan):", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($):", 20.0, 120.0, 65.0)
            total_charges = st.slider("Total Charges ($):", 100.0, 8500.0, 2000.0)
            contract = st.selectbox("Jenis Kontrak:", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service:", ["DSL", "Fiber optic", "No"])
        
        with col2:
            online_security = st.selectbox("Online Security:", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup:", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service:", ["Yes", "No"])
            partner = st.selectbox("Partner:", ["Yes", "No"])
            dependents = st.selectbox("Dependents:", ["Yes", "No"])
            senior_citizen = st.selectbox("Senior Citizen:", [0, 1])
        
        if st.button("🔍 Prediksi Churn", key="predict_btn"):
            # Prepare input data
            input_data = pd.DataFrame({
                'customerID': ['NEW001'],
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract': [contract],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'PhoneService': [phone_service],
                'Partner': [partner],
                'Dependents': [dependents],
                'SeniorCitizen': [senior_citizen]
            })
            
            try:
                # Predict
                prediction, probability = st.session_state.model.predict_churn(input_data, selected_model)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    status = "🔴 AKAN CHURN" if prediction == 1 else "🟢 AKAN TETAP"
                    st.markdown(f"### {status}")
                
                with col2:
                    st.markdown(f"### Probabilitas Churn: {probability*100:.2f}%")
                    
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.barh(['Churn', 'Retain'], [probability, 1-probability], 
                           color=['#e74c3c', '#2ecc71'])
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probabilitas")
                    for i, v in enumerate([probability, 1-probability]):
                        ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center')
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")

# ==================== PAGE: FILTER & SEGMENTASI ====================
elif page == "Filter & Segmentasi":
    st.title("🔍 Filter & Segmentasi Pelanggan")
    
    df = st.session_state.data
    
    st.subheader("⚙️ Pilih Filter")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contract_filter = st.multiselect("Jenis Kontrak:", 
                                        df['Contract'].unique(), 
                                        default=list(df['Contract'].unique()))
    with col2:
        internet_filter = st.multiselect("Internet Service:",
                                        df['InternetService'].unique(),
                                        default=list(df['InternetService'].unique()))
    with col3:
        partner_filter = st.multiselect("Partner:", ["Yes", "No"], default=["Yes", "No"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dependents_filter = st.multiselect("Dependents:", ["Yes", "No"], default=["Yes", "No"])
    with col2:
        senior_filter = st.multiselect("Senior Citizen:", [0, 1], default=[0, 1])
    with col3:
        phone_filter = st.multiselect("Phone Service:", ["Yes", "No"], default=["Yes", "No"])
    
    # Apply filters
    filtered_df = DataProcessor.filter_data(df, contract_filter, internet_filter, 
                                           partner_filter, dependents_filter, 
                                           senior_filter, phone_filter)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pelanggan (Filtered)", len(filtered_df))
    with col2:
        churn_filtered = (filtered_df['Churn'] == 1).sum()
        st.metric("Churn", churn_filtered)
    with col3:
        churn_rate_filtered = (churn_filtered / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Churn Rate", f"{churn_rate_filtered:.2f}%")
    
    st.markdown("---")
    st.subheader("⚠️ Pelanggan Berisiko Tinggi Churn")
    
    high_risk = DataProcessor.get_high_risk_customers(filtered_df)
    
    st.write(f"Ditemukan {len(high_risk)} pelanggan berisiko tinggi")
    if len(high_risk) > 0:
        st.dataframe(high_risk[['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'Churn']].head(10),
                    use_container_width=True)
    else:
        st.info("Tidak ada pelanggan berisiko tinggi dengan filter yang dipilih.")

# ==================== PAGE: ANALISIS FITUR ====================
elif page == "Analisis Fitur":
    st.title("📉 Analisis Hubungan Fitur dengan Churn")
    
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tenure vs Churn")
        tenure_churn = df.groupby('tenure')['Churn'].mean() * 100
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(tenure_churn.index, tenure_churn.values, alpha=0.6, s=100)
        z = np.polyfit(tenure_churn.index, tenure_churn.values, 3)
        p = np.poly1d(z)
        ax.plot(tenure_churn.index, p(tenure_churn.index), "r--", linewidth=2)
        ax.set_xlabel("Tenure (Bulan)")
        ax.set_ylabel("Churn Rate (%)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("💰 Monthly Charges vs Churn")
        fig, ax = plt.subplots(figsize=(10, 5))
        churn_charges = df[df['Churn'] == 1]['MonthlyCharges']
        retain_charges = df[df['Churn'] == 0]['MonthlyCharges']
        ax.hist([retain_charges, churn_charges], label=['Retain', 'Churn'], 
               color=['#2ecc71', '#e74c3c'], alpha=0.7, bins=20)
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Jumlah Pelanggan")
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Contract Type vs Churn")
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        contract_churn.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Jumlah Pelanggan")
        ax.legend(['Retain', 'Churn'])
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🔗 Korelasi Fitur")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, 
                   cbar_kws={'label': 'Korelasi'})
        ax.set_title("Heatmap Korelasi Fitur")
        st.pyplot(fig)
        plt.close()

# ==================== PAGE: MANAJEMEN DATA ====================
elif page == "Manajemen Data":
    st.title("📁 Manajemen Data Pelanggan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Data Baru")
        uploaded_file = st.file_uploader("Pilih file CSV atau Excel:", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('csv'):
                    new_df = pd.read_csv(uploaded_file)
                else:
                    new_df = pd.read_excel(uploaded_file)
                
                if st.button("Gunakan Data Ini"):
                    st.session_state.data = new_df
                    st.success("✅ Data berhasil diperbarui!")
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
    
    with col2:
        st.subheader("📥 Ekspor Hasil Prediksi")
        if len(st.session_state.data) > 0:
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="Download Data (CSV)",
                data=csv,
                file_name="churn_data.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    st.subheader("📋 Tabel Data Pelanggan")
    
    st.dataframe(st.session_state.data, use_container_width=True)
    
    st.markdown("---")
    st.info(f"Total baris: {len(st.session_state.data)} | Total kolom: {len(st.session_state.data.columns)}")

# ==================== PAGE: MODEL & EVALUASI ====================
elif page == "Model & Evaluasi":
    st.title("🤖 Model dan Evaluasi")
    
    df = st.session_state.data
    
    if st.button("🚀 Train Ulang Model", key="train_btn"):
        with st.spinner("Sedang melatih model..."):
            try:
                churn_model = ChurnModel()
                X_test, y_test = churn_model.train_all_models(df)
                
                st.session_state.model = churn_model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                st.success("✅ Model berhasil dilatih!")
                st.balloons()
            except Exception as e:
                st.error(f"Error dalam training model: {str(e)}")
    
    if st.session_state.model.get_available_models():
        st.markdown("---")
        
        # Model Selection
        selected_model = st.selectbox("Pilih Model:", st.session_state.model.get_available_models())
        
        # Get metrics
        metrics = st.session_state.model.get_model_metrics(selected_model)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔲 Confusion Matrix")
            try:
                cm = st.session_state.model.get_confusion_matrix(selected_model, st.session_state.y_test)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error menampilkan confusion matrix: {str(e)}")
        
        with col2:
            st.subheader("📈 ROC Curve")
            try:
                fpr, tpr, roc_auc = st.session_state.model.get_roc_curve(selected_model, st.session_state.y_test)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error menampilkan ROC curve: {str(e)}")
        
        st.markdown("---")
        st.subheader("📊 Classification Report")
        try:
            report = st.session_state.model.get_classification_report(selected_model, st.session_state.y_test)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error menampilkan classification report: {str(e)}")
        
        # Feature Importance
        st.markdown("---")
        st.subheader("⭐ Feature Importance")
        try:
            feat_importance = st.session_state.model.get_feature_importance(selected_model)
            
            if feat_importance is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feat_importance['Feature'], feat_importance['Importance'], color='#3498db')
                ax.set_xlabel("Importance")
                ax.set_title(f"Feature Importance - {selected_model}")
                st.pyplot(fig)
                plt.close()
                
                st.dataframe(feat_importance, use_container_width=True)
            else:
                st.info("Model ini tidak memiliki feature importance.")
        except Exception as e:
            st.warning(f"Feature importance tidak tersedia untuk model ini: {str(e)}")
    else:
        st.info("📌 Klik tombol 'Train Ulang Model' untuk melatih model machine learning.")

st.markdown("---")
st.markdown("<center>🎯 Aplikasi Prediksi Churn | Built with Streamlit</center>", unsafe_allow_html=True)