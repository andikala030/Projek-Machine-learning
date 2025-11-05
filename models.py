import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, 
                             f1_score, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

class ChurnModel:
    """Class untuk mengelola model machine learning churn prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = None
        self.le_dict = {}
        self.feature_columns = None
        
    def preprocess_data(self, df):
        """Preprocessing data - encode categorical variables"""
        df_processed = df.copy()
        
        # Encode categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'customerID':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.le_dict[col] = le
        
        return df_processed
    
    def prepare_features(self, df_processed):
        """Siapkan fitur untuk training"""
        X = df_processed.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        self.feature_columns = X.columns
        return X
    
    def scale_features(self, X, fit=True):
        """Scale fitur menggunakan StandardScaler"""
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression model"""
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        
        lr_pred = lr.predict(X_test)
        lr_proba = lr.predict_proba(X_test)[:, 1]
        
        self.models['Logistic Regression'] = lr
        self.results['Logistic Regression'] = {
            'predictions': lr_pred,
            'probabilities': lr_proba,
            'f1': f1_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred)
        }
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        rf_pred = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)[:, 1]
        
        self.models['Random Forest'] = rf
        self.results['Random Forest'] = {
            'predictions': rf_pred,
            'probabilities': rf_proba,
            'f1': f1_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'feature_importance': rf.feature_importances_
        }
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        xgb = XGBClassifier(n_estimators=100, random_state=42, 
                           use_label_encoder=False, eval_metric='logloss', verbosity=0)
        xgb.fit(X_train, y_train)
        
        xgb_pred = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        
        self.models['XGBoost'] = xgb
        self.results['XGBoost'] = {
            'predictions': xgb_pred,
            'probabilities': xgb_proba,
            'f1': f1_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred),
            'recall': recall_score(y_test, xgb_pred),
            'feature_importance': xgb.feature_importances_
        }
    
    def train_all_models(self, df, test_size=0.2, random_state=42):
        """Train semua model sekaligus"""
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Prepare features
        X = self.prepare_features(df_processed)
        y = df_processed['Churn']
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Train all models
        self.train_logistic_regression(X_train, X_test, y_train, y_test)
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_xgboost(X_train, X_test, y_train, y_test)
        
        return X_test, y_test
    
    def predict_churn(self, input_data, model_name='Random Forest'):
        """Prediksi churn untuk data baru"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' tidak ditemukan")
        
        # Preprocess input
        input_processed = input_data.copy()
        
        # Encode categorical columns
        categorical_cols = input_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.le_dict:
                input_processed[col] = self.le_dict[col].transform(input_processed[col].astype(str))
        
        # Select only feature columns
        input_features = input_processed[self.feature_columns]
        
        # Scale
        input_scaled = self.scaler.transform(input_features)
        
        # Predict
        model = self.models[model_name]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return prediction, probability
    
    def get_confusion_matrix(self, model_name, y_test):
        """Get confusion matrix untuk model"""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' tidak ditemukan")
        
        predictions = self.results[model_name]['predictions']
        cm = confusion_matrix(y_test, predictions)
        return cm
    
    def get_roc_curve(self, model_name, y_test):
        """Get ROC curve data untuk model"""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' tidak ditemukan")
        
        probabilities = self.results[model_name]['probabilities']
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        
        return fpr, tpr, roc_auc
    
    def get_classification_report(self, model_name, y_test):
        """Get classification report untuk model"""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' tidak ditemukan")
        
        predictions = self.results[model_name]['predictions']
        report = classification_report(y_test, predictions, 
                                      output_dict=True, zero_division=0)
        return report
    
    def get_model_metrics(self, model_name):
        """Get metrics untuk model"""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' tidak ditemukan")
        
        return {
            'f1': self.results[model_name]['f1'],
            'precision': self.results[model_name]['precision'],
            'recall': self.results[model_name]['recall']
        }
    
    def get_feature_importance(self, model_name):
        """Get feature importance untuk model"""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' tidak memiliki feature importance")
        
        if 'feature_importance' not in self.results[model_name]:
            return None
        
        importance = self.results[model_name]['feature_importance']
        features = self.feature_columns
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def get_available_models(self):
        """Get list model yang tersedia"""
        return list(self.models.keys())


class DataProcessor:
    """Class untuk memproses dan menganalisis data"""
    
    @staticmethod
    def generate_sample_data(n_samples=1000):
        """Generate sample data untuk testing"""
        np.random.seed(42)
        df = pd.DataFrame({
            'customerID': [f'C{i:05d}' for i in range(n_samples)],
            'tenure': np.random.randint(1, 73, n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(100, 8500, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No'], n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
        })
        return df
    
    @staticmethod
    def get_churn_statistics(df):
        """Hitung statistik churn"""
        total_customers = len(df)
        churn_customers = (df['Churn'] == 1).sum()
        churn_rate = (churn_customers / total_customers) * 100
        retention_rate = 100 - churn_rate
        
        return {
            'total_customers': total_customers,
            'churn_customers': churn_customers,
            'active_customers': total_customers - churn_customers,
            'churn_rate': churn_rate,
            'retention_rate': retention_rate
        }
    
    @staticmethod
    def get_tenure_churn(df):
        """Analisis churn berdasarkan tenure"""
        tenure_churn = df.groupby('tenure')['Churn'].agg(['sum', 'count'])
        tenure_churn['churn_rate'] = (tenure_churn['sum'] / tenure_churn['count']) * 100
        return tenure_churn
    
    @staticmethod
    def get_contract_churn(df):
        """Analisis churn berdasarkan jenis kontrak"""
        return pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    
    @staticmethod
    def filter_data(df, contract_filter, internet_filter, partner_filter, 
                    dependents_filter, senior_filter, phone_filter):
        """Filter data berdasarkan kriteria"""
        filtered_df = df[
            (df['Contract'].isin(contract_filter)) &
            (df['InternetService'].isin(internet_filter)) &
            (df['Partner'].isin(partner_filter)) &
            (df['Dependents'].isin(dependents_filter)) &
            (df['SeniorCitizen'].isin(senior_filter)) &
            (df['PhoneService'].isin(phone_filter))
        ]
        return filtered_df
    
    @staticmethod
    def get_high_risk_customers(df):
        """Identifikasi pelanggan berisiko tinggi churn"""
        high_risk = df[
            (df['tenure'] < 12) & 
            (df['MonthlyCharges'] > 80) &
            (df['Contract'] == 'Month-to-month')
        ]
        return high_risk