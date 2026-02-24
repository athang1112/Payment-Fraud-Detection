import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Set Page Config
st.set_page_config(page_title="Payment Fraud Detection", page_icon="🛡️", layout="wide")

# Custom CSS for High Contrast and Premium Design
st.markdown("""
<style>
    .stApp { background-color: #05070a; color: #ffffff; }
    .status-warning {
        background-color: rgba(255, 165, 0, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ffa500;
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 25px;
        border: 2px solid rgba(0, 212, 255, 0.4);
        text-align: center;
        transition: transform 0.3s ease;
        margin-bottom: 20px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.12);
        border-color: #00d4ff;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
    }
    .stNumberInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    .stButton>button {
        width: 100%;
        background-image: linear-gradient(to right, #00c6ff 0%, #0072ff 51%, #00c6ff 100%);
        border: none;
        color: white;
        padding: 15px 45px;
        text-align: center;
        text-transform: uppercase;
        transition: 0.5s;
        background-size: 200% auto;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        display: block;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-position: right center;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# Robust Data Loader with explicit error handling and retries
@st.cache_data(show_spinner="Loading data in chunks...")
def load_data_safely():
    current_dir = Path(__file__).parent.absolute()
    csv_path = current_dir / 'Fraud.csv'
    if not csv_path.exists():
        csv_path = Path('Fraud.csv').absolute()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']
            fraud_df = pd.DataFrame()
            non_fraud_sample = pd.DataFrame()
            
            # Using chunksize to keep memory footprint low
            # The 'with' block ensures the file handle is closed even if an error occurs
            with pd.read_csv(csv_path, usecols=cols, chunksize=500000) as reader:
                for chunk in reader:
                    fraud_df = pd.concat([fraud_df, chunk[chunk['isFraud'] == 1]])
                    if len(chunk) > 1000:
                        nf = chunk[chunk['isFraud'] == 0].sample(n=min(1000, len(chunk)//10), random_state=42)
                        non_fraud_sample = pd.concat([non_fraud_sample, nf])
            
            if fraud_df.empty or non_fraud_sample.empty:
                return None, None
                
            non_fraud_final = non_fraud_sample.sample(n=min(len(fraud_df), len(non_fraud_sample)), random_state=42)
            balanced_data = pd.concat([fraud_df, non_fraud_final]).reset_index(drop=True)
            
            # Type mapping
            m = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
            balanced_data["type"] = balanced_data["type"].map(m)
            balanced_data = balanced_data.dropna(subset=['type'])
            
            if balanced_data.empty:
                return None, None
                
            X = balanced_data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
            y = balanced_data['isFraud']
            return X, y
            
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(1) # Wait and retry
                continue
            else:
                st.error("❌ Permission Error: 'Fraud.csv' is locked. Please close it in other apps and refresh.")
                return None, None
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None, None
    return None, None

@st.cache_resource(show_spinner="Training model...")
def get_model_and_scaler():
    try:
        X, y = load_data_safely()
        if X is None or y is None or len(X) == 0:
            return None, None
            
        # Ensure data is 2D and writable
        X_train = np.array(X, copy=True, dtype=np.float64)
        y_train = np.array(y, copy=True, dtype=np.int64)
        
        if X_train.ndim != 2:
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_scaled, y_train)
        return model, scaler
    except Exception as e:
        # Catch-all to prevent app crash with traceback
        return None, None

# Sidebar
st.sidebar.title("🛡️ Fraud Guard")
selection = st.sidebar.radio("Navigation", ["Dashboard", "Fraud Prediction"])

# Pre-defined stats
stats = {
    "XGBoost": {"Accuracy": 0.994, "Precision": 0.994, "Recall": 0.994, "F1": 0.994},
    "Random Forest": {"Accuracy": 0.993, "Precision": 0.993, "Recall": 0.993, "F1": 0.993},
    "Decision Tree": {"Accuracy": 0.992, "Precision": 0.992, "Recall": 0.992, "F1": 0.992},
    "K-Nearest Neighbors": {"Accuracy": 0.988, "Precision": 0.988, "Recall": 0.988, "F1": 0.988},
    "Logistic Regression": {"Accuracy": 0.908, "Precision": 0.908, "Recall": 0.908, "F1": 0.908},
}

if selection == "Dashboard":
    st.title("📊 Model Performance Dashboard")
    algo = st.selectbox("Select Algorithm", list(stats.keys()))
    m = stats[algo]
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><h2 style="color:#00ff88">{m["Accuracy"]*100:.1f}%</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3>Precision</h3><h2 style="color:#00ff88">{m["Precision"]*100:.1f}%</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><h3>Recall</h3><h2 style="color:#00ff88">{m["Recall"]*100:.1f}%</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><h3>F1-Score</h3><h2 style="color:#00ff88">{m["F1"]*100:.1f}%</h2></div>', unsafe_allow_html=True)
    
    st.write("---")
    df_acc = pd.DataFrame({"Model": list(stats.keys()), "Accuracy": [v["Accuracy"] for v in stats.values()]})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Accuracy", y="Model", data=df_acc, hue="Model", palette="viridis", ax=ax, legend=False)
    ax.set_facecolor("#1a1c24")
    fig.patch.set_facecolor("#05070a")
    ax.tick_params(colors="white")
    st.pyplot(fig)

elif selection == "Fraud Prediction":
    st.title("🔮 Fraud Prediction")
    
    model, scaler = get_model_and_scaler()
    
    if model is not None:
        with st.container():
            st.markdown('<div class="metric-card" style="text-align: left;">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                t = st.number_input("Type (1:CASH_OUT, 2:PAYMENT, 3:CASH_IN, 4:TRANSFER, 5:DEBIT)", value=2, min_value=1, max_value=5)
                a = st.number_input("Amount", value=100.0, format="%.2f")
            with c2:
                o = st.number_input("Old Balance", value=0.0, format="%.2f")
                n = st.number_input("New Balance", value=0.0, format="%.2f")
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("RUN DETECTION"):
                inp = np.array([[t, a, o, n]], dtype=float)
                prediction = model.predict(scaler.transform(inp))[0]
                if prediction == 0:
                    st.success("### ✔️ Safe Transaction Detected")
                else:
                    st.error("### ❌ Fraud Warning Detected")
    else:
        st.warning("⚠️ Waiting for data to load. Please ensure 'Fraud.csv' is closed in other apps.")
        if st.button("Retry Loading"):
            st.cache_data.clear()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Stable Manual Input Active")
