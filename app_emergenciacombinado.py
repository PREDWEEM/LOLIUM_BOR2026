# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — OPTIMIZADO PARA BORDENAVE
# Lógica: Desfase Temporal (+60d) + Restricción Sigmoide + Auto-Load
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import os
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM BORDENAVE vK4.4", 
    layout="wide",
    page_icon="🌾"
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] {
        background-color: #dcfce7; 
        border-right: 1px solid #bbf7d0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #166534 !important;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bio-alert {
        padding: 10px;
        border-radius: 5px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. FUNCIONES TÉCNICAS (ANN + BIO)
# ---------------------------------------------------------
def sigmoid_restriction(prec_sum, threshold=15, k=0.4):
    """Calcula el factor hídrico suave para Bordenave."""
    return 1 / (1 + np.exp(-k * (prec_sum - threshold)))

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit:
        factor = (t_crit - t) / (t_crit - t_opt)
        return (t - t_base) * factor
    else: return 0.0

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer).flatten() + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(
            np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
        )
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def get_data(file_input):
    """Carga automática prioritario de meteo_daily.csv."""
    try:
        # 1. Prioridad: Archivo local meteo_daily.csv
        local_file = BASE / "meteo_daily.csv"
        if not file_input and local_file.exists():
            df = pd.read_csv(local_file, parse_dates=["Fecha"])
            st.sidebar.success(f"✅ Cargado: {local_file.name}")
        # 2. Opción: Subida manual
        elif file_input:
            df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            return None
        
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec'}
        df = df.rename(columns=mapeo)
        return df
    except Exception as e:
        st.error(f"Error leyendo datos: {e}")
        return None

# ---------------------------------------------------------
# 3. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.markdown("## ⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual (Opcional)", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Pico)", 0.05, 0.80, 0.15)
t_base_val = st.sidebar.number_input("T Base", value=2.0, step=0.5)
t_opt_max = st.sidebar.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

dga_optimo = st.sidebar.number_input("Objetivo Control", value=600, step=50)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=50)

# ---------------------------------------------------------
# 4. MOTOR DE CÁLCULO (OPTIMIZADO BORDENAVE)
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # --- LÓGICA DE DESFASE (+60 días) ---
    df["JD_Shifted"] = (df["Julian_days"] + 60).clip(1, 300)
    
    # 1. Predicción ANN con desfase
    X = df[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)
    
    # 2. Restricción Hídrica Sigmoide
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = sigmoid_restriction(df["Prec_sum_21d"])
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    
    # En Bordenave se elimina el bloqueo del JD <= 25 por emergencia temprana
    
    # 3. Cálculo TT
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    df["TT_cum"] = df["DG"].cumsum()

    # --- VISUALIZACIÓN ---
    st.title("🌾 PREDWEEM BORDENAVE vK4.4")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Máxima Emergencia", f"{df['EMERREL'].max():.3f}")
    col2.metric("TT Acumulado", f"{df['TT_cum'].iloc[-1]:.1f} °Cd")
    col3.metric("Lluvia Total", f"{df['Prec'].sum():.1f} mm")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR", "🌧️ PRECIPITACIONES", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        fig_emer = go.Figure()
        fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Tasa Diaria", line=dict(color='#166534', width=3), fill='tozeroy'))
        fig_emer.update_layout(title="Dinámica de Emergencia con Desfase Temporal", template="plotly_white")
        st.plotly_chart(fig_emer, use_container_width=True)

    with tab2:
        fig_prec = go.Figure(go.Bar(x=df["Fecha"], y=df["Prec"], marker_color='#60a5fa'))
        st.plotly_chart(fig_prec, use_container_width=True)

    # Exportación
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Bordenave.xlsx")
else:
    st.info("👋 Bienvenido. Cargando datos desde 'meteo_daily.csv' o esperando carga manual.")
    
