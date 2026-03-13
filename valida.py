# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.7 — LOLIUM BORDENAVE 2026
# Consolidado: Modelo Original + PEC Estricto + Sincronía Flexible
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM BORDENAVE vK4.7", 
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
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. LÓGICA TÉCNICA (ANN ORIGINAL + BIO)
# ---------------------------------------------------------
def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
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
            np.load(BASE/"IW.npy"), 
            np.load(BASE/"bias_IW.npy"), 
            np.load(BASE/"LW.npy"), 
            np.load(BASE/"bias_out.npy")
        )
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        return None, None

# ---------------------------------------------------------
# 3. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png", use_container_width=True)
st.sidebar.markdown("## 📂 1. Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima (bordenave)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta", 0.05, 0.80, 0.15)
dga_optimo = st.sidebar.number_input("TT Control (°Cd)", value=600, step=10)
residualidad = st.sidebar.number_input("Residualidad (días)", 0, 60, 20)

# ---------------------------------------------------------
# 4. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if archivo_meteo is not None and modelo_ann is not None:
    
    # --- PREPROCESAMIENTO CLIMA ---
    df = pd.read_csv(archivo_meteo) if archivo_meteo.name.endswith('.csv') else pd.read_excel(archivo_meteo)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # --- PREDICCIÓN (MODELO ORIGINAL SIN SHIFT) ---
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df["EMERREL"], _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(df["EMERREL"], 0.0)
    
    # --- RESTRICCIÓN HÍDRICA ORIGINAL ---
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    df.loc[df["Julian_days"] <= np.where(df["Prec_sum_21d"] > 50, 0, 25), "EMERREL"] = 0.0

    # Suavizado para Sincronía Flexible (Margen ±3 días)
    df["EMERREL_FLEX"] = df["EMERREL"].rolling(window=7, center=True, min_periods=1).mean()

    # --- BIO-TÉRMICO Y VENTANA ---
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, 2.0, 20.0, 30.0))
    
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_control = None
    if indices_pulso:
        f_inicio = df.loc[indices_pulso[0], "Fecha"]
        df_desde = df[df["Fecha"] >= f_inicio].copy()
        df_desde["DGA_cum"] = df_desde["DG"].cumsum()
        df_res = df_desde[df_desde["DGA_cum"] >= dga_optimo]
        if not df_res.empty: fecha_control = df_res.iloc[0]["Fecha"]
        
    # --- MÉTRICAS DE VALIDACIÓN ---
    if archivo_campo is not None:
        df_campo = pd.read_csv(archivo_campo) if archivo_campo.name.endswith('.csv') else pd.read_excel(archivo_campo)
        df_campo.columns = [c.upper().strip() for c in df_campo.columns]
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])
        
        # 1. Pearson Flexible (Margen ±3 días)
        sim_flex = []
        for f_obs in df_campo['FECHA']:
            ventana = (df['Fecha'] >= f_obs - timedelta(days=3)) & (df['Fecha'] <= f_obs + timedelta(days=3))
            sim_flex.append(df.loc[ventana, 'EMERREL_FLEX'].max())
        df_campo['Sim_Flexible'] = sim_flex
        pearson_r = df_campo['PLM2'].corr(df_campo['Sim_Flexible'])
        
        # 2. PEC Estricto (A la fecha de aplicación)
        pec = 0.0
        if fecha_control:
            total_real = df_campo['PLM2'].sum()
            alcanzadas = df_campo.loc[df_campo['FECHA'] <= fecha_control, 'PLM2'].sum()
            pec = (alcanzadas / total_real) * 100 if total_real > 0 else 0

    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM BORDENAVE - vK4.7")
    
    if archivo_campo is not None:
        st.markdown("<p class='metric-header'>🚜 DIAGNÓSTICO DE VALIDACIÓN</p>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Sincronía (Pearson r)", f"{pearson_r:.3f}", "Margen ±3d")
        k2.metric("Eficiencia (PEC)", f"{pec:.1f}%", "Al día de control")
        if fecha_control:
            k3.metric("Fecha Aplicación", fecha_control.strftime('%d-%m'))
        st.divider()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Modelo (Tasa Diaria)", line=dict(color='#166534', width=2.5)))
    
    if archivo_campo is not None:
        df_campo['PL_NORM'] = df_campo['PLM2'] / df_campo['PLM2'].max() * df['EMERREL'].max()
        fig.add_trace(go.Scatter(x=df_campo["FECHA"], y=df_campo['PL_NORM'], mode='markers', name="Campo (Escalado)", marker=dict(size=12, color='red', symbol='diamond')))

    if fecha_control:
        fig.add_vline(x=fecha_control.timestamp() * 1000, line_dash="dot", line_color="red", line_width=3, annotation_text="APLICACIÓN")
        fin_res = fecha_control + timedelta(days=residualidad)
        fig.add_vrect(x0=fecha_control.timestamp() * 1000, x1=fin_res.timestamp() * 1000, fillcolor="blue", opacity=0.1, layer="below", annotation_text=f"Residualidad {residualidad}d")

    fig.update_layout(height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Bienvenido. Por favor, cargue los archivos de clima y campo en la barra lateral.")
