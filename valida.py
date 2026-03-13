# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.5 — LOLIUM BORDENAVE 2026
# Actualización: Sincronía con Margen Flexible de 7 días (±3d)
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
# 1. CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM BORDENAVE vK4.5", layout="wide", page_icon="🌾")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
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
        ann = PracticalANNModel(np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"), np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy"))
        return ann
    except: return None

# ---------------------------------------------------------
# 3. INTERFAZ
# ---------------------------------------------------------
modelo_ann = load_models()
st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png", use_container_width=True)

st.sidebar.markdown("## 📂 Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo", type=["xlsx", "csv"])

# Parámetros logísticos
dga_optimo = st.sidebar.number_input("TT Control Post-em (°Cd)", value=600, step=10)
umbral_er = st.sidebar.slider("Umbral Alerta", 0.05, 0.80, 0.15)

if archivo_meteo is not None and modelo_ann is not None:
    # --- PROCESAMIENTO CLIMA ---
    df = pd.read_csv(archivo_meteo) if archivo_meteo.name.endswith('.csv') else pd.read_excel(archivo_meteo)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # --- PREDICCIÓN MODELO ---
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)
    
    # Suavizado para evaluación de sincronía (Ventana de 7 días)
    df["EMERREL_SMOOTH"] = df["EMERREL"].rolling(window=7, center=True, min_periods=1).mean()

    # --- BIO-TÉRMICO ---
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, 2.0, 20.0, 30.0))
    
    # Ventana de Control
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_control = None
    if indices_pulso:
        f_inicio = df.loc[indices_pulso[0], "Fecha"]
        df_desde = df[df["Fecha"] >= f_inicio].copy()
        df_desde["DGA_cum"] = df_desde["DG"].cumsum()
        df_res = df_desde[df_desde["DGA_cum"] >= dga_optimo]
        if not df_res.empty: fecha_control = df_res.iloc[0]["Fecha"]

    # --- VALIDACIÓN FLEXIBLE (MARGEN 7 DÍAS) ---
    if archivo_campo is not None:
        df_campo = pd.read_csv(archivo_campo) if archivo_campo.name.endswith('.csv') else pd.read_excel(archivo_campo)
        df_campo.columns = [c.upper().strip() for c in df_campo.columns]
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])
        
        # Sincronía Flexible: Buscamos el mejor match del modelo en ±3 días de cada observación
        sim_flex = []
        for f_obs in df_campo['FECHA']:
            mask = (df['Fecha'] >= f_obs - timedelta(days=3)) & (df['Fecha'] <= f_obs + timedelta(days=3))
            # Tomamos el valor de la curva suavizada en esa ventana
            val_sim = df.loc[mask, 'EMERREL_SMOOTH'].mean()
            sim_flex.append(val_sim)
        
        df_campo['Sim_Flexible'] = sim_flex
        pearson_r = df_campo['PLM2'].corr(df_campo['Sim_Flexible'])
        
        # PEC (Control Efectivo)
        total_campo = df_campo['PLM2'].sum()
        if fecha_control:
            cont_campo = df_campo.loc[df_campo['FECHA'] <= fecha_control, 'PLM2'].sum()
            pec = (cont_campo / total_campo) * 100 if total_campo > 0 else 0

    # --- DASHBOARD ---
    st.title("🌾 PREDWEEM - LARTIGAU / BORDENAVE")
    
    if archivo_campo is not None and fecha_control:
        st.markdown("<p class='metric-header'>🚜 VALIDACIÓN AGRONÓMICA (Margen ±3 días)</p>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Eficiencia (PEC)", f"{pec:.1f}%")
        k2.metric("Sincronía (Pearson r)", f"{pearson_r:.3f}", "Con margen de 7 días")
        k3.metric("Fecha Control", fecha_control.strftime('%d-%m-%Y'))
        st.info("💡 El cálculo de sincronía ahora asume una tolerancia de 7 días entre el pico predicho y la observación de campo.")

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Modelo (Tasa Diaria)", line=dict(color='#166534', width=2)))
    if archivo_campo is not None:
        # Normalizamos campo para visualización
        df_campo['PL_NORM'] = df_campo['PLM2'] / df_campo['PLM2'].max() * df['EMERREL'].max()
        fig.add_trace(go.Scatter(x=df_campo["FECHA"], y=df_campo['PL_NORM'], mode='markers', name="Campo (Escalado)", marker=dict(size=10, color='red', symbol='diamond')))
    
    if fecha_control:
        fig.add_vline(x=fecha_control.timestamp() * 1000, line_dash="dot", line_color="red", annotation_text="APLICACIÓN")

    st.plotly_chart(fig, use_container_width=True)
