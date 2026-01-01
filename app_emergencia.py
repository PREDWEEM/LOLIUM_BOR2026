# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import io
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURACIÓN Y ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – LOLIUM 2026", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    .stAppDeployButton {display: none;}
    .main { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 MODELOS Y FUNCIONES TÉCNICAS
# ===============================================================
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

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
        emer = (np.array(emer) + 1) / 2
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
        return None, None

def get_data(uploaded_file):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file, parse_dates=["Fecha"])
            else:
                return pd.read_excel(uploaded_file, parse_dates=["Fecha"])
        else:
            path_fixed = BASE / "meteo_daily.csv"
            if path_fixed.exists():
                mtime = datetime.fromtimestamp(path_fixed.stat().st_mtime)
                st.sidebar.info(f"📅 Actualizado: {mtime.strftime('%d/%m %H:%M')}")
                return pd.read_csv(path_fixed, parse_dates=["Fecha"])
            return None
    except Exception as e:
        return None

# Carga inicial
st.sidebar.header("📂 Gestión de Datos")
uploaded_file = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])

modelo_ann, cluster_model = load_models()
df = get_data(uploaded_file)

# ===============================================================
# 🖥️ INTERFAZ PRINCIPAL
# ===============================================================
st.title("🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026")

if df is not None and modelo_ann is not None:
    # 1. Preparación de datos
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # 2. Predicción ANN (Funciona todo el año)
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0 # Bloqueo biológico inicial
    
    df["EMERAC"] = df["EMERREL"].cumsum()
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    # 3. Visualización de Riesgo y Clima
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale='YlOrRd', zmin=0, zmax=1))
    fig_risk.update_layout(height=180, title="Evolución del Riesgo de Emergencia", margin=dict(t=40, b=10))
    st.plotly_chart(fig_risk, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df["Fecha"], y=df["TMAX"], name="T Máx", line=dict(color='red')))
        fig_temp.add_trace(go.Scatter(x=df["Fecha"], y=df["TMIN"], name="T Mín", line=dict(color='blue')))
        fig_temp.update_layout(title="Temperaturas (°C)", height=250)
        st.plotly_chart(fig_temp, use_container_width=True)
    with c2:
        fig_prec = go.Figure(go.Bar(x=df["Fecha"], y=df["Prec"], marker_color="teal", name="Lluvia"))
        fig_prec.update_layout(title="Precipitaciones (mm)", height=250)
        st.plotly_chart(fig_prec, use_container_width=True)

    # ===============================================================
    # 🛡️ JAULA ANTI-ERROR: ANÁLISIS DE PATRONES
    # ===============================================================
    st.divider()
    st.header("🌾 Análisis Funcional de Patrones")

    # Calculamos el mes del último dato para decidir si mostramos el análisis
    ultima_fecha = df["Fecha"].max()
    
    if ultima_fecha.month == 1:
        # SI ES ENERO: Mostramos mensaje informativo y evitamos tocar el cluster_model
        st.info(f"📅 **Fase de Recolección de Datos: ENERO**")
        st.write(f"Día actual registrado: **{ultima_fecha.strftime('%d/%m/%Y')}**")
        st.write("El análisis de patrones comparativos (DTW) se habilitará automáticamente el **1 de febrero**, "
                 "cuando existan datos suficientes en el rango de emergencia esperado (Feb-Sep).")
        st.progress(ultima_fecha.day / 31)
    else:
        # SI NO ES ENERO: Ejecutamos la lógica de clusters protegida
        try:
            if cluster_model is not None:
                JD_COMMON = cluster_model["JD_common"]
                # Solo ejecutamos si ya entramos en el rango del JD_COMMON
                if df["Julian_days"].max() >= JD_COMMON[0]:
                    curves_interp = cluster_model["curves_interp"]
                    meds_idx = cluster_model["medoids_k3"]
                    
                    emer_norm = df["EMERREL"].to_numpy() / (max_er if max_er > 0 else 1)
                    curve_year_interp = np.interp(JD_COMMON, df["Julian_days"], emer_norm)
                    
                    meds = [curves_interp[i] for i in meds_idx]
                    dists = [dtw_distance(curve_year_interp, m) for m in meds]
                    cluster_pred = np.argmin(dists)

                    names = {0: "🌾 Intermedio", 1: "🌱 Temprano", 2: "🍂 Tardío"}
                    colors = {0: "blue", 1: "green", 2: "orange"}
                    
                    st.markdown(f"### Patrón Detectado: <span style='color:{colors[cluster_pred]};'>{names[cluster_pred]}</span>", unsafe_allow_html=True)
                    
                    fig_cmp, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(JD_COMMON, curve_year_interp, label="Campaña 2026", color="black", lw=2)
                    ax.plot(JD_COMMON, meds[cluster_pred], label="Referencia Histórica", color=colors[cluster_pred], ls="--")
                    ax.legend()
                    st.pyplot(fig_cmp)
                else:
                    st.warning("Esperando alcanzar el día inicial del modelo de referencia.")
        except Exception as e:
            st.caption("Análisis de patrones en espera de más datos estacionales.")

    # Botón de descarga siempre visible si hay datos
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
    st.sidebar.download_button(label="📥 Descargar Predicciones", data=output.getvalue(), 
                               file_name="predicciones_2026.xlsx")

else:
    st.warning("Esperando sincronización de datos desde GitHub Actions...")
    if st.button("Verificar archivos"):
        st.write(f"Archivo 'meteo_daily.csv': {'✅' if (BASE/'meteo_daily.csv').exists() else '❌'}")

st.sidebar.markdown("---")
st.sidebar.caption("PREDWEEM vK3 | Bordenave")
