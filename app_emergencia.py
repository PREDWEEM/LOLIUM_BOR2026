# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, io
from pathlib import Path
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIG STREAMLIT + ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – LOLIUM BORDENAVE 2026", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.stAppDeployButton {display: none;}
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
        st.error(f"Error cargando archivos de modelo: {e}")
        return None, None

# ===============================================================
# 📂 GESTIÓN DE DATOS (CARGA Y DESCARGA)
# ===============================================================
st.sidebar.header("📂 Gestión de Datos")
uploaded_file = st.sidebar.file_uploader("Subir Clima (Excel o CSV)", type=["xlsx", "csv"])

def get_data(file_input):
    try:
        if file_input is not None:
            if file_input.name.endswith('.csv'):
                return pd.read_csv(file_input, parse_dates=["Fecha"])
            else:
                return pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path_fixed = BASE / "meteo_daily.csv"
            if path_fixed.exists():
                return pd.read_csv(path_fixed, parse_dates=["Fecha"])
            return None
    except Exception as e:
        st.error(f"Error al leer datos: {e}")
        return None

# Carga de modelos e inicio de lógica
modelo_ann, cluster_model = load_models()
df = get_data(uploaded_file)

if df is not None and modelo_ann is not None:
    # Limpieza y preparación
    cols_necesarias = ["Fecha", "TMAX", "TMIN", "Prec"]
    if not all(col in df.columns for col in cols_necesarias):
        st.error(f"El archivo debe contener las columnas: {cols_necesarias}")
        st.stop()
        
    df = df.dropna(subset=cols_necesarias).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Predicción ANN
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    df["EMERAC"] = df["EMERREL"].cumsum()
    
    # Cálculo de Riesgo
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    # BOTÓN DE DESCARGA EN SIDEBAR
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
    
    st.sidebar.download_button(
        label="📥 Descargar Predicciones (Excel)",
        data=output.getvalue(),
        file_name="predicciones_lolium_predweem.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ===============================================================
    # 🖥️ VISUALIZACIÓN
    # ===============================================================
    st.title("🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026")
    
    # Mapa de Riesgo
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale='Viridis', zmin=0, zmax=1,
        hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>"))
    fig_risk.update_layout(height=200, title="Evolución del Riesgo de Emergencia", margin=dict(t=40, b=10))
    st.plotly_chart(fig_risk, use_container_width=True)

    # Clasificación Funcional
    st.divider()
    st.header("🌾 Análisis Funcional de Patrones")

    UMBRAL_RELEVANCIA = 0.10
    if max_er < UMBRAL_RELEVANCIA:
        st.warning(f"⚠️ Pico máximo ({max_er:.3f}) por debajo del umbral de {UMBRAL_RELEVANCIA}. No se asigna patrón funcional.")
    else:
        # Lógica DTW
        JD_COMMON = cluster_model["JD_common"]
        curves_interp = cluster_model["curves_interp"]
        meds_idx = cluster_model["medoids_k3"]
        
        emer_norm = df["EMERREL"].to_numpy() / max_er
        curve_year_interp = np.interp(JD_COMMON, df["Julian_days"], emer_norm)
        
        meds = [curves_interp[i] for i in meds_idx]
        dists = [dtw_distance(curve_year_interp, m) for m in meds]
        cluster_pred = np.argmin(dists)

        names = {0: "🌾 Intermedio / Bimodal", 1: "🌱 Temprano / Compacto", 2: "🍂 Tardío / Extendido"}
        colors = {0: "blue", 1: "green", 2: "orange"}
        
        st.markdown(f"### Patrón Asignado: <span style='color:{colors[cluster_pred]};'>{names[cluster_pred]}</span>", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_cmp, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(JD_COMMON, curve_year_interp, label="Datos Actuales", color="black", lw=2)
            ax.plot(JD_COMMON, meds[cluster_pred], label="Patrón de Referencia", color=colors[cluster_pred], ls="--")
            ax.set_title("Ajuste a Patrones Históricos")
            ax.legend()
            st.pyplot(fig_cmp)
        with c2:
            cert = 1 - (min(dists) / sum(dists))
            st.metric("Certidumbre de Patrón", f"{cert:.1%}")
            st.info(f"El patrón '{names[cluster_pred]}' es el que mejor describe la forma de emergencia en este lote.")

    with st.expander("🔍 Ver tabla de datos"):
        st.dataframe(df.style.format(precision=3))

else:
    st.warning("👈 Por favor, sube un archivo o coloca 'meteo_daily.csv' en la carpeta raíz.")

st.sidebar.markdown("---")
st.sidebar.caption("Formato: Fecha, TMAX, TMIN, Prec")
