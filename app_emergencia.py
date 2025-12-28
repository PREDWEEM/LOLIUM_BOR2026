# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, requests, xml.etree.ElementTree as ET
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIG STREAMLIT + ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM vK3 – LOLIUM BORDENAVE 2026",
    layout="wide",
)

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
# 🔧 FUNCIONES AUXILIARES Y ANN
# ===============================================================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

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
def load_ann():
    return PracticalANNModel(
        np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
        np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
    )

def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

# ===============================================================
# 📊 PROCESAMIENTO DE DATOS
# ===============================================================
modelo_ann = safe(load_ann, "Error cargando pesos ANN")
if not modelo_ann: st.stop()

path_daily = BASE / "meteo_daily.csv"
if not path_daily.exists():
    st.error("❌ No se encontró meteo_daily.csv")
    st.stop()

df = pd.read_csv(path_daily, parse_dates=["Fecha"]).dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
df["Julian_days"] = df["Fecha"].dt.dayofyear

# Predicción ANN
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel, _ = modelo_ann.predict(X)
df["EMERREL"] = np.maximum(emerrel, 0.0)
df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
df["EMERAC"] = df["EMERREL"].cumsum()

# ===============================================================
# 🔥 SECCIÓN 1: RIESGO Y VISUALIZACIÓN
# ===============================================================
st.title("🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026")

max_er = df["EMERREL"].max()
df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

fig_risk = go.Figure(data=go.Heatmap(
    z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
    colorscale='Viridis', zmin=0, zmax=1, showscale=True,
    hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>"))
fig_risk.update_layout(height=200, title="Mapa interactivo de riesgo diario", margin=dict(t=40, b=20))
st.plotly_chart(fig_risk, use_container_width=True)

# ===============================================================
# 🌾 SECCIÓN 2: CLASIFICACIÓN FUNCIONAL K=3
# ===============================================================
st.header("🌾 Clasificación funcional K=3 (DTW)")

# Carga de modelo clustering
@st.cache_resource
def load_k3():
    with open(BASE/"modelo_clusters_k3.pkl", "rb") as f: return pickle.load(f)

c_model = safe(load_k3, "Error modelo K3")
if not c_model: st.stop()

JD_COMMON = c_model["JD_common"]
curves_interp = c_model["curves_interp"]
medoids_idx = c_model["medoids_k3"]

# Umbral de seguridad
UMBRAL_RELEVANCIA = 0.10
max_actual = df["EMERREL"].max()

if max_actual < UMBRAL_RELEVANCIA:
    st.warning(f"⚠️ **Señal débil ({max_actual:.3f}):** Emergencia insuficiente para clasificación funcional (Mínimo requerido: {UMBRAL_RELEVANCIA}).")
    ignorar_analisis = True
else:
    ignorar_analisis = False
    # Procesar curva del año
    emer_norm = df["EMERREL"].to_numpy() / max_actual
    curve_year_interp = np.interp(JD_COMMON, df["Julian_days"], emer_norm)
    
    # Medoides
    meds = [curves_interp[i] for i in medoids_idx]
    dists = [dtw_distance(curve_year_interp, m) for m in meds]
    cluster_pred = np.argmin(dists)

    # UI Resultados
    col_names = {0: "🌾 Intermedio / Bimodal", 1: "🌱 Temprano / Compacto", 2: "🍂 Tardío / Extendido"}
    col_colors = {0: "blue", 1: "green", 2: "orange"}
    
    st.markdown(f"### Patrón: <span style='color:{col_colors[cluster_pred]};'>{col_names[cluster_pred]}</span>", unsafe_allow_html=True)

    # --- Análisis y Gráficos (Solo si hay señal) ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("📊 Comparativa de Curvas")
        fig_cmp, ax = plt.subplots()
        ax.plot(JD_COMMON, curve_year_interp, label="Año Actual", color="black", lw=2)
        ax.plot(JD_COMMON, meds[cluster_pred], label="Patrón Histórico", color=col_colors[cluster_pred], ls="--")
        ax.set_ylabel("Emergencia Normalizada")
        ax.legend()
        st.pyplot(fig_cmp)

    with c2:
        st.subheader("📋 Implicancias de Manejo")
        desc = {
            1: "Enfoque en **residuales de febrero**. Ventana corta.",
            0: "Estrategia **bimodal**. Control temprano + refuerzo en mayo.",
            2: "Control **tardío**. Monitoreo extendido hasta otoño-invierno."
        }
        st.info(desc.get(cluster_pred))

# ===============================================================
# 🔮 SECCIÓN 3: DIAGNÓSTICO ANTICIPADO
# ===============================================================
st.divider()
st.header("🔮 Diagnóstico Anticipado")

if max_actual < UMBRAL_RELEVANCIA:
    st.info("ℹ️ Esperando pulsos de emergencia significativos para diagnóstico anticipado.")
else:
    # Reutilizamos cálculos de DTW para certidumbre
    cert = 1 - (min(dists) / sum(dists))
    color_c = "green" if cert > 0.55 else "orange" if cert > 0.4 else "red"
    
    mc1, mc2 = st.columns(2)
    mc1.metric("Confianza del Diagnóstico", f"{cert*100:.1f}%")
    mc2.write(f"Estado: **:{color_c}[{ 'Alta Consistencia' if cert > 0.55 else 'Señal en Evolución' }]**")
    st.progress(min(max(cert, 0.0), 1.0))

st.caption("PREDWEEM vK3 - INTA Bordenave 2026")


