# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026 (ACTUALIZADO)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle, io, os
from pathlib import Path

# ---------------------------------------------------------
# 0. ROBUSTNESS: GENERADOR DE ARCHIVOS MOCK
# ---------------------------------------------------------
BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

def create_mock_files():
    """Genera o regenera archivos base con el patrón bimodal desplazado (> día 50)."""
    # 1. Pesos de la Red Neuronal (Mock)
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    # 2. Modelo de Clusters (Ajustado según requerimiento)
    # IMPORTANTE: Se fuerza la recreación si el usuario necesita actualizar la forma
    jd = np.arange(1, 366)
    
    # --- PATRÓN 0: BIMODAL DESPLAZADO ---
    # Centramos el primer pico en el día 90 para que la emergencia sea notable post-día 50
    p0_pico1 = np.exp(-((jd - 90)**2) / 600)  
    p0_pico2 = 0.4 * np.exp(-((jd - 250)**2) / 1000) 
    p0 = (p0_pico1 + p0_pico2)
    p0 = p0 / p0.max()

    # Patrón 1: Temprano / Compacto
    p1 = np.exp(-((jd - 110)**2) / 600)
    
    # Patrón 2: Tardío / Extendido
    p2 = np.exp(-((jd - 230)**2) / 1500)
    
    mock_cluster = {
        "JD_common": jd,
        "curves_interp": [p0, p1, p2],
        "medoids_k3": [0, 1, 2]
    }
    
    with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
        pickle.dump(mock_cluster, f)

    # 3. Clima base
    if not (BASE / "meteo_daily.csv").exists():
        dates = pd.date_range(start="2026-01-01", periods=180)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(20, 30, size=180),
            "TMIN": np.random.uniform(5, 15, size=180),
            "Prec": np.random.choice([0, 0, 10, 25], size=180)
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

# Ejecutar creación de archivos
create_mock_files()

# ---------------------------------------------------------
# 1. MODELOS Y LÓGICA TÉCNICA
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. CONFIGURACIÓN DE PÁGINA Y DATOS
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – Bordenave", layout="wide", page_icon="🌾")

st.sidebar.title("🌾 PREDWEEM vK3")
st.sidebar.info("Patrón Bimodal ajustado: Inicio post-día 50.")

umbral_alerta = st.sidebar.slider("Umbral de Alerta", 0.1, 1.0, 0.5, 0.05)
archivo_subido = st.sidebar.file_uploader("Subir Clima", type=["xlsx", "csv"])

def get_data(file_input):
    if file_input:
        df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
    else:
        path = BASE / "meteo_daily.csv"
        df = pd.read_csv(path, parse_dates=["Fecha"]) if path.exists() else None
    
    if df is not None:
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
        df = df.rename(columns=mapeo)
    return df

modelo_ann, cluster_model = load_models()
df = get_data(archivo_subido)

# ---------------------------------------------------------
# 3. PROCESAMIENTO Y DASHBOARD
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Predicción ANN
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    
    # --- AJUSTE SOLICITADO: Filtro biológico post-día 50 ---
    df.loc[df["Julian_days"] <= 50, "EMERREL"] = 0.0 
    
    st.title("🌾 PREDWEEM vK3 — BORDENAVE 2026")

    # Gráfico de Intensidad
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], fill='tozeroy', line_color='#15803d', name="Emergencia"))
    fig_m.add_hline(y=umbral_alerta, line_dash="dash", line_color="red")
    fig_m.update_layout(height=350, title="Dinámica de Emergencia (Ajustada inicio día 50)")
    st.plotly_chart(fig_m, use_container_width=True)

    # Análisis Funcional
    st.divider()
    st.header("📊 Clasificación de Patrón")
    
    fecha_corte = pd.Timestamp("2026-05-01")
    df_cuatrimestre = df[df["Fecha"] < fecha_corte].copy()

    if not df_cuatrimestre.empty:
        jd_corte = df_cuatrimestre["Julian_days"].max()
        JD_COMMON = cluster_model["JD_common"]
        jd_obs_grid = JD_COMMON[JD_COMMON <= jd_corte]
        
        max_val = df_cuatrimestre["EMERREL"].max() if df_cuatrimestre["EMERREL"].max() > 0 else 1
        curva_obs_norm = np.interp(jd_obs_grid, df_cuatrimestre["Julian_days"], df_cuatrimestre["EMERREL"] / max_val)
        
        dists = []
        meds = cluster_model["curves_interp"]
        for m in meds:
            m_slice = m[JD_COMMON <= jd_corte]
            m_slice_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
            dists.append(dtw_distance(curva_obs_norm, m_slice_norm))
        
        cluster_pred = np.argmin(dists)
        nombres = {0: "🌾 Bimodal (Post-Día 50)", 1: "🌱 Temprano", 2: "🍂 Tardío"}
        
        st.subheader(f"Resultado: {nombres[cluster_pred]}")
        
        fig_p, ax = plt.subplots(figsize=(10, 3))
        ax.plot(JD_COMMON, meds[cluster_pred], color="blue", ls="--", label="Patrón Referencia")
        ax.fill_between(jd_obs_grid, curva_obs_norm, color="green", alpha=0.3, label="Observado")
        ax.axvline(50, color="red", alpha=0.3, label="Límite Día 50")
        ax.legend()
        st.pyplot(fig_p)

    # Exportar
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("📥 Descargar Resultados", output.getvalue(), "prediccion_bordenave.xlsx")
else:
    st.warning("Cargue datos para iniciar el análisis.")
