# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026 (Versión Adaptada)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle, io
from pathlib import Path

# ---------------------------------------------------------
# 0. GENERADOR DE ARCHIVOS MOCK (Ajustado con nuevas ventanas temporales)
# ---------------------------------------------------------
BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

def create_mock_files_if_missing():
    """Genera patrones con rangos flexibles y picos optimizados para Bordenave."""
    
    # Archivos de la ANN
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    # Modelo de Clusters K3
    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        
        # 1. PATRÓN TEMPRANO (Otoño explosivo)
        # Pico centrado exactamente en el día 90 (Marzo)
        p1 = np.exp(-((jd - 90)**2)/600)  

        # 2. PATRÓN BIMODAL (Rango flexible Otoño-Invierno)
        # Pico 1: Centrado en 105 para cubrir la ventana 80-130 (Varianza 1200)
        # Pico 2: Centrado en 185 para cubrir la ventana 170-200 (Varianza 1200)
        p1_bimodal = np.exp(-((jd - 105)**2)/1200)
        p2_bimodal = 0.5 * np.exp(-((jd - 185)**2)/1200) 
        p2 = p1_bimodal + p2_bimodal

        # 3. PATRÓN TARDÍO (Otoño seco / Escape primaveral)
        # Centrado en el día 245 (Agosto) para evitar solapamiento con el Bimodal en 200
        p3 = np.exp(-((jd - 245)**2)/1500) 

        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3], # [0]=Bimodal, [1]=Temprano, [2]=Tardío
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

    # Clima Mock
    if not (BASE / "meteo_daily.csv").exists():
        dates = pd.date_range(start="2026-01-01", periods=180)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(20, 30, size=180),
            "TMIN": np.random.uniform(5, 15, size=180),
            "Prec": np.random.choice([0, 0, 10, 25], size=180)
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

# Ejecutar generador
create_mock_files_if_missing()

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
    except: return None, None

# ---------------------------------------------------------
# 2. CONFIGURACIÓN UI
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – Bordenave", layout="wide", page_icon="🌾")
st.sidebar.title("🌾 PREDWEEM vK3")
umbral_alerta = st.sidebar.slider("Umbral de Alerta", 0.1, 1.0, 0.5)
archivo_subido = st.sidebar.file_uploader("Subir Clima", type=["xlsx", "csv"])

modelo_ann, cluster_model = load_models()

def get_data(file_input):
    if file_input:
        df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
    else:
        path = BASE / "meteo_daily.csv"
        df = pd.read_csv(path, parse_dates=["Fecha"]) if path.exists() else None
    if df is not None:
        df.columns = [c.upper().strip() for c in df.columns]
        df = df.rename(columns={'FECHA': 'Fecha', 'LLUVIA': 'Prec', 'PREC': 'Prec'})
    return df

df = get_data(archivo_subido)

# ---------------------------------------------------------
# 3. DASHBOARD PRINCIPAL
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Predicción
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 30, "EMERREL"] = 0.0 

    st.title("🌾 PREDWEEM vK3 — BORDENAVE 2026")
    
    # Gráficos de Pulsos Diarios
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], fill='tozeroy', line_color='#15803d', name="Tasa Diaria"))
    fig_m.add_hline(y=umbral_alerta, line_dash="dash", line_color="red")
    fig_m.update_layout(height=300, title="Dinámica de Emergencia Diaria Predicha")
    st.plotly_chart(fig_m, use_container_width=True)

    # Análisis Funcional
    st.divider()
    st.header("📊 Análisis de Patrones Estacionales")
    
    # Corte al 1 de Mayo para clasificación
    fecha_corte = pd.Timestamp("2026-05-01")
    df_cuat = df[df["Fecha"] < fecha_corte].copy()

    if not df_cuat.empty:
        jd_corte = df_cuat["Julian_days"].max()
        JD_COMMON = cluster_model["JD_common"]
        jd_obs_grid = JD_COMMON[JD_COMMON <= jd_corte]
        
        max_e = df_cuat["EMERREL"].max() if df_cuat["EMERREL"].max() > 0 else 1
        curva_obs_norm = np.interp(jd_obs_grid, df_cuat["Julian_days"], df_cuat["EMERREL"] / max_e)
        
        dists = []
        meds = cluster_model["curves_interp"]
        for m in meds:
            m_slice = m[JD_COMMON <= jd_corte]
            m_slice_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
            dists.append(dtw_distance(curva_obs_norm, m_slice_norm))
            
        cluster_pred = np.argmin(dists)
        nombres = {0: "🌾 Bimodal (Otoño-Invierno)", 1: "🌱 Temprano (Otoño Explosivo)", 2: "🍂 Tardío (Escape Primaveral)"}
        colores = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"#### Patrón Detectado: <span style='color:{colores[cluster_pred]}'>{nombres[cluster_pred]}</span>", unsafe_allow_html=True)
            fig_p, ax = plt.subplots(figsize=(10, 4))
            ax.plot(JD_COMMON, meds[cluster_pred], color=colores[cluster_pred], ls="--", alpha=0.4, label="Referencia Patrón")
            factor = meds[cluster_pred].max()
            ax.plot(jd_obs_grid, curva_obs_norm * factor, color="black", lw=2.5, label="Observado (Ene-Abr)")
            ax.axvline(jd_corte, color="red", ls=":", label="Corte 1 de Mayo")
            ax.set_xlabel("Día Juliano")
            ax.legend()
            st.pyplot(fig_p)
        with c2:
            st.metric("Similitud (DTW)", f"{min(dists):.2f}")
            st.info(f"El sistema detecta un comportamiento tipo {nombres[cluster_pred]}.")

    # Descarga
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "predweem_2026.xlsx")

else:
    st.warning("Cargue datos para activar el modelo.")
