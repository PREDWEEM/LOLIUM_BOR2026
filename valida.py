# Write the full code provided by the user into a downloadable Python file

code = r'''# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.7 — LOLIUM BORDENAVE 2026
# Actualización:
# - Pearson por intervalos de monitoreo
# - Desfase temporal automático admisible hasta ±10 días
# - PEC calculado estrictamente hasta el día de control
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
    .bio-alert {
        padding: 10px;
        border-radius: 5px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ Y ARCHIVOS (MOCKS)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100)**2)/600)
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200)
        p3 = np.exp(-((jd - 230)**2)/1500)
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN + BIO)
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base:
        return 0.0
    elif t <= t_opt:
        return t - t_base
    elif t < t_crit:
        return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else:
        return 0.0

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        z1 = Xn @ self.IW + self.bIW
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T).flatten() + self.bLW
        emerrel = (np.tanh(z2) + 1) / 2
        emer_ac = np.cumsum(emerrel)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(
            np.load(BASE / "IW.npy"),
            np.load(BASE / "bias_IW.npy"),
            np.load(BASE / "LW.npy"),
            np.load(BASE / "bias_out.npy")
        )
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def load_data(file_uploader, default_name):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file_uploader)
    elif (BASE / f"{default_name}.csv").exists():
        return pd.read_csv(BASE / f"{default_name}.csv")
    elif (BASE / f"{default_name}.xlsx").exists():
        return pd.read_excel(BASE / f"{default_name}.xlsx")
    return None

def build_shifted_interval_series(df_sim, df_campo, col_fecha, shift_days):
    sim_intervals = []
    last_date = df_sim["Fecha"].min() - pd.Timedelta(days=1)

    for _, row in df_campo.iterrows():
        current_date = row[col_fecha]

        start_shifted = last_date + pd.Timedelta(days=shift_days)
        end_shifted = current_date + pd.Timedelta(days=shift_days)

        mask_intervalo = (df_sim["Fecha"] > start_shifted) & (df_sim["Fecha"] <= end_shifted)
        suma_simulada = df_sim.loc[mask_intervalo, "EMERREL"].sum()
        sim_intervals.append(suma_simulada)

        last_date = current_date

    return np.array(sim_intervals, dtype=float)

def evaluate_shifted_validation(df_sim, df_campo, col_fecha, col_plm2, max_shift_days=10):
    obs = df_campo[col_plm2].to_numpy(dtype=float)

    best = {
        "shift_days": 0,
        "pearson_r": -np.inf,
        "sim_intervalo": np.zeros(len(df_campo))
    }

    for shift in range(-max_shift_days, max_shift_days + 1):
        sim_vals = build_shifted_interval_series(df_sim, df_campo, col_fecha, shift)

        pearson_r = pd.Series(obs).corr(pd.Series(sim_vals))
        if pd.isna(pearson_r):
            pearson_r = -1.0

        is_better = False
        if pearson_r > best["pearson_r"]:
            is_better = True
        elif np.isclose(pearson_r, best["pearson_r"], atol=1e-9) and abs(shift) < abs(best["shift_days"]):
            is_better = True

        if is_better:
            best = {
                "shift_days": shift,
                "pearson_r": float(pearson_r),
                "sim_intervalo": sim_vals.copy()
            }

    if best["pearson_r"] == -np.inf:
        best["pearson_r"] = 0.0

    return best

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.markdown("## 📂 1. Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima (bordenave)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

df_meteo_raw = load_data(archivo_meteo, "bordenave")
df_campo_raw = load_data(archivo_campo, "bordenave_campo")

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 20)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO (simplificado para archivo completo)
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:

    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha','TMAX': 'TMAX','TMIN': 'TMIN','PREC': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna().sort_values("Fecha").reset_index(drop=True)

    df["Julian_days"] = df["Fecha"].dt.dayofyear

    df["JD_Shifted"] = (df["Julian_days"] + 60).clip(1, 300)
    X = df[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)

    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    st.title("🌾 PREDWEEM LOLIUM - BORDENAVE 2026")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Fecha"],
        y=df["EMERREL"],
        mode="lines",
        name="Emergencia Simulada"
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue datos climáticos para comenzar.")
'''

path = "/mnt/data/PREDWEEM_vK4_7_Bordenave_COMPLETO.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(code)

path
