# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.5 — LOLIUM BORDENAVE 2026
# Actualización:
# - Pearson por intervalos de monitoreo
# - Emparejamiento por Proximidad con Regla Anti-Cruce
# - CORRECCIÓN DEFINITIVA: Eliminación total de réplicas (Ecos) del análisis.
# - SELECCIÓN DE PICO: En flushes < 7 días, se prioriza el más cercano al dato de campo.
# - NUEVO MATCH N-A-1: Observaciones de la "rampa de subida" pueden emparejarse al mismo pico simulado.
# - NUEVO: TN asimétrico. Match de Campo < 0.05 con Simulación < 0.30
# - Mantenimiento de la Arquitectura ANN y Shifts específicos de Bordenave
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from datetime import timedelta
from pathlib import Path
from scipy.signal import find_peaks

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM BORDENAVE vK4.9.5",
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
        mock_cluster = {"JD_common": jd, "curves_interp": [p2, p1, p3], "medoids_k3": [0, 1, 2]}
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN BORDENAVE + BIO)
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf); dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0]); self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        z1 = Xn @ self.IW + self.bIW
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T).flatten() + self.bLW
        emerrel = (np.tanh(z2) + 1) / 2
        return emerrel, np.cumsum(emerrel)

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"), np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy"))
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f: k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}"); return None, None

def load_data(file_uploader, default_name):
    if file_uploader: return pd.read_excel(file_uploader) if file_uploader.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file_uploader)
    elif (BASE / f"{default_name}.csv").exists(): return pd.read_csv(BASE / f"{default_name}.csv")
    return None

def build_shifted_interval_series(df_sim, df_campo, col_fecha, shift_days):
    sim_intervals = []; last_date = df_sim["Fecha"].min() - pd.Timedelta(days=1)
    for _, row in df_campo.iterrows():
        current_date = row[col_fecha]
        start_shifted = last_date + pd.Timedelta(days=shift_days)
        end_shifted = current_date + pd.Timedelta(days=shift_days)
        mask = (df_sim["Fecha"] > start_shifted) & (df_sim["Fecha"] <= end_shifted)
        sim_intervals.append(df_sim.loc[mask, "EMERREL"].sum())
        last_date = current_date
    return np.array(sim_intervals, dtype=float)

def evaluate_shifted_validation(df_sim, df_campo, col_fecha, col_plm2, max_shift_days=10):
    obs = df_campo[col_plm2].to_numpy(dtype=float)
    best = {"shift_days": 0, "pearson_r": -np.inf, "sim_intervalo": np.zeros(len(df_campo))}
    for shift in range(-max_shift_days, max_shift_days + 1):
        sim_vals = build_shifted_interval_series(df_sim, df_campo, col_fecha, shift)
        pearson_r = pd.Series(obs).corr(pd.Series(sim_vals))
        if not pd.isna(pearson_r) and pearson_r > best["pearson_r"]:
            best = {"shift_days": shift, "pearson_r": float(pearson_r), "sim_intervalo": sim_vals.copy()}
    return best

def evaluate_cohort_detection(df_sim, df_campo, col_fecha, col_plm2, tol_anticipo=14, tol_retraso=14, min_dist_picos=7, umbral_min_pico=0.30):
    sim_dates, sim_vals = df_sim['Fecha'].values, df_sim['EMERREL'].values
    obs_dates, obs_vals = df_campo[col_fecha].values, df_campo[col_plm2].values
    obs_vals_norm = df_campo['Campo_Normalizado'].values
    max_obs_date = pd.to_datetime(obs_dates.max())

    # Detección simulada
    sim_padded = np.pad(sim_vals, (1, 1), 'constant')
    peaks_sim_padded, _ = find_peaks(sim_padded, height=umbral_min_pico)
    peaks_sim = peaks_sim_padded - 1
    sim_peak_dates = pd.to_datetime(sim_dates[peaks_sim])

    # Detección campo
    min_h_obs = np.max(obs_vals) * 0.05 if np.max(obs_vals) > 0 else 0.01
    peaks_obs = np.where(obs_vals >= min_h_obs)[0]
    obs_peak_dates = pd.to_datetime(obs_dates[peaks_obs])

    # Filtro de Ecos (Cercanía a campo)
    skip_indices = set()
    for i in range(len(sim_peak_dates)):
        if i in skip_indices: continue
        grupo = [i]
        for j in range(i + 1, len(sim_peak_dates)):
            if (sim_peak_dates[j] - sim_peak_dates[grupo[0]]).days <= min_dist_picos: grupo.append(j)
            else: break
        if len(grupo) > 1:
            mejor_idx = min(grupo, key=lambda idx: min([abs((obs_d - sim_peak_dates[idx]).days) for obs_d in obs_peak_dates]) if len(obs_peak_dates)>0 else 0)
            for idx in grupo: 
                if idx != mejor_idx: skip_indices.add(idx)

    # Matching N-a-1
    valid_pairs = []
    for i, s_date in enumerate(sim_peak_dates):
        if i in skip_indices: continue
        for j, o_date in enumerate(obs_peak_dates):
            diff = (o_date - s_date).days
            if -tol_retraso <= diff <= tol_anticipo:
                valid_pairs.append((i, j, diff, abs(diff) + (abs(i-j)*0.001)))
    
    valid_pairs.sort(key=lambda x: x[3])
    matched_sim, matched_obs, tp_points, matched_links, offsets = set(), set(), [], [], []

    for s_idx, o_idx, diff, cost in valid_pairs:
        if o_idx not in matched_obs:
            crossing = any((s_idx > ms and o_idx < mo) or (s_idx < ms and o_idx > mo) for ms, mo in matched_links)
            if not crossing:
                if s_idx not in matched_sim: tp_points.append((sim_peak_dates[s_idx], sim_vals[peaks_sim[s_idx]]))
                matched_sim.add(s_idx); matched_obs.add(o_idx); matched_links.append((s_idx, o_idx)); offsets.append(diff)

    # FP, FN, TN
    fp_points = [(sim_peak_dates[i], sim_vals[peaks_sim[i]]) for i in range(len(sim_peak_dates)) if i not in matched_sim and i not in skip_indices and sim_peak_dates[i] <= max_obs_date]
    fn_points = [(obs_peak_dates[j], obs_vals_norm[peaks_obs[j]]) for j in range(len(obs_peak_dates)) if j not in matched_obs]
    
    tn_points = []
    for j, o_date in enumerate(obs_dates):
        if obs_vals_norm[j] < 0.05:
            s_idx_arr = np.where(sim_dates == o_date)[0]
            if len(s_idx_arr) > 0 and sim_vals[s_idx_arr[0]] < umbral_min_pico: tn_points.append((pd.to_datetime(o_date), sim_vals[s_idx_arr[0]]))

    tp, fp, fn, tn = len(matched_obs), len(fp_points), len(fn_points), len(tn_points)
    prec = tp/(tp+fp) if tp+fp>0 else 0; rec = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0

    return {"f1_score": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "mean_offset": np.mean(offsets) if offsets else 0, "tp_points": tp_points, "fp_points": fp_points, "fn_points": fn_points, "tn_points": tn_points, "zeroed_indices": [peaks_sim[idx] for idx in skip_indices]}

# ---------------------------------------------------------
# 4. MOTOR Y UI BORDENAVE
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()
st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png", use_container_width=True)
archivo_meteo = st.sidebar.file_uploader("1. Clima (Bordenave)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
dga_optimo = st.sidebar.number_input("TT Control Post-emergente (°Cd)", value=600, step=10)
tol_anticipo = st.sidebar.number_input("Anticipo (+)", value=14); tol_retraso = st.sidebar.number_input("Retraso (-)", value=14)
umbral_pico_sim = st.sidebar.number_input("Umbral Mín. Pico Simulado", value=0.30, step=0.05)

df_meteo_raw = load_data(archivo_meteo, "bordenave")
df_campo_raw = load_data(archivo_campo, "bordenave_campo")

if df_meteo_raw is not None and modelo_ann is not None:
    df = df_meteo_raw.copy(); df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha']); df = df.dropna().sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Shift Bordenave
    df["JD_Shifted"] = (df["Julian_days"] + 60).clip(1, 300)
    X = df[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X); df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    # Factor Hídrico y Restricción Bordenave
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["EMERREL"] *= 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df.loc[(df["Julian_days"] <= 15) & (df["Prec_sum_21d"] <= 50), "EMERREL"] = 0.0

    df["DG"] = ((df["TMAX"] + df["TMIN"])/2).apply(lambda x: calculate_tt_scalar(x, 2.0, 20.0, 30.0))

    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy(); col_f, col_p = df_campo.columns[0], df_campo.columns[1]
        df_campo[col_f] = pd.to_datetime(df_campo[col_f])
        df_campo['Campo_Normalizado'] = df_campo[col_p] / df_campo[col_p].max()
        metrics = evaluate_cohort_detection(df, df_campo, col_f, col_p, tol_anticipo, tol_retraso, 7, umbral_pico_sim)
        df.loc[metrics["zeroed_indices"], "EMERREL"] = 0.0
        
        st.title("🌾 PREDWEEM LOLIUM - BORDENAVE 2026")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-Score", f"{metrics['f1_score']:.2f}")
        c2.metric("Aciertos (TP|TN)", f"{metrics['tp']} | {metrics['tn']}")
        c3.metric("Errores (FP|FN)", f"{metrics['fp']} | {metrics['fn']}", delta_color="inverse")
        c4.metric("Sesgo", f"{metrics['mean_offset']:+.1f} d")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Simulado", line=dict(color='#166534', width=2)))
        fig.add_trace(go.Scatter(x=df_campo[col_f], y=df_campo['Campo_Normalizado'], name="Campo", mode='markers+lines', marker=dict(color='#dc2626')))
        if metrics['tp_points']: fig.add_trace(go.Scatter(x=[p[0] for p in metrics['tp_points']], y=[p[1] for p in metrics['tp_points']], mode='markers', name='✅ TP', marker=dict(symbol='star', size=12, color='#10b981')))
        if metrics['tn_points']: fig.add_trace(go.Scatter(x=[p[0] for p in metrics['tn_points']], y=[p[1] for p in metrics['tn_points']], mode='markers', name='✅ TN', marker=dict(symbol='square', size=8, color='#3b82f6')))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Cargue los datos climáticos de Bordenave para iniciar.")
