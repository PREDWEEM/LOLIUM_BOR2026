# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.5 — LOLIUM BORDENAVE 2026
# Actualización:
# - Pearson por intervalos de monitoreo
# - Emparejamiento por Proximidad con Regla Anti-Cruce
# - CORRECCIÓN DEFINITIVA: Eliminación total de réplicas (Ecos) del análisis.
# - SELECCIÓN DE PICO: En flushes < 7 días, se prioriza el más cercano al dato de campo.
# - NUEVO MATCH N-A-1: Observaciones de la "rampa de subida" pueden emparejarse al mismo pico simulado.
# - TN asimétrico: Match de Campo < 0.05 con Simulación < 0.30
# - Detección agronómica de flushes de campo (Bypass SciPy)
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
# 3. LÓGICA TÉCNICA (ANN + BIO + SEÑALES)
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
        # Arquitectura específica de Bordenave mantenida
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
    best = {"shift_days": 0, "pearson_r": -np.inf, "sim_intervalo": np.zeros(len(df_campo))}

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
            best = {"shift_days": shift, "pearson_r": float(pearson_r), "sim_intervalo": sim_vals.copy()}

    if best["pearson_r"] == -np.inf:
        best["pearson_r"] = 0.0

    return best

def evaluate_cohort_detection(df_sim, df_campo, col_fecha, col_plm2, tol_anticipo=14, tol_retraso=14, min_dist_picos=7, umbral_min_pico=0.30):
    sim_dates = df_sim['Fecha'].values
    sim_vals = df_sim['EMERREL'].values
    obs_dates = df_campo[col_fecha].values
    obs_vals = df_campo[col_plm2].values
    obs_vals_norm = df_campo['Campo_Normalizado'].values
    
    sim_vals_peaks = sim_vals.copy()
    max_obs_date = pd.to_datetime(obs_dates.max())
    
    # --- PADDING Y DETECCIÓN SIMULADA ---
    sim_vals_padded = np.pad(sim_vals, (1, 1), 'constant', constant_values=(0, 0))
    peaks_sim_padded, _ = find_peaks(sim_vals_padded, height=umbral_min_pico, distance=1)
    
    peaks_sim = peaks_sim_padded - 1
    peaks_sim = peaks_sim[(peaks_sim >= 0) & (peaks_sim < len(sim_vals))]
    sim_peak_dates = pd.to_datetime(sim_dates[peaks_sim])
    
    # --- DETECCIÓN AGRONÓMICA OBSERVADA (BYPASS SCIPY) ---
    min_h_obs = np.max(obs_vals) * 0.05 if np.max(obs_vals) > 0 else 0.01
    peaks_obs = np.where(obs_vals >= min_h_obs)[0]
    obs_peak_dates = pd.to_datetime(obs_dates[peaks_obs])
    
    # --- FILTRO DE PICOS SIMULADOS CONTIGUOS (ELIMINACIÓN DE ECOS) ---
    ventana_contigua = min_dist_picos 
    skip_indices = set()

    for i in range(len(sim_peak_dates)):
        if i in skip_indices:
            continue

        grupo_contiguos = [i]
        for j in range(i + 1, len(sim_peak_dates)):
            if (sim_peak_dates[j] - sim_peak_dates[grupo_contiguos[0]]).days <= ventana_contigua:
                grupo_contiguos.append(j)
            else:
                break

        if len(grupo_contiguos) > 1:
            mejor_idx = grupo_contiguos[0]
            min_distancia_global = float('inf')

            for idx in grupo_contiguos:
                if len(obs_peak_dates) > 0:
                    distancias = [abs((obs_date - sim_peak_dates[idx]).days) for obs_date in obs_peak_dates]
                    dist_minima_local = min(distancias)
                else:
                    dist_minima_local = 0

                if dist_minima_local < min_distancia_global:
                    min_distancia_global = dist_minima_local
                    mejor_idx = idx

            for idx in grupo_contiguos:
                if idx != mejor_idx:
                    skip_indices.add(idx)

    zeroed_indices = []
    for idx in skip_indices:
        sim_vals_peaks[peaks_sim[idx]] = 0.0
        zeroed_indices.append(peaks_sim[idx])

    # --- BEST-MATCH-FIRST POR PROXIMIDAD PURA + ANTI-CRUCE CRONOLÓGICO ---
    valid_pairs = []
    for i, sim_date in enumerate(sim_peak_dates):
        if i in skip_indices:
            continue
            
        for j, obs_date in enumerate(obs_peak_dates):
            days_diff = (obs_date - sim_date).days
            if -tol_retraso <= days_diff <= tol_anticipo:
                cost = abs(days_diff) + (abs(i - j) * 0.001)
                valid_pairs.append((i, j, days_diff, cost))
                
    valid_pairs.sort(key=lambda x: x[3])
    
    tp_points = []
    fp_points = []
    fn_points = []
    tn_points = []
    matched_sim = set()
    matched_obs = set()
    matched_links = []
    offsets = []
    
    for sim_idx, obs_idx, diff, cost in valid_pairs:
        # MATCH N-A-1: Permite que múltiples observaciones se anclen a un solo pico simulado
        if obs_idx not in matched_obs:
            crossing = False
            for m_sim, m_obs in matched_links:
                if (sim_idx > m_sim and obs_idx < m_obs) or (sim_idx < m_sim and obs_idx > m_obs):
                    crossing = True
                    break
            
            if not crossing:
                # Solo dibujamos el TP una vez por pico simulado
                if sim_idx not in matched_sim:
                    tp_points.append((sim_peak_dates[sim_idx], sim_vals[peaks_sim[sim_idx]]))
                
                matched_sim.add(sim_idx)
                matched_obs.add(obs_idx)
                matched_links.append((sim_idx, obs_idx))
                offsets.append(diff)
            
    # Falsos Positivos omitiendo réplicas
    for i in range(len(sim_peak_dates)):
        if i not in matched_sim and i not in skip_indices:
            if sim_peak_dates[i] <= max_obs_date:
                fp_points.append((sim_peak_dates[i], sim_vals_peaks[peaks_sim[i]]))
            
    for j in range(len(obs_peak_dates)):
        if j not in matched_obs:
            obs_idx = peaks_obs[j]
            es_tn_encubierto = False
            
            if obs_vals_norm[obs_idx] < 0.05:
                sim_idx_arr = np.where(sim_dates == obs_dates[obs_idx])[0]
                if len(sim_idx_arr) > 0 and sim_vals[sim_idx_arr[0]] < umbral_min_pico:
                    es_tn_encubierto = True
            
            if not es_tn_encubierto:
                fn_points.append((obs_peak_dates[j], obs_vals_norm[peaks_obs[j]]))

    # --- CÁLCULO DE TRUE NEGATIVES ---
    for j, obs_date in enumerate(obs_dates):
        if obs_vals_norm[j] < 0.05:
            sim_idx_arr = np.where(sim_dates == obs_date)[0]
            if len(sim_idx_arr) > 0:
                sim_idx = sim_idx_arr[0]
                if sim_vals[sim_idx] < umbral_min_pico:
                    tn_points.append((pd.to_datetime(obs_date), sim_vals[sim_idx]))
            
    # TP se define por las observaciones reales absorbidas (N-A-1)
    tp = len(matched_obs)
    fp = len(fp_points)
    fn = len(fn_points)
    tn = len(tn_points)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_offset = np.mean(offsets) if offsets else 0.0
    
    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_offset": mean_offset,
        "tp_points": tp_points,
        "fp_points": fp_points,
        "fn_points": fn_points,
        "tn_points": tn_points,
        "zeroed_indices": zeroed_indices
    }

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png", use_container_width=True)
st.sidebar.markdown("## 📂 1. Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima (bordenave)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

df_meteo_raw = load_data(archivo_meteo, "bordenave")
df_campo_raw = load_data(archivo_campo, "bordenave_campo")

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 20)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2:
    t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)

t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("TT Control Post-emergente (°Cd)", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800, step=10)

st.sidebar.markdown("## 🧪 3. Validación")
max_desfase_validacion = st.sidebar.slider(
    "Desfase máximo admisible Pearson (días)",
    min_value=0, max_value=15, value=10,
    help="Desfase general de la curva para el cálculo de Pearson."
)

st.sidebar.markdown("**Tolerancia Cohortes (Días)**")
col_v1, col_v2 = st.sidebar.columns(2)
with col_v1:
    tol_anticipo = st.number_input("Anticipo (+)", value=14, step=1)
with col_v2:
    tol_retraso = st.number_input("Retraso (-)", value=14, step=1)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    min_dist_picos = st.number_input("Separación Flushes (días)", value=7, disabled=True)
with col_p2:
    umbral_pico_sim = st.number_input("Umbral Mín. Pico Simulado", value=0.30, step=0.05)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO (BORDENAVE vK4.9.5)
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:

    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    df_campo = None
    col_fecha = None
    col_plm2 = None
    
    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)

        max_plm2 = df_campo[col_plm2].max()
        df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2 if max_plm2 > 0 else 0

    # SHIFT ESPECÍFICO DE BORDENAVE
    df["JD_Shifted"] = (df["Julian_days"] + 60).clip(1, 300)
    X = df[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    
    # RESTRICCIÓN HÍDRICA ESPECÍFICA DE BORDENAVE
    df.loc[(df["Julian_days"] <= 15) & (df["Prec_sum_21d"] <= 50), "EMERREL"] = 0.0

    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values:
        fecha_hoy = df['Fecha'].max()

    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    dga_hoy, dga_7dias = 0.0, 0.0
    fecha_inicio_ventana, fecha_control = None, None
    msg_estado = "Esperando pico de emergencia..."

    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()

        df_control = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_control.empty:
            fecha_control = df_control.iloc[0]["Fecha"]

        dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        dga_7dias = dga_hoy + df.iloc[idx_hoy + 1: idx_hoy + 8]["DG"].sum() if idx_hoy + 8 <= len(df) else dga_hoy
        msg_estado = f"Pico detectado el {fecha_inicio_ventana.strftime('%d/%m')}"

    pearson_r, best_shift_days = 0.0, 0
    pec, peak_lag, lead_time = 0.0, 0, 0
    desfase_t50 = 0
    cohort_metrics = {"f1_score": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "mean_offset": 0, "tp_points": [], "fp_points": [], "fn_points": [], "tn_points": [], "zeroed_indices": []}

    if df_campo is not None:
        best_val = evaluate_shifted_validation(df, df_campo, col_fecha, col_plm2, max_desfase_validacion)
        best_shift_days = best_val["shift_days"]
        pearson_r = best_val["pearson_r"]
        df_campo["Sim_Intervalo"] = best_val["sim_intervalo"]
        
        cohort_metrics = evaluate_cohort_detection(df, df_campo, col_fecha, col_plm2, tol_anticipo, tol_retraso, min_dist_picos, umbral_pico_sim)
        
        if cohort_metrics.get("zeroed_indices"):
            df.loc[cohort_metrics["zeroed_indices"], "EMERREL"] = 0.0

        tot_plm2 = df_campo[col_plm2].sum()
        if tot_plm2 > 0:
            df_campo['cum_plm2_norm'] = df_campo[col_plm2].cumsum() / tot_plm2
            t50_obs_date = df_campo[df_campo['cum_plm2_norm'] >= 0.5].iloc[0][col_fecha]
            
            max_obs_date = df_campo[col_fecha].max()
            df_sim_trunc = df[df['Fecha'] <= max_obs_date].copy()
            tot_emer = df_sim_trunc['EMERREL'].sum()
            
            if tot_emer > 0:
                df_sim_trunc['cum_emer_norm'] = df_sim_trunc['EMERREL'].cumsum() / tot_emer
                t50_sim_date = df_sim_trunc[df_sim_trunc['cum_emer_norm'] >= 0.5].iloc[0]['Fecha']
                desfase_t50 = (t50_sim_date - t50_obs_date).days

        if fecha_control:
            malezas_totales_campo = df_campo[col_plm2].sum()
            malezas_controladas_efectivamente = df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum()
            pec = (malezas_controladas_efectivamente / malezas_totales_campo) * 100 if malezas_totales_campo > 0 else 0
            
            idx_pico_campo = df_campo[col_plm2].idxmax()
            peak_lag = (fecha_control - df_campo.loc[idx_pico_campo, col_fecha]).days
            
            df_alertas = df[df['EMERREL'] >= umbral_er]
            fecha_primera_alerta = df_alertas['Fecha'].iloc[0] if not df_alertas.empty else fecha_inicio_ventana
            lead_time = (fecha_control - fecha_primera_alerta).days

    # -----------------------------------------------------
    # VISUALIZACIÓN FRONT-END
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - BORDENAVE 2026")

    colorscale_hard = [[0.0, "green"], [0.14, "green"], [0.15, "yellow"], [0.34, "yellow"], [0.35, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"], colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Riesgo (Tasa Diaria)")
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DE DECISIÓN", "🌧️ PRECIPITACIONES", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        if df_campo is not None:
            st.markdown("<p class='metric-header'>🚜 SINCRONÍA POBLACIONAL (TENDENCIA GLOBAL)</p>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Pearson (r)", f"{pearson_r:.3f}", "Correlación de curva")
            c2.metric("Shift Óptimo", f"{best_shift_days:+d} d", "Corrimiento Max Pearson")
            
            t50_label = "Anticipo (-)" if desfase_t50 < 0 else "Atraso (+)" if desfase_t50 > 0 else "Sincronizado"
            c3.metric("Desfase Global (T50)", f"{desfase_t50:+d} días", t50_label, delta_color="inverse" if desfase_t50 > 0 else "normal" if desfase_t50 < 0 else "off")

            st.markdown("<p class='metric-header' style='margin-top:15px;'>🎯 SINCRONÍA DE COHORTES (PULSOS)</p>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("F1-Score", f"{cohort_metrics['f1_score']:.2f}", f"Ventana (+{tol_anticipo} / -{tol_retraso} d)", delta_color="normal")
            
            k2.metric("Aciertos (TP | TN)", f"{cohort_metrics['tp']} | {cohort_metrics['tn']}", "Picos | Ceros Coincidentes")
            k3.metric("Errores (FP / FN)", f"{cohort_metrics['fp']} / {cohort_metrics['fn']}", "Inventados / Omitidos", delta_color="inverse")
            
            sesgo = cohort_metrics['mean_offset']
            sesgo_label = "Anticipo Medio" if sesgo < 0 else "Atraso Medio" if sesgo > 0 else "Sincronizado"
            k4.metric("Sesgo Medio (Picos)", f"{sesgo:+.1f} d", sesgo_label, delta_color="inverse" if sesgo > 0 else "normal" if sesgo < 0 else "off")
            
            if fecha_control:
                st.markdown("<p class='metric-header' style='margin-top:15px;'>⚙️ LOGÍSTICA DE CONTROL</p>", unsafe_allow_html=True)
                l1, l2, l3 = st.columns(3)
                l1.metric("Control Efectivo (PEC)", f"{pec:.1f}%", "A la fecha de aplicación")
                l2.metric("Lag (Desfase)", f"{peak_lag} días", "Vs Pico de Campo")
                l3.metric("Lead Time", f"{lead_time} días", "Anticipación Logística")
            st.markdown("---")

        col_main, col_gauge = st.columns([2, 1])

        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria (Simulada)', line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'))
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral Alerta ({umbral_er})")

            if df_campo is not None:
                fig_emer.add_trace(go.Scatter(x=df_campo[col_fecha], y=df_campo['Campo_Normalizado'], mode='markers+lines', name='Recuentos a Campo', marker=dict(color='#dc2626', size=8, symbol='circle'), line=dict(color='rgba(220, 38, 38, 0.4)', dash='dot')))
                
                if cohort_metrics['tp_points']:
                    tp_x = [p[0] for p in cohort_metrics['tp_points']]
                    tp_y = [p[1] for p in cohort_metrics['tp_points']]
                    fig_emer.add_trace(go.Scatter(x=tp_x, y=tp_y, mode='markers', name='✅ TP (Detectado)', marker=dict(color='#10b981', size=14, symbol='star', line=dict(width=1, color='DarkSlateGrey'))))
                
                if cohort_metrics['tn_points']:
                    tn_x = [p[0] for p in cohort_metrics['tn_points']]
                    tn_y = [p[1] for p in cohort_metrics['tn_points']]
                    fig_emer.add_trace(go.Scatter(x=tn_x, y=tn_y, mode='markers', name='✅ TN (Reposo Coincidente)', marker=dict(color='#3b82f6', size=12, symbol='square', line=dict(width=1, color='DarkBlue'))))

                if cohort_metrics['fp_points']:
                    fp_x = [p[0] for p in cohort_metrics['fp_points']]
                    fp_y = [p[1] for p in cohort_metrics['fp_points']]
                    fig_emer.add_trace(go.Scatter(x=fp_x, y=fp_y, mode='markers', name='❌ FP (Inventado)', marker=dict(color='#ef4444', size=12, symbol='x', line=dict(width=2, color='DarkRed'))))
                
                if cohort_metrics['fn_points']:
                    fn_x = [p[0] for p in cohort_metrics['fn_points']]
                    fn_y = [p[1] for p in cohort_metrics['fn_points']]
                    fig_emer.add_trace(go.Scatter(x=fn_x, y=fn_y, mode='markers', name='⚠️ FN (Omitido)', marker=dict(color='#f97316', size=12, symbol='triangle-up', line=dict(width=1, color='Black'))))

            if fecha_control:
                fig_emer.add_vline(x=fecha_control.timestamp() * 1000, line_dash="dot", line_color="red", line_width=3, annotation_text=f"Control ({dga_optimo}°Cd)", annotation_position="top left")
                fin_res = fecha_control + timedelta(days=residualidad)
                fig_emer.add_vrect(x0=fecha_control.timestamp() * 1000, x1=fin_res.timestamp() * 1000, fillcolor="blue", opacity=0.1, layer="below", line_width=0, annotation_text=f"Protección ({residualidad}d)")

            fig_emer.update_layout(title="Dinámica de Emergencia y Momento Crítico", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico detectado)")
                if fecha_control:
                    st.error(f"🎯 **MOMENTO CRÍTICO DE CONTROL:** {fecha_control.strftime('%d-%m-%Y')}. Acumulados **{dga_optimo} °Cd**.")
                else:
                    st.info(f"⏳ **En Progreso:** Aún no se han acumulado los {dga_optimo} °Cd requeridos.")
            else:
                st.warning(f"⏳ Esperando primera alerta (Tasa diaria >= {umbral_er}).")

        with col_gauge:
            max_axis = dga_critico * 1.2
            fig_gauge = go.Figure()
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number", value=dga_hoy, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>TT ACUMULADO (°Cd)</b>", 'font': {'size': 18}},
                gauge={'axis': {'range': [None, max_axis]}, 'bar': {'color': "#1e293b", 'thickness': 0.3}, 'steps': [{'range': [0, dga_optimo], 'color': "#4ade80"}, {'range': [dga_optimo, dga_critico], 'color': "#facc15"}, {'range': [dga_critico, max_axis], 'color': "#f87171"}], 'threshold': {'line': {'color': "#2563eb", 'width': 6}, 'thickness': 0.8, 'value': dga_7dias}}
            ))
            fig_gauge.add_annotation(x=0.5, y=-0.1, text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>", showarrow=False, font=dict(size=14, color="#1e3a8a"))
            fig_gauge.update_layout(height=350, margin=dict(t=80, b=50, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.header("🌧️ Dinámica de Precipitaciones Diarias")
        fig_prec = go.Figure(data=[go.Bar(x=df["Fecha"], y=df["Prec"], marker_color='#60a5fa', opacity=0.8)])
        fig_prec.update_layout(title="Precipitación Diaria Registrada", xaxis_title="Fecha", yaxis_title="Milímetros (mm)", height=400)
        st.plotly_chart(fig_prec, use_container_width=True)

    with tab3:
        st.header("🔍 Clasificación DTW")
        fecha_corte = pd.Timestamp("2026-05-01")
        df_obs = df[df["Fecha"] < fecha_corte].copy()
        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)
            dists = [dtw_distance(obs_norm, m[JD_COM <= jd_corte] / m[JD_COM <= jd_corte].max() if m[JD_COM <= jd_corte].max() > 0 else m[JD_COM <= jd_corte]) for m in cluster_model["curves_interp"]]
            pred = int(np.argmin(dists))
            
            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(
                    x=JD_COM, 
                    y=cluster_model["curves_interp"][pred], 
                    name="Patrón", 
                    line=dict(
                        dash='dash', 
                        color={0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}.get(pred)
                    )
                ))
                
                fp.add_trace(go.Scatter(
                    x=jd_grid, 
                    y=obs_norm * cluster_model["curves_interp"][pred].max(), 
                    name="2026", 
                    line=dict(color='black', width=3)
                ))
                st.plotly_chart(fp, use_container_width=True)
                
            with c2:
                nombres_clusters = {0: '🌾 Bimodal', 1: '🌱 Temprano', 2: '🍂 Tardío'}
                st.success(f"### {nombres_clusters.get(pred, 'Desconocido')}")
                st.metric("DTW Score", f"{min(dists):.2f}")
        else:
            st.info("Datos insuficientes para clasificación DTW.")

    with tab4:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]
        fig_bio = go.Figure(data=[go.Scatter(x=x_temps, y=y_tt, mode='lines', line=dict(color='#2563eb', width=4), fill='tozeroy')])
        st.plotly_chart(fig_bio, use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        if df_campo is not None:
            df_campo.to_excel(writer, index=False, sheet_name='Campo_Validacion')
            resumen_val = {
                'Métrica': [
                    'PEC (%)', 'Lag Control (días)', 'Lead Time Control (días)', 
                    'Pearson (r)', 'Shift Óptimo Max Pearson (días)', 'Desfase T50 Global (días)',
                    'F1-Score Cohortes', 'Picos Coincidentes (TP)', 'Reposos Coincidentes (TN)',
                    'Falsos Positivos (FP)', 'Falsos Negativos (FN)', 'Sesgo Medio Picos (días)'
                ],
                'Valor': [
                    pec, peak_lag, lead_time, 
                    pearson_r, best_shift_days, desfase_t50,
                    cohort_metrics['f1_score'], cohort_metrics['tp'], cohort_metrics['tn'],
                    cohort_metrics['fp'], cohort_metrics['fn'], cohort_metrics['mean_offset']
                ]
            }
            pd.DataFrame(resumen_val).to_excel(writer, sheet_name='Validacion_Campo', index=False)

    st.sidebar.download_button("📥 Descargar Reporte Completo", output.getvalue(), "PREDWEEM_Integral_Bordenave_vK4_9_5.xlsx")

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue datos climáticos para comenzar.")
