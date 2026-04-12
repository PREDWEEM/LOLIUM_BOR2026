# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.8 — LOLIUM BORDENAVE 2026
# Actualización:
# - UNIFICACIÓN MECANÍSTICA 100%: 
#   * Reemplazo de flujos diarios por INTEGRACIÓN EN INTERVALOS DE CAMPO.
#   * Agregado de métricas robustas: RMSE de trayectoria y CCC (Concordancia).
# - NUEVO GRÁFICO: "Llenado de la Caja" (Curvas Acumuladas) + Dispersión 1:1.
# - UI: "Datos del Lote" movido a st.expander en la página principal con slider continuo.
# - ADAPTACIÓN BORDENAVE: Coordenadas mantenidas estrictamente en -37.81 según indicación.
# - NUEVO: Bypass Agronómico de Ruptura de Dormición por Choque Hídrico (Reintegrado al motor).
# - NUEVO: Escudo Termofisiológico Dinámico (Media Móvil 10d) para inhibición estival.
# - NUEVO: Corte Hídrico Estricto (20% HR) acoplado a la sigmoide.
# - NUEVO: Bloqueo de emergencia (0%) hasta que una LLUVIA PUNTUAL supere la Capacidad de Campo.
# - Pearson por intervalos de monitoreo y Emparejamiento por Proximidad (Regla Anti-Cruce).
# - Eliminación total de réplicas (Ecos) en cadena y aplanamiento visual.
# - Detección agronómica de flushes de campo (Lógica Gemelos Flanqueantes + Filtro de Indulto para FP).
# - NUEVO: Módulo Mecanístico de Balance Hídrico Superficial (BHS) con Secado Exponencial (Kr).
# - MEJORA: Sensibilidad térmica e hídrica agresiva según nivel de rastrojo.
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import time
import base64
from datetime import timedelta
from pathlib import Path
from scipy.signal import find_peaks

# ---------------------------------------------------------
# 1. PANTALLA DE CARGA ULTRARRÁPIDA Y ESTILO
# ---------------------------------------------------------
if 'arranque_fase' not in st.session_state:
    st.set_page_config(page_title="PREDWEEM BORDENAVE", layout="wide", page_icon="🌾")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("🚜 **Iniciando Servidor PREDWEEM Integral...** Cargando módulos de validación y librerías pesadas.")
    st.progress(20)
    
    st.session_state.arranque_fase = 1
    time.sleep(0.1)
    st.rerun()

if 'arranque_fase' in st.session_state and st.session_state.arranque_fase == 1:
    st.session_state.arranque_fase = 2 

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
    
    div[data-testid="stVerticalBlockBorderWrapper"], 
    div[data-testid="stContainerBorder"],
    div[data-testid="stContainer"] > div > div[style*="border"],
    div[data-testid="stVerticalBlock"] > div[style*="border-radius"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        padding: 15px !important;
        border: 1px solid #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

def set_bg_hack(main_bg_file):
    try:
        with open(main_bg_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""<style>.stApp {{ background-image: url(data:image/png;base64,{encoded_string}); background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }}</style>""",
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

set_bg_hack("fondo_predweem_v3.png") 

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
# 3. LÓGICA TÉCNICA (ANN + BIO + SEÑALES + BHS + INTEGRAL)
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

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-37.81):
    # Latitud ajustada estrictamente a Bordenave
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
    )
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    
    et0 = 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange)
    return np.maximum(et0, 0)

def balance_hidrico_superficial(prec, et0, w_max=20.0, ke_suelo_max=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0 
    
    for i in range(1, n):
        kr = w[i-1] / w_max 
        ke_dinamico = ke_suelo_max * kr
        evaporacion_real = et0[i] * ke_dinamico
        w[i] = w[i-1] + prec[i] - evaporacion_real
        w[i] = max(0.0, min(w_max, w[i]))
        
    return w

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
    
    github_url = f"https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/{default_name}.csv"
    try:
        return pd.read_csv(github_url)
    except:
        return None

# --- NUEVAS FUNCIONES DE INTEGRACIÓN DE INTERVALOS ---
def sincronizar_series_por_intervalos(df_sim, df_campo, col_fecha, col_plm2):
    df_sync = df_campo.copy()
    
    total_campo = df_sync[col_plm2].sum()
    df_sync['Campo_Relativo'] = df_sync[col_plm2] / total_campo if total_campo > 0 else 0
    
    sim_acumulada_intervalos = []
    fecha_anterior = df_sim["Fecha"].min() - pd.Timedelta(days=1)
    
    for _, row in df_sync.iterrows():
        fecha_actual = row[col_fecha]
        mask_ventana = (df_sim["Fecha"] > fecha_anterior) & (df_sim["Fecha"] <= fecha_actual)
        flujo_simulado_ventana = df_sim.loc[mask_ventana, "EMERREL"].sum()
        
        sim_acumulada_intervalos.append(flujo_simulado_ventana)
        fecha_anterior = fecha_actual
        
    df_sync['Simulado_Intervalo'] = sim_acumulada_intervalos
    total_sim = df_sync['Simulado_Intervalo'].sum()
    df_sync['Sim_Relativo'] = df_sync['Simulado_Intervalo'] / total_sim if total_sim > 0 else 0.0
        
    df_sync['Campo_Acumulado'] = df_sync['Campo_Relativo'].cumsum()
    df_sync['Sim_Acumulado'] = df_sync['Sim_Relativo'].cumsum()
    
    return df_sync

def calcular_metricas_validacion_integral(df_sync):
    obs = df_sync['Campo_Relativo'].values
    sim = df_sync['Sim_Relativo'].values
    obs_acum = df_sync['Campo_Acumulado'].values
    sim_acum = df_sync['Sim_Acumulado'].values
    
    pearson_r = np.corrcoef(obs, sim)[0, 1] if np.std(obs) > 0 and np.std(sim) > 0 else 0.0
    rmse_acumulado = np.sqrt(np.mean((obs_acum - sim_acum)**2))
    
    mean_obs, mean_sim = np.mean(obs_acum), np.mean(sim_acum)
    var_obs, var_sim = np.var(obs_acum), np.var(sim_acum)
    covar = np.mean((obs_acum - mean_obs) * (sim_acum - mean_sim))
    
    ccc_acumulado = (2 * covar) / (var_obs + var_sim + (mean_obs - mean_sim)**2) if (var_obs + var_sim) > 0 else 0.0
    
    return {
        "Pearson_Flujos": pearson_r,
        "RMSE_Acumulado": rmse_acumulado,
        "CCC_Acumulado": ccc_acumulado
    }

def evaluate_cohort_detection(df_sim, df_campo, col_fecha, col_plm2, tol_anticipo=14, tol_retraso=14, min_dist_picos=7, umbral_min_pico=0.30):
    sim_dates = df_sim['Fecha'].values
    sim_vals = df_sim['EMERREL'].values
    obs_dates = df_campo[col_fecha].values
    obs_vals = df_campo[col_plm2].values
    obs_vals_norm = df_campo['Campo_Normalizado'].values
    
    sim_vals_peaks = sim_vals.copy()
    max_obs_date = pd.to_datetime(obs_dates.max())
    
    sim_vals_padded = np.pad(sim_vals, (1, 1), 'constant', constant_values=(0, 0))
    peaks_sim_padded, _ = find_peaks(sim_vals_padded, height=umbral_min_pico, distance=1)
    
    peaks_sim = peaks_sim_padded - 1
    peaks_sim = peaks_sim[(peaks_sim >= 0) & (peaks_sim < len(sim_vals))]
    sim_peak_dates = pd.to_datetime(sim_dates[peaks_sim])
    
    min_h_obs = np.max(obs_vals) * 0.05 if np.max(obs_vals) > 0 else 0.01
    peaks_obs = np.where(obs_vals >= min_h_obs)[0]
    obs_peak_dates = pd.to_datetime(obs_dates[peaks_obs])
    
    ventana_contigua = min_dist_picos 
    skip_indices = set()

    i = 0
    while i < len(sim_peak_dates):
        grupo_contiguos = [i]
        j = i + 1
        while j < len(sim_peak_dates) and (sim_peak_dates[j] - sim_peak_dates[j-1]).days <= ventana_contigua:
            grupo_contiguos.append(j)
            j += 1

        if len(grupo_contiguos) > 1:
            mejor_idx = grupo_contiguos[0]
            min_distancia_global = float('inf')

            for idx in grupo_contiguos:
                if len(obs_peak_dates) > 0:
                    distancias = [abs((obs_date - sim_peak_dates[idx]).days) for obs_date in obs_peak_dates]
                    dist_minima_local = min(distancias)
                else:
                    dist_minima_local = float('inf')

                if dist_minima_local < min_distancia_global:
                    min_distancia_global = dist_minima_local
                    mejor_idx = idx
                elif dist_minima_local == min_distancia_global:
                    if sim_vals[peaks_sim[idx]] > sim_vals[peaks_sim[mejor_idx]]:
                        mejor_idx = idx

            for idx in grupo_contiguos:
                if idx != mejor_idx:
                    es_flanqueante = False
                    for obs_date in obs_peak_dates:
                        d_idx = sim_peak_dates[idx]
                        d_mejor = sim_peak_dates[mejor_idx]
                        if (d_idx <= obs_date <= d_mejor) or (d_mejor <= obs_date <= d_idx):
                            es_flanqueante = True
                            break
                    if not es_flanqueante:
                        skip_indices.add(idx)
        i = j

    zeroed_indices = []
    umbral_base = 0.05 

    for idx in skip_indices:
        p_idx = peaks_sim[idx]
        k = p_idx
        while k >= 0 and sim_vals[k] >= umbral_base:
            zeroed_indices.append(k)
            if k > 0 and sim_vals[k-1] > sim_vals[k]:
                break
            k -= 1
        k = p_idx + 1
        while k < len(sim_vals) and sim_vals[k] >= umbral_base:
            zeroed_indices.append(k)
            if k < len(sim_vals) - 1 and sim_vals[k+1] > sim_vals[k]:
                break
            k += 1

    zeroed_indices = list(set(zeroed_indices))
    for z_idx in zeroed_indices:
        sim_vals_peaks[z_idx] = 0.0

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
    
    tp_points, fp_points, fn_points, tn_points = [], [], [], []
    matched_sim, matched_obs, matched_links, offsets = set(), set(), [], []
    
    for sim_idx, obs_idx, diff, cost in valid_pairs:
        if obs_idx not in matched_obs:
            crossing = False
            for m_sim, m_obs in matched_links:
                if (sim_idx > m_sim and obs_idx < m_obs) or (sim_idx < m_sim and obs_idx > m_obs):
                    crossing = True
                    break
            
            if not crossing:
                if sim_idx not in matched_sim:
                    tp_points.append((sim_peak_dates[sim_idx], sim_vals[peaks_sim[sim_idx]]))
                matched_sim.add(sim_idx)
                matched_obs.add(obs_idx)
                matched_links.append((sim_idx, obs_idx))
                offsets.append(diff)
                
    for i in range(len(sim_peak_dates)):
        if i not in matched_sim and i not in skip_indices:
            sim_date_i = sim_peak_dates[i]
            for m_sim, m_obs in matched_links:
                obs_date = obs_peak_dates[m_obs]
                sim_date_m = sim_peak_dates[m_sim]
                if (sim_date_i <= obs_date <= sim_date_m) or (sim_date_m <= obs_date <= sim_date_i):
                    picos_intermedios = 0
                    min_idx, max_idx = min(i, m_sim), max(i, m_sim)
                    for k in range(min_idx + 1, max_idx):
                        if k not in skip_indices:
                            picos_intermedios += 1
                    if picos_intermedios == 0:
                        days_diff = (obs_date - sim_date_i).days
                        if -tol_retraso <= days_diff <= tol_anticipo:
                            tp_points.append((sim_date_i, sim_vals[peaks_sim[i]]))
                            matched_sim.add(i)
                            break

    for i in range(len(sim_peak_dates)):
        if i not in matched_sim and i not in skip_indices:
            if sim_peak_dates[i] <= max_obs_date:
                es_error_real = True
                sim_date_i = sim_peak_dates[i]
                for obs_idx in matched_obs:
                    obs_date = obs_peak_dates[obs_idx]
                    if -tol_retraso <= (obs_date - sim_date_i).days <= tol_anticipo:
                        es_error_real = False
                        break
                if es_error_real:
                    for m_sim in matched_sim:
                        sim_date_m = sim_peak_dates[m_sim]
                        if abs((sim_date_i - sim_date_m).days) <= min_dist_picos:
                            es_error_real = False
                            break
                if es_error_real:
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

    for j, obs_date in enumerate(obs_dates):
        if obs_vals_norm[j] < 0.05:
            sim_idx_arr = np.where(sim_dates == obs_date)[0]
            if len(sim_idx_arr) > 0:
                sim_idx = sim_idx_arr[0]
                if sim_vals[sim_idx] < umbral_min_pico:
                    tn_points.append((pd.to_datetime(obs_date), sim_vals[sim_idx]))
            
    tp = len(matched_obs)
    fp = len(fp_points)
    fn = len(fn_points)
    tn = len(tn_points)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_offset = np.mean(offsets) if offsets else 0.0
    
    return {
        "f1_score": f1, "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_offset": mean_offset, "tp_points": tp_points, "fp_points": fp_points,
        "fn_points": fn_points, "tn_points": tn_points, "zeroed_indices": zeroed_indices
    }

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

# --- HEADER PRINCIPAL ---
st.title("🌾 PREDWEEM LOLIUM - BORDENAVE (BA) lat=-37.803 lon=-63.04")

# --- MENÚ DESPLEGABLE: DATOS DEL LOTE (MAIN PAGE) ---
with st.expander("📂 1. Datos del Lote", expanded=True):
    col_upload, col_rastrojo = st.columns(2)
    
    with col_upload:
        archivo_meteo = st.file_uploader("1. Clima (Bordenave)", type=["xlsx", "csv"])
        archivo_campo = st.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])
        
    with col_rastrojo:
        with st.container(border=True):
            st.markdown("#### 🌾 Manejo de Superficie") 
            
            # Slider continuo
            cobertura_pct = st.slider(
                "Cobertura de Rastrojo en Suelo (%)",
                min_value=0, max_value=100, value=95, step=5,
                help="0% = Suelo desnudo / Labranza convencional. 100% = Cobertura total (Ej. Cultivo de Servicio denso)."
            )

            x_cobertura = [0, 30, 70, 100] 
            y_ke = [0.95, 0.50, 0.25, 0.10]
            ke_val = float(np.interp(cobertura_pct, x_cobertura, y_ke))
            
            y_mod_termico = [1.00, 0.95, 0.90, 0.80]
            mod_termico = float(np.interp(cobertura_pct, x_cobertura, y_mod_termico))
            
            # Tarjeta visual HTML
            html_card = f"""
            <div style="
                background-color: #ffffff;
                padding: 15px 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border: 1px solid #e2e8f0;
                margin-top: 15px;
            ">
                <h5 style="color: #1e293b; margin-top: 0; margin-bottom: 12px; font-size: 0.95rem;">
                    Parámetros Dinámicos Aplicados
                </h5>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #475569; font-size: 0.9rem;">Coeficiente Hídrico Suelo (Ke):</span>
                    <span style="color: #0284c7; font-weight: bold; font-size: 1.05rem;">{ke_val:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #475569; font-size: 0.9rem;">Modulador Térmico Suelo:</span>
                    <span style="color: #b91c1c; font-weight: bold; font-size: 1.05rem;">{mod_termico:.2f}</span>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

df_meteo_raw = load_data(archivo_meteo, "meteo_daily")
df_campo_raw = load_data(archivo_campo, "bordenave_campo")

# --- SIDEBAR ---
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.50)

st.sidebar.markdown("**Ruptura de Dormición Estival (Escudo)**")
umbral_termoinhibicion = st.sidebar.number_input(
    "Umbral Termoinhibición (°C)", 
    min_value=15.0, max_value=35.0, value=24.0, step=0.5,
    help="Si la T° Media móvil supera este valor, se bloquea a 0%."
)

st.sidebar.markdown("**Ruptura de Dormición (Otoño Temprano)**")
umbral_choque_hidrico = st.sidebar.slider(
    "Choque Hídrico 3 días (mm)", 
    min_value=20.0, max_value=100.0, value=30.0
)

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

st.sidebar.markdown("## 🧪 3. Validación de Cohortes")
col_v1, col_v2 = st.sidebar.columns(2)
with col_v1:
    tol_anticipo = st.number_input("Anticipo (+)", value=14, step=1)
with col_v2:
    tol_retraso = st.number_input("Retraso (-)", value=14, step=1)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    min_dist_picos = st.number_input("Separación Flushes (días)", min_value=1, max_value=45, value=7, step=1)
with col_p2:
    umbral_pico_sim = st.number_input("Umbral Mín. Pico Simulado", value=0.20, step=0.05)

st.sidebar.divider()
st.sidebar.markdown("## 💧 4. Balance Hídrico (Suelo)")
w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=20.0, step=1.0)


# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO (MECANÍSTICO)
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:

    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
    
    df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * mod_termico)
    df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * mod_termico)

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

    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    # --- BYPASS AGRONÓMICO: RUPTURA DE DORMICIÓN TEMPRANA ---
    limite_juliano_temprano = 110 # Aprox. 20 de Abril
    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (df["Julian_days"] <= limite_juliano_temprano) & (df["Prec_3d"] >= umbral_choque_hidrico)
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 1.0)

    # Cálculo ET0 con la latitud estricta
    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-37.81)
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo_max=ke_val)
    
    humedad_relativa = df["W_superficial"] / w_max_val
    df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.30)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]

    df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0
    df['Lluvia_Recarga'] = (df['Prec'] >= w_max_val).cummax()
    df.loc[~df['Lluvia_Recarga'], "EMERREL"] = 0.0

    df["Tmedia"] = df["Tmedia_aire"]
    df["Tmedia_10d"] = df["Tmedia"].rolling(window=10, min_periods=1).mean()
    mask_inhibicion = df["Tmedia_10d"] >= umbral_termoinhibicion
    df.loc[mask_inhibicion, "EMERREL"] = 0.0

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

    # Inicializar métricas robustas
    pearson_r, rmse_acum, ccc_acum = 0.0, 0.0, 0.0
    pec, peak_lag, lead_time = 0.0, 0, 0
    desfase_t50 = 0
    cohort_metrics = {"f1_score": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "mean_offset": 0, "tp_points": [], "fp_points": [], "fn_points": [], "tn_points": [], "zeroed_indices": []}

    if df_campo is not None:
        # 1. Procesar la sincronización exacta por intervalos de lectura
        df_sincronizado = sincronizar_series_por_intervalos(
            df_sim=df, 
            df_campo=df_campo, 
            col_fecha=col_fecha, 
            col_plm2=col_plm2
        )
        
        # 2. Calcular las métricas biológicamente ajustadas
        metricas_robustas = calcular_metricas_validacion_integral(df_sincronizado)
        pearson_r = metricas_robustas["Pearson_Flujos"]
        rmse_acum = metricas_robustas["RMSE_Acumulado"]
        ccc_acum = metricas_robustas["CCC_Acumulado"]
        
        df_campo["Sim_Intervalo"] = df_sincronizado["Sim_Relativo"] # Solo para excel export

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
    colorscale_hard = [[0.0, "green"], [0.01, "green"], [0.02, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"], colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Riesgo (Tasa Diaria)")
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DE DECISIÓN", "💧 PRECIPITACIONES Y SUELO", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        if df_campo is not None:
            st.markdown("<p class='metric-header'>🚜 FIDELIDAD DE SIMULACIÓN (INTEGRAL)</p>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Correlación (Pearson)", f"{pearson_r:.3f}", "Sincronía de Flujos")
            c2.metric("Concordancia (CCC)", f"{ccc_acum:.3f}", "Fidelidad de Trayectoria")
            c3.metric("Error (RMSE)", f"{rmse_acum:.3f}", "Magnitud de desvío", delta_color="inverse")
            
            t50_label = "Anticipo (-)" if desfase_t50 < 0 else "Atraso (+)" if desfase_t50 > 0 else "Sincronizado"
            c4.metric("Desfase Global (T50)", f"{desfase_t50:+d} días", t50_label, delta_color="inverse" if desfase_t50 > 0 else "normal" if desfase_t50 < 0 else "off")

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
            fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria Simulada', line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'))
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

        # -----------------------------------------------------
        # NUEVOS GRÁFICOS: CURVAS ACUMULADAS Y DISPERSIÓN 1:1
        # -----------------------------------------------------
        if df_campo is not None and 'df_sincronizado' in locals():
            st.markdown("---")
            st.markdown("<p class='metric-header' style='margin-top:20px;'>📈 VALIDACIÓN DE TRAYECTORIA Y PRECISIÓN</p>", unsafe_allow_html=True)
            
            # Crear dos columnas para mostrar ambos gráficos a la par
            col_curva, col_disp = st.columns([2, 1])
            
            with col_curva:
                fig_acum = go.Figure()
                
                # Curva Real de Campo
                fig_acum.add_trace(go.Scatter(
                    x=df_sincronizado[col_fecha], 
                    y=df_sincronizado['Campo_Acumulado'] * 100, 
                    mode='markers+lines', 
                    name='Real a Campo (%)', 
                    marker=dict(color='#dc2626', size=10, symbol='diamond'),
                    line=dict(color='#dc2626', width=2)
                ))
                
                # Curva Simulada por el Modelo
                fig_acum.add_trace(go.Scatter(
                    x=df_sincronizado[col_fecha], 
                    y=df_sincronizado['Sim_Acumulado'] * 100, 
                    mode='lines', 
                    name='Simulado Modelo (%)', 
                    line=dict(color='#166534', width=3, dash='dash')
                ))

                fig_acum.update_layout(
                    title="Dinámica de Llenado (Curvas Acumuladas)",
                    xaxis_title="Fechas de Monitoreo",
                    yaxis_title="Emergencia Acumulada (%)",
                    height=400,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_acum, use_container_width=True)

            with col_disp:
                fig_1to1 = go.Figure()
                
                # Línea 1:1 (Concordancia Perfecta)
                fig_1to1.add_trace(go.Scatter(
                    x=[0, 100], y=[0, 100], 
                    mode='lines', 
                    name='Modelo Perfecto', 
                    line=dict(color='gray', dash='dash', width=2)
                ))
                
                # Puntos reales vs simulados
                fig_1to1.add_trace(go.Scatter(
                    x=df_sincronizado['Campo_Acumulado'] * 100, 
                    y=df_sincronizado['Sim_Acumulado'] * 100, 
                    mode='markers', 
                    name='Lecturas a Campo', 
                    marker=dict(color='#2563eb', size=10, line=dict(width=1, color='DarkBlue')),
                    text=df_sincronizado[col_fecha].dt.strftime('%d-%m-%Y'),
                    hovertemplate="<b>%{text}</b><br>Campo: %{x:.1f}%<br>Modelo: %{y:.1f}%<extra></extra>"
                ))

                fig_1to1.update_layout(
                    title=f"Ajuste 1:1 (CCC: {ccc_acum:.3f})",
                    xaxis_title="Observado en Campo (%)",
                    yaxis_title="Simulado por PREDWEEM (%)",
                    height=400,
                    xaxis=dict(range=[0, 105]),
                    yaxis=dict(range=[0, 105]),
                    showlegend=False
                )
                st.plotly_chart(fig_1to1, use_container_width=True)

    with tab2:
        st.header("💧 Dinámica Hídrica del Suelo (Balance Superficial)")
        fig_hidrico = go.Figure()
        
        fig_hidrico.add_trace(go.Bar(x=df["Fecha"], y=df["Prec"], name='Lluvia Diaria (mm)', marker_color='#93c5fd', opacity=0.7))
        fig_hidrico.add_trace(go.Scatter(x=df["Fecha"], y=df["W_superficial"], name='Agua en Suelo (0-10cm)', mode='lines', line=dict(color='#0284c7', width=3), fill='tozeroy', fillcolor='rgba(2, 132, 199, 0.2)'))
        fig_hidrico.add_hline(y=w_max_val, line_dash="dot", line_color="#334155", annotation_text=f"Capacidad Máx. ({w_max_val} mm)", annotation_position="top left")

        fig_hidrico.update_layout(title="Precipitación vs. Retención Real de Humedad", xaxis_title="Fecha", yaxis_title="Milímetros (mm)", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_hidrico, use_container_width=True)

    with tab3:
        st.header("🔍 Clasificación DTW (Bordenave)")
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
                    x=JD_COM, y=cluster_model["curves_interp"][pred], name="Patrón Histórico", 
                    line=dict(dash='dash', color={0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}.get(pred))
                ))
                fp.add_trace(go.Scatter(x=jd_grid, y=obs_norm * cluster_model["curves_interp"][pred].max(), name="2026", line=dict(color='black', width=3)))
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
                    'Pearson (Flujos)', 'RMSE (Acumulado)', 'CCC (Acumulado)', 'Desfase T50 Global (días)',
                    'F1-Score Cohortes', 'Picos Coincidentes (TP)', 'Reposos Coincidentes (TN)',
                    'Falsos Positivos (FP)', 'Falsos Negativos (FN)', 'Sesgo Medio Picos (días)'
                ],
                'Valor': [
                    pec, peak_lag, lead_time, 
                    pearson_r, rmse_acum, ccc_acum, desfase_t50,
                    cohort_metrics['f1_score'], cohort_metrics['tp'], cohort_metrics['tn'],
                    cohort_metrics['fp'], cohort_metrics['fn'], cohort_metrics['mean_offset']
                ]
            }
            pd.DataFrame(resumen_val).to_excel(writer, sheet_name='Validacion_Campo', index=False)
            pd.DataFrame({'Configuracion': ['T_Base', 'T_Optima', 'T_Critica', 'W_Max', 'Ke', 'Mod_Termico', 'Umbral_Termoinhibicion'], 'Valor': [t_base_val, t_opt_max, t_critica, w_max_val, ke_val, mod_termico, umbral_termoinhibicion]}).to_excel(writer, sheet_name='Bio_Params', index=False)

    st.sidebar.download_button("📥 Descargar Reporte Completo", output.getvalue(), "PREDWEEM_Integral_Bordenave_vK4_9_8.xlsx")

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue datos climáticos para comenzar.")
