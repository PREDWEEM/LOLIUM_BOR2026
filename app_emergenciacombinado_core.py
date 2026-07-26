# -*- coding: utf-8 -*-
# ===============================================================
# 👑 PREDWEEM INTEGRAL vK4.9.15 VISUAL V3 — LOLIUM BORDENAVE 2026
# Actualización y Rigor Científico:
# - ENTRADAS ANN CORREGIDAS: JD, TMAX aire, TMIN aire y precipitación.
# - ADAPTACIÓN BORDENAVE: Coordenadas fijas en -37.761671 para el motor de ET0.
# - IDENTIDAD: PREDWEEM by GUILLERMO R. CHANTRE.
# - LATENCIA INICIAL: Bloqueo estricto de emergencia los primeros 15 días del año.
# - ESCUDO TERMOFISIOLÓGICO: Horizonte de termoinhibición dinámico ajustado a 5 días.
# - UMBRAL HÍDRICO: Choque hídrico de 3 días fijado en 45 mm.
# - PRIMER PICO VÁLIDO: La campaña se habilita únicamente cuando EMERREL > 0.20.
# - VALIDACIÓN DE FRECUENCIA VARIABLE: Incorporación del método de Integración 
#   Dinámica por Intervalo Real (Event-to-Event), eliminando artefactos de interpolación.
# - OPTIMIZADOR 2D BIO-FÍSICO: Barrido de parámetros de suelo (W_Max y Ke) ajustado a ventanas de campo.
# - MICROCLIMA PRESERVADO: Modulador térmico invernal activo (Días julianos 152-264).
# - COINCIDENCIA OPERATIVA: Métricas F1-Score, Exactitud Global y Matriz de Confusión interactiva.
# - SINCRONÍA DE INICIO: Evaluación del desfase temporal del primer flujo (Gatillo de DGA).
# - UX DINÁMICA: Sombreados de fondo del gráfico de dinámica vinculados a las fechas reales de muestreo.
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import time
from datetime import timedelta
from pathlib import Path
import base64

# ---------------------------------------------------------
# 1. PANTALLA DE CARGA ULTRARRÁPIDA
# ---------------------------------------------------------
if 'arranque_fase' not in st.session_state:
    st.set_page_config(page_title="PREDWEEM BORDENAVE INTEGRAL", layout="wide", page_icon="🌾")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("🚜 **Iniciando Servidor PREDWEEM Bordenave...** Cargando motores por evento real.")
    st.progress(20)
    
    st.session_state.arranque_fase = 1
    time.sleep(0.1)
    st.rerun()

if 'arranque_fase' in st.session_state and st.session_state.arranque_fase == 1:
    st.session_state.arranque_fase = 2 

# ---------------------------------------------------------
# 2. CONFIGURACIÓN DE ESTILOS GLOBALES
# ---------------------------------------------------------
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

UMBRAL_PRIMER_PICO = 0.20

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
# 3. VALIDACIÓN DE ACTIVOS CIENTÍFICOS
# ---------------------------------------------------------
REQUIRED_MODEL_FILES = (
    "IW.npy",
    "bias_IW.npy",
    "LW.npy",
    "bias_out.npy",
    "modelo_clusters_k3.pkl",
)

def validate_required_model_files():
    """Detiene la aplicación si falta algún activo científico requerido."""
    missing = [name for name in REQUIRED_MODEL_FILES if not (BASE / name).is_file()]
    if missing:
        st.error(
            "Faltan activos requeridos del modelo PREDWEEM: "
            + ", ".join(missing)
            + ". Verifique el checkout privado y el despliegue de Streamlit."
        )
        st.stop()

validate_required_model_files()

# ---------------------------------------------------------
# 4. LÓGICA TÉCNICA E INTEGRACIÓN DINÁMICA POR EVENTO REAL
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
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-37.761671):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)

def balance_hidrico_superficial(prec, et0, w_max=20.0, ke_suelo=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0 
    for i in range(1, n):
        evapacoracion_real = et0[i] * ke_suelo
        w[i] = max(0.0, min(w_max, w[i-1] + prec[i] - evapacoracion_real))
    return w

def aplicar_filtro_primer_pico(df, umbral=UMBRAL_PRIMER_PICO):
    """
    Habilita la campaña desde el primer valor de EMERREL
    estrictamente superior al umbral. Los pulsos anteriores
    se conservan para auditoría y se cancelan en EMERREL.
    """
    df = df.copy()
    df["EMERREL_ANTES_FILTRO_PRIMER_PICO"] = df["EMERREL"].copy()

    candidatos = df.index[df["EMERREL"] > umbral].tolist()

    if candidatos:
        idx_primer_pico = candidatos[0]
        df["Primer_Pico_Habilitado"] = df.index >= idx_primer_pico
        df.loc[df.index < idx_primer_pico, "EMERREL"] = 0.0
    else:
        idx_primer_pico = None
        df["Primer_Pico_Habilitado"] = False
        df["EMERREL"] = 0.0

    return df, idx_primer_pico

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
    def normalize(self, X): return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        emerrel = (np.tanh((a1 @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"), np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy"))
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f: k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def load_data(file_uploader, default_name):
    """Carga archivos aportados por el usuario o recursos locales del checkout."""
    if file_uploader:
        suffix = Path(file_uploader.name).suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_uploader)
        return pd.read_csv(file_uploader)

    local_candidates = (
        BASE / f"{default_name}.csv",
        BASE / f"{default_name}.xlsx",
        BASE / f"{default_name}.xls",
    )
    for candidate in local_candidates:
        if candidate.is_file():
            if candidate.suffix.lower() in {".xlsx", ".xls"}:
                return pd.read_excel(candidate)
            return pd.read_csv(candidate)

    st.warning(
        f"No se encontró un archivo local para '{default_name}'. "
        "Verifique que el recurso esté incluido en el checkout privado."
    )
    return None

def sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2):
    df_campo = df_campo.sort_values(col_fecha).copy()
    df_campo['Campo_Acum_Abs'] = df_campo[col_plm2].cumsum()
    
    fechas_reales = df_campo[col_fecha].tolist()
    registros = []
    
    for i in range(1, len(fechas_reales)):
        f_inicio = fechas_reales[i-1]
        f_fin = fechas_reales[i]
        dias_intervalo = (f_fin - f_inicio).days
        
        obs_inicio = df_campo.loc[df_campo[col_fecha] == f_inicio, 'Campo_Acum_Abs'].values[0]
        obs_fin = df_campo.loc[df_campo[col_fecha] == f_fin, 'Campo_Acum_Abs'].values[0]
        flujo_obs = max(0.0, obs_fin - obs_inicio)
        
        mask_sim = (df_sim['Fecha'] > f_inicio) & (df_sim['Fecha'] <= f_fin)
        flujo_sim = df_sim.loc[mask_sim, 'EMERREL'].sum()
        
        acum_sim_fin = df_sim.loc[df_sim['Fecha'] <= f_fin, 'EMERREL'].sum()
        
        registros.append({
            'Fecha': f_fin,
            'Dias_Intervalo': dias_intervalo,
            'Flujo_Obs_Abs': flujo_obs,
            'Flujo_Sim_Abs': flujo_sim,
            'Acum_Obs_Abs': obs_fin,
            'Acum_Sim_Abs': acum_sim_fin
        })
        
    df_res = pd.DataFrame(registros)
    if df_res.empty:
        return pd.DataFrame()
        
    total_obs = df_res['Flujo_Obs_Abs'].sum()
    total_sim = df_sim.loc[df_sim['Fecha'] <= fechas_reales[-1], 'EMERREL'].sum()
    
    df_res['Campo_Relativo'] = df_res['Flujo_Obs_Abs'] / total_obs if total_obs > 0 else 0.0
    df_res['Sim_Relativo'] = df_res['Flujo_Sim_Abs'] / total_sim if total_sim > 0 else 0.0
    
    df_res['Campo_Acumulado'] = df_res['Acum_Obs_Abs'] / df_campo['Campo_Acum_Abs'].max() if df_campo['Campo_Acum_Abs'].max() > 0 else 0.0
    df_res['Sim_Acumulado'] = df_res['Acum_Sim_Abs'] / df_sim['EMERREL'].sum() if df_sim['EMERREL'].sum() > 0 else 0.0
    
    return df_res

def calcular_metricas_validacion_integral(df_sync, umbral_deteccion=0.05):
    if df_sync.empty or len(df_sync) < 2:
        return {"Pearson_Flujos": 0.0, "NSE_Flujos": 0.0, "KGE_Flujos": 0.0, 
                "RMSE_Acumulado": 0.0, "CCC_Acumulado": 0.0, "R2_Acumulado": 0.0,
                "Exactitud_Global": 0.0, "F1_Score_Coincidencia": 0.0, 
                "Hits": 0, "Misses": 0, "Falsos_Positivos": 0, "Correctos_Negativos": 0}

    mask_activos = (df_sync['Campo_Relativo'] > 0) | (df_sync['Sim_Relativo'] > 0)
    df_activos = df_sync[mask_activos].copy()
    
    if len(df_activos) < 2:
        pearson_r, nse_flujos, kge_flujos = 0.0, 0.0, 0.0
    else:
        obs = df_activos['Campo_Relativo'].values
        sim = df_activos['Sim_Relativo'].values
        
        std_obs, std_sim = np.std(obs), np.std(sim)
        pearson_r = np.corrcoef(obs, sim)[0, 1] if std_obs > 0 and std_sim > 0 else 0.0
        
        var_obs_sum = np.sum((obs - np.mean(obs))**2)
        nse_flujos = 1 - (np.sum((sim - obs)**2) / var_obs_sum) if var_obs_sum > 0 else 0.0
        
        if np.mean(obs) > 0 and std_obs > 0:
            r = pearson_r
            alpha = std_sim / std_obs
            beta = np.mean(sim) / np.mean(obs)
            kge_flujos = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge_flujos = 0.0

    obs_acum, sim_acum = df_sync['Campo_Acumulado'].values, df_sync['Sim_Acumulado'].values
    rmse_acumulado = np.sqrt(np.mean((obs_acum - sim_acum)**2))
    
    mean_obs_ac, mean_sim_ac = np.mean(obs_acum), np.mean(sim_acum)
    var_obs_ac, var_sim_ac = np.var(obs_acum), np.var(sim_acum)
    covar_ac = np.mean((obs_acum - mean_obs_ac) * (sim_acum - mean_sim_ac))
    
    denominador_ccc = var_obs_ac + var_sim_ac + (mean_obs_ac - mean_sim_ac)**2
    ccc_acumulado = (2 * covar_ac) / denominador_ccc if denominador_ccc > 0 else 0.0
    
    ss_res_ac = np.sum((obs_acum - sim_acum)**2)
    ss_tot_ac = np.sum((obs_acum - mean_obs_ac)**2)
    r2_acumulado = 1 - (ss_res_ac / ss_tot_ac) if ss_tot_ac > 0 else 0.0
    
    # --- MÉTRICAS DE COINCIDENCIA POR INTERVALO (AGREEMENT) ---
    obs_eventos = df_sync['Campo_Relativo'] > umbral_deteccion
    sim_eventos = df_sync['Sim_Relativo'] > umbral_deteccion

    hits = np.sum(obs_eventos & sim_eventos)                  
    misses = np.sum(obs_eventos & ~sim_eventos)              
    false_alarms = np.sum(~obs_eventos & sim_eventos)        
    correct_negatives = np.sum(~obs_eventos & ~sim_eventos)  

    total_intervalos = len(df_sync)

    exactitud = (hits + correct_negatives) / total_intervalos if total_intervalos > 0 else 0.0
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    recall = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Pearson_Flujos": pearson_r, 
        "NSE_Flujos": nse_flujos,
        "KGE_Flujos": kge_flujos,
        "RMSE_Acumulado": rmse_acumulado, 
        "CCC_Acumulado": ccc_acumulado,
        "R2_Acumulado": r2_acumulado,
        "Exactitud_Global": exactitud,
        "F1_Score_Coincidencia": f1_score,
        "Hits": int(hits),
        "Misses": int(misses),
        "Falsos_Positivos": int(false_alarms),
        "Correctos_Negativos": int(correct_negatives)
    }

# ---------------------------------------------------------
# 4.5 MÓDULO OPTIMIZADOR 2D (CALIBRACIÓN SUELO POR EVENTO)
# ---------------------------------------------------------
def optimizar_parametros_hidricos_2d(
    df_meteo,
    df_campo,
    modelo_ann,
    latitud_bordenave=-37.761671,
    calentamiento_suelo=0.0,
    umbral_choque_hidrico=45.0
):
    df = df_meteo.copy()
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
    df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * 0.90)
    df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * 0.90)
    
    mask_oi = (df["Julian_days"] >= 152) & (df["Julian_days"] <= 264)
    df.loc[mask_oi, "TMAX_suelo"] += calentamiento_suelo
    df.loc[mask_oi, "TMIN_suelo"] += calentamiento_suelo

    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=latitud_bordenave)
    
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    
    rango_w_max = np.arange(10.0, 36.0, 2.0)
    rango_ke = np.arange(0.2, 0.85, 0.1)
    
    resultados = []
    col_fecha = df_campo.columns[0]
    col_plm2 = df_campo.columns[1]
            
    for w_max in rango_w_max:
        for ke in rango_ke:
            df_sim = df.copy()
            df_sim["EMERREL_RAW"] = np.maximum(emerrel_raw, 0.0)
            df_sim.loc[df_sim["Julian_days"] <= 15, "EMERREL_RAW"] = 0.0
            
            df_sim["W_superficial"] = balance_hidrico_superficial(df_sim["Prec"].values, df_sim["ET0"].values, w_max=w_max, ke_suelo=ke)
            humedad_relativa = df_sim["W_superficial"] / w_max
            df_sim["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))
            
            # Choque hídrico de 3 días, coherente con el motor principal
            df_sim["Prec_3d"] = df_sim["Prec"].rolling(
                window=3,
                min_periods=1
            ).sum()
            mask_ruptura_opt = (
                (df_sim["Julian_days"] > 15)
                & (df_sim["Julian_days"] <= 110)
                & (df_sim["Prec_3d"] >= umbral_choque_hidrico)
            )
            df_sim.loc[mask_ruptura_opt, "EMERREL_RAW"] = np.maximum(
                df_sim.loc[mask_ruptura_opt, "EMERREL_RAW"],
                1.0
            )

            df_sim["EMERREL"] = df_sim["EMERREL_RAW"] * df_sim["Hydric_Factor"]
            df_sim.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0
            df_sim['Lluvia_Recarga'] = (df_sim['Prec'] >= w_max).cummax()
            df_sim.loc[~df_sim['Lluvia_Recarga'], "EMERREL"] = 0.0
            
            df_sim["Tmedia_5d"] = df_sim["Tmedia_aire"].rolling(window=5, min_periods=1).mean()
            df_sim.loc[df_sim["Tmedia_5d"] >= 24.0, "EMERREL"] = 0.0
            
            # Misma validación de inicio usada por el motor principal
            df_sim, _ = aplicar_filtro_primer_pico(
                df_sim,
                umbral=UMBRAL_PRIMER_PICO
            )
            
            df_sync = sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2)
            metricas = calcular_metricas_validacion_integral(df_sync)
            
            resultados.append({
                "W_Max (mm)": w_max,
                "Ke_Suelo": round(ke, 2),
                "F1-Score": metricas["F1_Score_Coincidencia"],
                "NSE (Flujos)": metricas["NSE_Flujos"],
                "KGE": metricas["KGE_Flujos"],
                "CCC (Acumulado)": metricas["CCC_Acumulado"],
                "R2": metricas["R2_Acumulado"],
                "RMSE": metricas["RMSE_Acumulado"]
            })
            
    df_resultados = pd.DataFrame(resultados)
    return df_resultados.sort_values(by=["F1-Score", "NSE (Flujos)"], ascending=[False, False]).reset_index(drop=True)

# ---------------------------------------------------------
# 5. INTERFAZ PRINCIPAL Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.title("🌾 PREDWEEM LOLIUM — BORDENAVE (BA) lat=-37.761671 lon=-63.083717")
st.caption("Bordenave · VISUAL V3 · ventana eficiente verde · exportación PNG de alta resolución")

with st.expander("📂 1. Datos del Lote", expanded=True):
    col_upload, col_rastrojo = st.columns(2)
    
    with col_upload:
        archivo_meteo = st.file_uploader("1. Clima (Bordenave)", type=["xlsx", "csv"])
        archivo_campo = st.file_uploader("2. Campo (Validación Real Variable)", type=["xlsx", "csv"])
        
    with col_rastrojo:
        with st.container(border=True):
            st.markdown("#### 🌾 Manejo de Superficie")
            cobertura_pct = st.slider(
                "Cobertura de Rastrojo en Suelo (%)",
                min_value=0, max_value=100, value=75, step=5,
                help="0% = Suelo desnudo / Labranza. 100% = Cobertura total (Ej. Cultivo de Servicio)."
            )

            x_cobertura = [0, 30, 70, 100]
            ke_val = float(np.interp(cobertura_pct, x_cobertura, [0.85, 0.50, 0.25, 0.10]))
            mod_termico = float(np.interp(cobertura_pct, x_cobertura, [0.95, 0.90, 0.85, 0.80]))

            html_card = f"""
            <div style="background-color: #ffffff; padding: 15px 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; margin-top: 15px;">
                <h5 style="color: #1e293b; margin-top: 0; margin-bottom: 12px; font-size: 0.95rem;">Parámetros Dinámicos Aplicados</h5>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #475569; font-size: 0.9rem;">Coeficiente Hídrico Suelo (Ke):</span>
                    <span style="color: #0284c7; font-weight: bold; font-size: 1.05rem;">{ke_val:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #475569; font-size: 0.9rem;">Modulador térmico diagnóstico:</span>
                    <span style="color: #b91c1c; font-weight: bold; font-size: 1.05rem;">{mod_termico:.2f}</span>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

df_meteo_raw = load_data(archivo_meteo, "meteo_daily")
df_campo_raw = load_data(archivo_campo, "bordenave_campo")

# --- SIDEBAR ---
logo_path = BASE / "logo.png"
if logo_path.is_file():
    st.sidebar.image(str(logo_path), width="stretch")
else:
    st.sidebar.markdown("## 🌾 PREDWEEM")
    st.sidebar.caption("Motor predictivo Lolium — Bordenave")

st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.001, 0.80, 0.001)

st.sidebar.markdown("**Ruptura de Dormición Estival (Escudo)**")
umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0, 0.5)

st.sidebar.markdown("**Ruptura de Dormición (Otoño Temprano)**")
umbral_choque_hidrico = st.sidebar.slider(
    "Choque Hídrico 3 días (mm)",
    min_value=10.0,
    max_value=100.0,
    value=45.0,
    step=1.0
)

st.sidebar.markdown("**Validación del Primer Pico**")
st.sidebar.info(
    f"El inicio de la campaña se habilita únicamente cuando "
    f"EMERREL > {UMBRAL_PRIMER_PICO:.2f}."
)

residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 0)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("TT Control Post-emergente (°Cd)", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800, step=10)

st.sidebar.divider()
st.sidebar.markdown("## 🌡️ 3. Microclima de Suelo (Invierno)")
calentamiento_suelo = st.sidebar.slider(
    "Aumento de Temperatura (°C)", 
    min_value=0.0, max_value=8.0, value=0.0, step=0.5,
    help="Suma N grados a la temperatura estimada del suelo entre JD 152 y 264 solo con fines diagnósticos; no modifica la ANN."
)

st.sidebar.divider()
st.sidebar.markdown("## 💧 4. Balance Hídrico (Suelo)")
w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=20.0, step=1.0)

st.sidebar.divider()
st.sidebar.markdown("## 📊 5. Estado de Validación")
st.sidebar.info("🔬 **Modo Event-to-Event Habilitado**: Los desvíos se calculan de manera dinámica adaptándose al intervalo real de muestreo a campo (7 a 21 días), protegiendo la varianza pura.")

# --- MODO DESARROLLADOR: CALIBRADOR 2D ---
with st.sidebar.expander("🛠️ Modo Dev: Calibrador Bio-Físico 2D", expanded=False):
    st.caption("Mapea combinaciones óptimas de suelo directamente contra las ventanas reales de tus datos de campo.")
    if st.button("Ejecutar Barrido Hídrico"):
        if df_meteo_raw is not None and df_campo_raw is not None and modelo_ann is not None:
            with st.spinner('Ejecutando iteraciones hídricas puras por evento...'):
                df_meteo_opt = df_meteo_raw.copy()
                df_meteo_opt.columns = [c.upper().strip() for c in df_meteo_opt.columns]
                df_meteo_opt = df_meteo_opt.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
                
                df_campo_opt = df_campo_raw.copy()
                col_fecha_opt = 'FECHA' if 'FECHA' in df_campo_opt.columns else df_campo_opt.columns[0]
                df_campo_opt[col_fecha_opt] = pd.to_datetime(df_campo_opt[col_fecha_opt])
                
                tabla_optima = optimizar_parametros_hidricos_2d(
                    df_meteo_opt,
                    df_campo_opt,
                    modelo_ann,
                    latitud_bordenave=-37.761671,
                    calentamiento_suelo=calentamiento_suelo,
                    umbral_choque_hidrico=umbral_choque_hidrico
                )
                
            st.success("¡Barrido 2D por eventos completado de forma rigurosa!")
            st.dataframe(tabla_optima.head(15))
        else:
            st.error("Se requieren datos de Clima y Campo.")

# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:

    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Simulación Térmica
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
    df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * mod_termico)
    df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * mod_termico)

    # CALENTAMIENTO DIAGNÓSTICO DEL SUELO; NO MODIFICA LAS ENTRADAS DE LA ANN
    mask_oi = (df["Julian_days"] >= 152) & (df["Julian_days"] <= 264)
    df.loc[mask_oi, "TMAX_suelo"] += calentamiento_suelo
    df.loc[mask_oi, "TMIN_suelo"] += calentamiento_suelo

    df_campo, col_fecha, col_plm2 = None, None, None
    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        max_plm2 = df_campo[col_plm2].max()
        df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2 if max_plm2 > 0 else 0

    # ----------------------------------------------------
    # CORRECCIÓN: Lógica Fisiológica Ordenada
    # ----------------------------------------------------
    # 1. Predicción Neural Base
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    # 2. Bypass Ruptura Temprana (Bordenave = 1.0) - ESTRICTAMENTE LUEGO DEL DÍA 15
    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (df["Julian_days"] > 15) & (df["Julian_days"] <= 110) & (df["Prec_3d"] >= umbral_choque_hidrico)
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 1.0)

    # 3. Balance Hídrico Superficial (Bordenave)
    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-37.761671)
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo=ke_val)
    humedad_relativa = df["W_superficial"] / w_max_val
    df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]

    df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0
    df['Lluvia_Recarga'] = (df['Prec'] >= w_max_val).cummax()
    df.loc[~df['Lluvia_Recarga'], "EMERREL"] = 0.0

    # 4. Escudo Termofisiológico
    df["Tmedia"] = df["Tmedia_aire"]
    df["Tmedia_5d"] = df["Tmedia"].rolling(window=5, min_periods=1).mean()
    df.loc[df["Tmedia_5d"] >= umbral_termoinhibicion, "EMERREL"] = 0.0

    # 5. BLOQUEO FINAL ESTRICTO: Latencia Temprana (Primeros 15 días del año)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0

    # 6. VALIDACIÓN DEL PRIMER PICO
    # La campaña comienza en el primer valor estrictamente superior a 0.20.
    df, idx_primer_pico = aplicar_filtro_primer_pico(
        df,
        umbral=UMBRAL_PRIMER_PICO
    )
    # ----------------------------------------------------

    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values: fecha_hoy = df['Fecha'].max()
    indices_pulso = [idx_primer_pico] if idx_primer_pico is not None else []

    # --- CÁLCULO DE FECHA LÍMITE (800 °Cd) ---
    dga_hoy, dga_7dias = 0.0, 0.0
    fecha_inicio_ventana, fecha_control, fecha_limite = None, None, None
    msg_estado = "Esperando pico de emergencia..."

    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
        
        # Fecha de 600 °Cd
        df_control = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_control.empty: fecha_control = df_control.iloc[0]["Fecha"]
        
        # Fecha de 800 °Cd
        df_limite = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_critico]
        if not df_limite.empty: fecha_limite = df_limite.iloc[0]["Fecha"]
        
        dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        
        dga_7dias = dga_hoy + df.iloc[idx_hoy + 1: idx_hoy + 8]["DG"].sum() if idx_hoy + 8 <= len(df) else dga_hoy
        msg_estado = (
            f"Pico validado > {UMBRAL_PRIMER_PICO:.2f} "
            f"el {fecha_inicio_ventana.strftime('%d/%m')}"
        )

    # Sincronización Rigurosa por Eventos Variables
    pearson_r, nse_flujos, kge_flujos, rmse_acum, ccc_acum, r2_acum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    exactitud_global, f1_score_coincidencia = 0.0, 0.0
    hits_val, misses_val, falsos_pos_val, correctos_neg_val = 0, 0, 0, 0
    pec, peak_lag, lead_time, desfase_t50 = 0.0, 0, 0, 0
    df_sincronizado = pd.DataFrame()
    
    # --- CÁLCULO DE SINCRONÍA DE INICIO ---
    lag_inicio_dias = None
    fecha_primer_flujo_obs = None

    if df_campo is not None:
        muestreos_con_plantas = df_campo[df_campo[col_plm2] > 0]
        if not muestreos_con_plantas.empty:
            fecha_primer_flujo_obs = muestreos_con_plantas.iloc[0][col_fecha]
            if fecha_inicio_ventana is not None:
                lag_inicio_dias = (fecha_inicio_ventana - fecha_primer_flujo_obs).days

        df_sincronizado = sincronizar_intervalos_variables(df, df_campo, col_fecha, col_plm2)
        if not df_sincronizado.empty:
            metricas_robustas = calcular_metricas_validacion_integral(df_sincronizado, umbral_deteccion=0.05)
            
            pearson_r = metricas_robustas["Pearson_Flujos"]
            nse_flujos = metricas_robustas["NSE_Flujos"]
            kge_flujos = metricas_robustas["KGE_Flujos"]
            rmse_acum = metricas_robustas["RMSE_Acumulado"]
            ccc_acum = metricas_robustas["CCC_Acumulado"]
            r2_acum = metricas_robustas["R2_Acumulado"]
            exactitud_global = metricas_robustas["Exactitud_Global"]
            f1_score_coincidencia = metricas_robustas["F1_Score_Coincidencia"]
            
            hits_val = metricas_robustas["Hits"]
            misses_val = metricas_robustas["Misses"]
            falsos_pos_val = metricas_robustas["Falsos_Positivos"]
            correctos_neg_val = metricas_robustas["Correctos_Negativos"]

            tot_plm2 = df_campo[col_plm2].sum()
            if tot_plm2 > 0:
                df_campo['cum_plm2_norm'] = df_campo[col_plm2].cumsum() / tot_plm2
                t50_obs_date = df_campo[df_campo['cum_plm2_norm'] >= 0.5].iloc[0][col_fecha]
                df_sim_trunc = df[df['Fecha'] <= df_campo[col_fecha].max()].copy()
                tot_emer = df_sim_trunc['EMERREL'].sum()
                
                if tot_emer > 0:
                    df_sim_trunc['cum_emer_norm'] = df_sim_trunc['EMERREL'].cumsum() / tot_emer
                    t50_sim_date = df_sim_trunc[df_sim_trunc['cum_emer_norm'] >= 0.5].iloc[0]['Fecha']
                    desfase_t50 = (t50_sim_date - t50_obs_date).days

            if fecha_control:
                malezas_totales_campo = df_campo[col_plm2].sum()
                pec = ((df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum() / malezas_totales_campo) * 100 if malezas_totales_campo > 0 else 0)
                peak_lag = (fecha_control - df_campo.loc[df_campo[col_plm2].idxmax(), col_fecha]).days
                df_alertas = df[df['EMERREL'] >= umbral_er]
                lead_time = (fecha_control - (df_alertas['Fecha'].iloc[0] if not df_alertas.empty else fecha_inicio_ventana)).days

    # Transformación Logarítmica Analítica
    c_log = 0.01
    df["EMERREL_LOG"] = np.log10(df["EMERREL"] + c_log)
    umbral_er_log = np.log10(umbral_er + c_log)
    if df_campo is not None:
        df_campo['Campo_Normalizado_LOG'] = np.log10(df_campo['Campo_Normalizado'] + c_log)

    # VISUALIZACIÓN FRONT-END
    colorscale_hard = [[0.0, "green"], [0.01, "green"], [0.02, "red"], [1.0, "red"]]
    st.plotly_chart(go.Figure(data=go.Heatmap(z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"], colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False)).update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Riesgo Temporal (Tasa Diaria Bordenave)"), width="stretch")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DE DECISIÓN", "💧 PRECIPITACIONES Y SUELO", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        if df_campo is not None:
            st.markdown("<p class='metric-header'>🚜 FIDELIDAD DE SIMULACIÓN (Evaluación Rigurosa sobre Ventanas Reales de Campo)</p>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            
            c1.metric("Eficiencia (KGE)", f"{kge_flujos:.3f}", "Ajuste por Evento")
            c2.metric("Predictivo (NSE)", f"{nse_flujos:.3f}", "Flujos Puros")
            c3.metric("Trayectoria (CCC)", f"{ccc_acum:.3f}", "Curva Acum.")
            c4.metric("Error (RMSE)", f"{rmse_acum:.3f}", "Desvío Acumulado", delta_color="inverse")
            c5.metric("Desfase (T50)", f"{desfase_t50:+d} días", "Sincronía Operativa", delta_color="inverse" if desfase_t50 > 0 else "normal" if desfase_t50 < 0 else "off")

            st.markdown("<p class='metric-header' style='margin-top:15px;'>🎯 COINCIDENCIA POR INTERVALO DE MUESTREO</p>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("F1-Score (Coincidencia)", f"{f1_score_coincidencia:.3f}", "Fidelidad en ventanas activas")
            d2.metric("Exactitud Global", f"{exactitud_global * 100:.1f}%", "Acuerdo total (Invierno + Verano)")
            d3.metric("Falsos Positivos", f"{falsos_pos_val}", "Intervalos simulados sin contraparte real", delta_color="inverse")

            # --- SECCIÓN: SINCRONÍA DE INICIO ---
            st.markdown("<p class='metric-header' style='margin-top:15px;'>⏰ SINCRONÍA DE INICIO (Gatillo de Tiempo Térmico)</p>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            s1.metric("Inicio Simulado", fecha_inicio_ventana.strftime('%d-%m-%Y') if fecha_inicio_ventana else "N/A")
            s2.metric("Inicio Observado (Campo)", fecha_primer_flujo_obs.strftime('%d-%m-%Y') if fecha_primer_flujo_obs else "N/A")
            
            str_lag = "N/A"
            if lag_inicio_dias is not None:
                str_lag = f"{lag_inicio_dias:+} días"
            s3.metric("Desfase de Gatillo", str_lag, "Negativo = Modelo Anticipa", delta_color="inverse" if (lag_inicio_dias and lag_inicio_dias > 0) else "normal" if (lag_inicio_dias and lag_inicio_dias < 0) else "off")
            
            # --- TABLA HTML: MATRIZ DE CONFUSIÓN ---
            html_cm = f"""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.1); border:1px solid #e2e8f0; margin-top:15px;">
                <p style="color:#1e293b; font-weight:bold; margin-top:0; margin-bottom:10px;">🧩 Matriz de Confusión (Intervalos de Monitoreo)</p>
                <table style="width:100%; text-align:center; border-collapse: collapse; font-family:sans-serif;">
                    <tr>
                        <th style="border-bottom:2px solid #e2e8f0; padding:10px; color:#475569; width:34%;">Realidad ⬇ \\ Simulación ➡</th>
                        <th style="border-bottom:2px solid #e2e8f0; padding:10px; background-color:#eff6ff; color:#1e3a8a; width:33%;">🚨 Modelo Predice FLUJO</th>
                        <th style="border-bottom:2px solid #e2e8f0; padding:10px; background-color:#f8fafc; color:#475569; width:33%;">💤 Modelo Predice INACTIVO</th>
                    </tr>
                    <tr>
                        <td style="border-bottom:1px solid #e2e8f0; padding:10px; font-weight:bold; color:#166534; background-color:#f0fdf4;">🌱 Campo: HUBO Flujo</td>
                        <td style="border-bottom:1px solid #e2e8f0; padding:10px; background-color:#dcfce7; font-size:1.1em; font-weight:bold; color:#166534;">{hits_val} <span style="font-size:0.8em; font-weight:normal;">(Hits / Coincidencia)</span></td>
                        <td style="border-bottom:1px solid #e2e8f0; padding:10px; background-color:#fee2e2; font-size:1.1em; font-weight:bold; color:#991b1b;">{misses_val} <span style="font-size:0.8em; font-weight:normal;">(Omisiones)</span></td>
                    </tr>
                    <tr>
                        <td style="padding:10px; font-weight:bold; color:#475569; background-color:#f8fafc;">🛑 Campo: SIN Flujo</td>
                        <td style="padding:10px; background-color:#fee2e2; font-size:1.1em; font-weight:bold; color:#991b1b;">{falsos_pos_val} <span style="font-size:0.8em; font-weight:normal;">(Falsas Alarmas)</span></td>
                        <td style="padding:10px; background-color:#dcfce7; font-size:1.1em; font-weight:bold; color:#166534;">{correctos_neg_val} <span style="font-size:0.8em; font-weight:normal;">(Correctos Negativos)</span></td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(html_cm, unsafe_allow_html=True)

            if fecha_control:
                st.markdown("<p class='metric-header' style='margin-top:15px;'>⚙️ LOGÍSTICA DE CONTROL EN LOTE</p>", unsafe_allow_html=True)
                l1, l2, l3 = st.columns(3)
                l1.metric("Control Efectivo (PEC)", f"{pec:.1f}%", "A la fecha de aplicación")
                l2.metric("Lag (Desfase)", f"{peak_lag} días", "Vs Pico de Campo Real")
                l3.metric("Lead Time", f"{lead_time} días", "Ventana de Alerta")
            st.markdown("---")

        col_main, col_gauge = st.columns([3.4, 1])

        with col_main:
            fig_emer = go.Figure()

            # Bandas alternas según los intervalos reales de monitoreo.
            if df_campo is not None:
                fechas_reales_lote = (
                    pd.to_datetime(
                        df_campo[col_fecha],
                        errors="coerce"
                    )
                    .dropna()
                    .sort_values()
                    .tolist()
                )
                for i in range(1, len(fechas_reales_lote), 2):
                    fig_emer.add_vrect(
                        x0=fechas_reales_lote[i - 1],
                        x1=fechas_reales_lote[i],
                        fillcolor="rgba(148, 163, 184, 0.065)",
                        layer="below",
                        line_width=0,
                    )

            # Ventana agronómica eficiente con verde suave.
            if fecha_control is not None and fecha_limite is not None:
                fig_emer.add_vrect(
                    x0=fecha_control,
                    x1=fecha_limite,
                    fillcolor="rgba(34, 197, 94, 0.10)",
                    layer="below",
                    line_width=0,
                )

            # Banda de residualidad, si corresponde.
            if fecha_control is not None and residualidad > 0:
                fig_emer.add_vrect(
                    x0=fecha_control,
                    x1=fecha_control + timedelta(days=int(residualidad)),
                    fillcolor="rgba(37, 99, 235, 0.055)",
                    layer="below",
                    line_width=0,
                )

            # Serie simulada: sin relleno para mejorar la lectura.
            fig_emer.add_trace(
                go.Scatter(
                    x=df["Fecha"],
                    y=df["EMERREL_LOG"],
                    mode="lines",
                    name="Tasa diaria simulada (log)",
                    line=dict(
                        color="#075FCF",
                        width=2.4,
                    ),
                    hovertemplate=(
                        "<b>%{x|%d-%m-%Y}</b><br>"
                        "Simulado: %{y:.3f}<extra></extra>"
                    ),
                )
            )

            # Serie observada.
            if df_campo is not None:
                fig_emer.add_trace(
                    go.Scatter(
                        x=df_campo[col_fecha],
                        y=df_campo["Campo_Normalizado_LOG"],
                        mode="markers+lines",
                        name="Campo normalizado (log)",
                        marker=dict(
                            color="#60A5FA",
                            size=9,
                            symbol="circle",
                            line=dict(
                                color="#FFFFFF",
                                width=1.4,
                            ),
                        ),
                        line=dict(
                            color="#60A5FA",
                            width=2.2,
                            dash="dash",
                        ),
                        hovertemplate=(
                            "<b>%{x|%d-%m-%Y}</b><br>"
                            "Campo: %{y:.3f}<extra></extra>"
                        ),
                    )
                )

            # Umbral de alerta.
            fig_emer.add_hline(
                y=umbral_er_log,
                line_dash="dash",
                line_color="#111827",
                line_width=1.7,
            )
            fig_emer.add_annotation(
                x=0.995,
                xref="paper",
                y=umbral_er_log,
                yref="y",
                text=f"Alerta ({umbral_er:.3f})",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                yshift=7,
                bgcolor="rgba(255,255,255,0.94)",
                bordercolor="rgba(148,163,184,0.50)",
                borderwidth=1,
                borderpad=4,
                font=dict(
                    size=12,
                    color="#374151",
                ),
            )

            # Inicio del recuento térmico.
            if fecha_inicio_ventana is not None:
                fig_emer.add_vline(
                    x=fecha_inicio_ventana,
                    line_dash="dot",
                    line_color="#111827",
                    line_width=1.5,
                )
                fig_emer.add_annotation(
                    x=fecha_inicio_ventana,
                    xref="x",
                    y=1.035,
                    yref="paper",
                    text="Inicio térmico",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor="rgba(148,163,184,0.45)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(
                        size=12,
                        color="#111827",
                    ),
                )

            # Fecha de control.
            if fecha_control is not None:
                fig_emer.add_vline(
                    x=fecha_control,
                    line_dash="dot",
                    line_color="#111827",
                    line_width=1.5,
                )
                fig_emer.add_annotation(
                    x=fecha_control,
                    xref="x",
                    y=1.105,
                    yref="paper",
                    text=f"Control ({dga_optimo} °Cd)",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor="rgba(148,163,184,0.45)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(
                        size=12,
                        color="#111827",
                    ),
                )

            # Fecha límite.
            if fecha_limite is not None:
                fig_emer.add_vline(
                    x=fecha_limite,
                    line_dash="dot",
                    line_color="#166534",
                    line_width=1.5,
                )
                fig_emer.add_annotation(
                    x=fecha_limite,
                    xref="x",
                    y=1.035,
                    yref="paper",
                    text=f"Límite ({dga_critico} °Cd)",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    bgcolor="rgba(240,253,244,0.96)",
                    bordercolor="rgba(34,197,94,0.45)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(
                        size=12,
                        color="#166534",
                    ),
                )

            # Rótulo de la ventana eficiente.
            if fecha_control is not None and fecha_limite is not None:
                fig_emer.add_annotation(
                    x=fecha_control + (fecha_limite - fecha_control) / 2,
                    xref="x",
                    y=0.975,
                    yref="paper",
                    text="Ventana eficiente",
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    bgcolor="rgba(240,253,244,0.94)",
                    borderpad=3,
                    font=dict(
                        size=11,
                        color="#166534",
                    ),
                )

            # Marcas mensuales en español.
            primer_mes = pd.Timestamp(
                df["Fecha"].min()
            ).to_period("M").to_timestamp()
            ultimo_mes = pd.Timestamp(
                df["Fecha"].max()
            ).to_period("M").to_timestamp()
            marcas_mensuales = pd.date_range(
                primer_mes,
                ultimo_mes,
                freq="MS",
            )
            nombres_meses = {
                1: "Ene",
                2: "Feb",
                3: "Mar",
                4: "Abr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Ago",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dic",
            }
            etiquetas_meses = [
                f"{nombres_meses[fecha.month]} {fecha.year}"
                for fecha in marcas_mensuales
            ]

            fig_emer.update_layout(
                title=dict(
                    text=(
                        "Dinámica fisiológica de emergencia "
                        "(intervalos reales de monitoreo)"
                    ),
                    x=0.0,
                    xanchor="left",
                    font=dict(
                        size=21,
                        color="#111827",
                        family="Arial, sans-serif",
                    ),
                ),
                xaxis=dict(
                    title=dict(
                        text="Fecha",
                        font=dict(
                            size=15,
                            color="#4B5563",
                        ),
                        standoff=14,
                    ),
                    tickmode="array",
                    tickvals=marcas_mensuales,
                    ticktext=etiquetas_meses,
                    tickfont=dict(
                        size=12,
                        color="#4B5563",
                    ),
                    showgrid=False,
                    showline=True,
                    linecolor="#6B7280",
                    linewidth=1,
                    ticks="outside",
                    ticklen=6,
                    zeroline=False,
                    automargin=True,
                ),
                yaxis=dict(
                    title=dict(
                        text="Log10(EMERREL + 0,01)",
                        font=dict(
                            size=15,
                            color="#4B5563",
                        ),
                        standoff=12,
                    ),
                    range=[-2.18, 0.12],
                    tickmode="array",
                    tickvals=[-2.0, -1.5, -1.0, -0.5, 0.0],
                    tickfont=dict(
                        size=12,
                        color="#4B5563",
                    ),
                    showgrid=True,
                    gridcolor="rgba(148, 163, 184, 0.28)",
                    griddash="dash",
                    gridwidth=1,
                    showline=True,
                    linecolor="#6B7280",
                    linewidth=1,
                    zeroline=False,
                    automargin=True,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.17,
                    xanchor="right",
                    x=1.0,
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="rgba(148,163,184,0.35)",
                    borderwidth=1,
                    font=dict(
                        size=12,
                        color="#374151",
                    ),
                ),
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#CBD5E1",
                    font=dict(
                        size=12,
                        color="#111827",
                    ),
                ),
                height=630,
                margin=dict(
                    l=82,
                    r=28,
                    t=132,
                    b=78,
                ),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(
                    family="Arial, sans-serif",
                    color="#374151",
                ),
                dragmode="zoom",
            )

            fig_emer.update_xaxes(
                rangeslider_visible=False,
                fixedrange=False,
            )
            fig_emer.update_yaxes(fixedrange=False)

            st.plotly_chart(
                fig_emer,
                width="stretch",
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "scrollZoom": True,
                    "modeBarButtonsToRemove": [
                        "lasso2d",
                        "select2d",
                    ],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "PREDWEEM_Bordenave_dinamica_emergencia",
                        "height": 1000,
                        "width": 2000,
                        "scale": 2,
                    },
                },
            )

            if fecha_inicio_ventana:
                st.success(
                    f"📅 **Inicio de Conteo Térmico:** "
                    f"{fecha_inicio_ventana.strftime('%d-%m-%Y')} "
                    f"(primer pico validado > {UMBRAL_PRIMER_PICO:.2f})"
                )
                if fecha_control:
                    st.error(
                        f"🎯 **MOMENTO CRÍTICO DE CONTROL:** "
                        f"{fecha_control.strftime('%d-%m-%Y')}. "
                        f"Se acumularon **{dga_optimo} °Cd** "
                        f"post-emergencia."
                    )
            else:
                st.warning(
                    f"⏳ Esperando un primer pico de emergencia "
                    f"estrictamente superior a "
                    f"{UMBRAL_PRIMER_PICO:.2f}."
                )

        col_gauge = col_gauge  # Mantenimiento de estructura visual
        with col_gauge:
            max_axis = dga_critico * 1.2
            st.plotly_chart(go.Figure().add_trace(go.Indicator(mode="gauge+number", value=dga_hoy, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "<b>TT POST-EMERGENCIA (°Cd)</b>", 'font': {'size': 18}}, gauge={'axis': {'range': [None, max_axis]}, 'bar': {'color': "#1e293b", 'thickness': 0.3}, 'steps': [{'range': [0, dga_optimo], 'color': "#4ade80"}, {'range': [dga_optimo, dga_critico], 'color': "#facc15"}, {'range': [dga_critico, max_axis], 'color': "#f87171"}], 'threshold': {'line': {'color': "#2563eb", 'width': 6}, 'thickness': 0.8, 'value': dga_7dias}})).add_annotation(x=0.5, y=-0.1, text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>", showarrow=False, font=dict(size=14, color="#1e3a8a"), align="center").update_layout(height=350, margin=dict(t=80, b=50, l=30, r=30)), width="stretch")

        if df_campo is not None and not df_sincronizado.empty:
            st.markdown("---")
            st.markdown("<p class='metric-header' style='margin-top:20px;'>📈 DISPERSIÓN DE AJUSTE Y RECTAS 1:1</p>", unsafe_allow_html=True)
            col_curva, col_disp = st.columns([2, 1])
            
            with col_curva:
                fig_acum = go.Figure()
                fig_acum.add_trace(go.Scatter(x=df_sincronizado['Fecha'], y=df_sincronizado['Campo_Acumulado'] * 100, mode='markers+lines', name='Real de Campo (%)', marker=dict(color='#dc2626', size=8, symbol='diamond'), line=dict(color='#dc2626', width=2)))
                fig_acum.add_trace(go.Scatter(x=df_sincronizado['Fecha'], y=df_sincronizado['Sim_Acumulado'] * 100, mode='lines', name='Simulado PREDWEEM (%)', line=dict(color='#166534', width=3, dash='dash')))
                st.plotly_chart(fig_acum.update_layout(title="Llenado Cinético (Curvas Acumuladas Puras)", xaxis_title="Calendario", yaxis_title="Emergencia Acumulada (%)", height=430, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)), width="stretch")

            with col_disp:
                tab_flujos, tab_acum = st.tabs(["1:1 Flujos", "1:1 Acumulado"])
                
                with tab_flujos:
                    fig_1to1 = go.Figure()
                    fig_1to1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='1:1', line=dict(color='gray', dash='dash')))
                    fig_1to1.add_trace(go.Scatter(
                        x=df_sincronizado['Campo_Relativo'], y=df_sincronizado['Sim_Relativo'], 
                        mode='markers', name='Eventos',
                        marker=dict(color='#2563eb', size=12, line=dict(width=1, color='DarkBlue')),
                        text=df_sincronizado['Fecha'].dt.strftime('%d-%m-%Y'),
                        hovertemplate="<b>Intervalo fin: %{text}</b><br>Obs: %{x:.3f}<br>Sim: %{y:.3f}<extra></extra>"
                    ))
                    st.plotly_chart(fig_1to1.update_layout(title="Ajuste 1:1 de Flujos por Evento Real", xaxis_title="Observado Relativo", yaxis_title="Simulado Relativo", height=380, showlegend=False, margin=dict(t=40, b=0, l=0, r=0)), width="stretch")

                with tab_acum:
                    fig_1to1_ac = go.Figure()
                    fig_1to1_ac.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='1:1', line=dict(color='gray', dash='dash')))
                    fig_1to1_ac.add_trace(go.Scatter(
                        x=df_sincronizado['Campo_Acumulado'], y=df_sincronizado['Sim_Acumulado'],
                        mode='markers', name='Acumulado',
                        marker=dict(color='#dc2626', size=12, symbol='diamond', line=dict(width=1, color='DarkRed')),
                        text=df_sincronizado['Fecha'].dt.strftime('%d-%m-%Y'),
                        hovertemplate="<b>%{text}</b><br>Obs Acum: %{x:.3f}<br>Sim Acum: %{y:.3f}<extra></extra>"
                    ))
                    st.plotly_chart(fig_1to1_ac.update_layout(title=f"Ajuste 1:1 Acumulado (R²: {r2_acum:.3f} | RMSE: {rmse_acum:.3f})", xaxis_title="Obs. Acumulada (Norm)", yaxis_title="Sim. Acumulada (Norm)", height=380, showlegend=False, margin=dict(t=40, b=0, l=0, r=0)), width="stretch")

    with tab2:
        st.header("💧 Dinámica Hídrica del Suelo (Bordenave)")
        fig_hidrico = go.Figure()
        fig_hidrico.add_trace(go.Bar(x=df["Fecha"], y=df["Prec"], name='Lluvia Diaria (mm)', marker_color='#93c5fd', opacity=0.7))
        fig_hidrico.add_trace(go.Scatter(x=df["Fecha"], y=df["W_superficial"], name='Agua en Suelo (0-10cm)', mode='lines', line=dict(color='#0284c7', width=3), fill='tozeroy', fillcolor='rgba(2, 132, 199, 0.2)'))
        fig_hidrico.add_hline(y=w_max_val, line_dash="dot", line_color="#334155", annotation_text=f"Capacidad Máx. Suelo ({w_max_val} mm)", annotation_position="top left")
        st.plotly_chart(fig_hidrico.update_layout(title="Precipitación vs. Retención Real de Humedad", xaxis_title="Fecha", yaxis_title="Milímetros (mm)", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)), width="stretch")

    with tab3:
        st.header("🔍 Clasificación DTW (Bordenave)")
        df_obs = df[df["Fecha"] < pd.Timestamp("2026-05-01")].copy()
        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)
            dists = [dtw_distance(obs_norm, m[JD_COM <= jd_corte] / m[JD_COM <= jd_corte].max() if m[JD_COM <= jd_corte].max() > 0 else m[JD_COM <= jd_corte]) for m in cluster_model["curves_interp"]]
            pred = int(np.argmin(dists))
            cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}

            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(x=JD_COM, y=cluster_model["curves_interp"][pred], name="Patrón Histórico", line=dict(dash='dash', color=cols.get(pred))))
                fp.add_trace(go.Scatter(x=jd_grid, y=obs_norm * cluster_model["curves_interp"][pred].max(), name="2026", line=dict(color='black', width=3)))
                st.plotly_chart(fp, width="stretch")
            with c2:
                nombres_patrones = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
                st.success(f"### {nombres_patrones.get(pred, 'Desconocido')}")
                st.metric("DTW Score", f"{min(dists):.2f}")
        else:
            st.info("Datos insuficientes para clasificación por Dinámica Temporal (DTW).")

    with tab4:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)
        st.plotly_chart(go.Figure().add_trace(go.Scatter(x=x_temps, y=[calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps], mode='lines', line=dict(color='#2563eb', width=4), fill='tozeroy')), width="stretch")

    # REPORTE EN EXCEL
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        if df_campo is not None and not df_sincronizado.empty:
            df_campo.to_excel(writer, index=False, sheet_name='Campo_Validacion')
            val_lag = lag_inicio_dias if lag_inicio_dias is not None else "N/A"
            pd.DataFrame({
                'Métrica de Validación': ['PEC (%)', 'Lag Control (días)', 'Lead Time Control (días)', 'Pearson (Flujos)', 'NSE (Flujos Reales Evento)', 'KGE (Flujos)', 'RMSE (Acumulado)', 'R2 (Acumulado)', 'CCC (Acumulado)', 'Desfase T50 Global (días)', 'F1-Score (Coincidencia)', 'Exactitud Global', 'Hits (Aciertos)', 'Misses (Omisiones)', 'Falsos Positivos', 'Correctos Negativos', 'Desfase Primer Flujo (días)'], 
                'Valor': [pec, peak_lag, lead_time, pearson_r, nse_flujos, kge_flujos, rmse_acum, r2_acum, ccc_acum, desfase_t50, f1_score_coincidencia, exactitud_global, hits_val, misses_val, falsos_pos_val, correctos_neg_val, val_lag]
            }).to_excel(writer, sheet_name='Validacion_Campo', index=False)
        pd.DataFrame({
            'Configuracion': [
                'T_Base', 'T_Optima', 'T_Critica', 'W_Max', 'Ke',
                'Mod_Termico', 'Umbral_Termoinhibicion',
                'Umbral_Choque_Hidrico_3d',
                'Umbral_Primer_Pico',
                'Calentamiento_Suelo_OI'
            ],
            'Valor': [
                t_base_val, t_opt_max, t_critica, w_max_val, ke_val,
                mod_termico, umbral_termoinhibicion,
                umbral_choque_hidrico,
                UMBRAL_PRIMER_PICO,
                calentamiento_suelo
            ]
        }).to_excel(writer, sheet_name='Bio_Params', index=False)

    st.sidebar.download_button("📥 Descargar Reporte Bordenave", output.getvalue(), "PREDWEEM_Integral_Bordenave_vK4_9_15.xlsx")

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue los datos climáticos de Bordenave para comenzar.")
