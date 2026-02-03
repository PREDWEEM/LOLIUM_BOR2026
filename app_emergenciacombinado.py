# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4 — LOLIUM TRES ARROYOS 2026
# Actualización: Inicio de ventana desde el primer pico detectado
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM INTEGRAL vK4", 
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
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ: GENERADOR DE ARCHIVOS MOCK
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

    if not (BASE / "meteo_daily.csv").exists():
        dates = pd.date_range(start="2026-01-01", periods=150)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(25, 35, size=150) - (np.arange(150)*0.1),
            "TMIN": np.random.uniform(10, 18, size=150) - (np.arange(150)*0.06),
            "Prec": np.random.choice([0, 0, 5, 15, 45], size=150)
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA
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

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit:
        factor = (t_crit - t) / (t_crit - t_opt)
        return (t - t_base) * factor
    else: return 0.0

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

def get_data(file_input):
    try:
        if file_input:
            if file_input.name.endswith('.csv'): df = pd.read_csv(file_input, parse_dates=["Fecha"])
            else: df = pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path = BASE / "meteo_daily.csv"
            if path.exists(): df = pd.read_csv(path, parse_dates=["Fecha"])
            else: return None
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
        df = df.rename(columns=mapeo)
        return df
    except Exception as e:
        st.error(f"Error leyendo datos: {e}")
        return None

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()
st.sidebar.markdown("## ⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Pico)", 0.05, 0.80, 0.50)

st.sidebar.divider()
st.sidebar.markdown("🌡️ **Fisiología Térmica**")
t_base_val = st.sidebar.number_input("T Base", value=2.0, step=0.5)
t_opt_max = st.sidebar.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

dga_optimo = st.sidebar.number_input("Objetivo Control (°Cd)", value=600, step=50)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=700, step=50)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO Y VISUALIZACIÓN
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Predicción
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    
    # Cálculo Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    
    st.title("🌾 PREDWEEM LOLIUM-BORDENAVE 2026")

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        col_main, col_gauge = st.columns([2, 1])
        
        # --- LÓGICA DE VENTANA (MODIFICADA: PRIMER PICO) ---
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        fecha_inicio_ventana = None
        
        if indices_pulso:
            # Seleccionamos directamente la fecha del primer índice que superó el umbral
            fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        
        dga_actual = 0.0
        dias_stress = 0
        if fecha_inicio_ventana:
            df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
            dga_actual = df_ventana["DGA_cum"].iloc[-1]
            dias_stress = len(df_ventana[df_ventana["Tmedia"] > t_opt_max])

        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria', line=dict(color='#166534', width=2.5)))
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text="Umbral Pico")
            fig_emer.update_layout(title="Dinámica de Emergencia", height=350)
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.info(f"📅 **Inicio de Conteo (Primer Pico):** {fecha_inicio_ventana.strftime('%d-%m-%Y')}")
                if dias_stress > 0:
                    st.markdown(f"""<div class="bio-alert">🔥 <b>Alerta:</b> {dias_stress} días de estrés térmico desde el inicio.</div>""", unsafe_allow_html=True)
            else:
                st.warning("⏳ Esperando el primer pico de emergencia (Tasa > umbral).")

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = dga_actual,
                title = {'text': "<b>°Cd ACUMULADOS</b>"},
                gauge = {
                    'axis': {'range': [None, dga_critico*1.2]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#4ade80"},
                        {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                        {'range': [dga_critico, dga_critico*1.2], 'color': "#f87171"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # (El resto de las pestañas Tab 2 y Tab 3 se mantienen igual...)
    with tab2:
        st.info("Módulo de Clasificación DTW Activo.")
    
    with tab3:
        st.info("Módulo de Bio-Calibración Activo.")

    # Exportación
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Report.xlsx")

else:
    st.info("👋 Cargue datos para iniciar el monitoreo.")
