# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.1 — LOLIUM TRES ARROYOS 2026
# Actualización: Inicio de conteo desde el PRIMER pico + Heatmap
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
# 3. LÓGICA TÉCNICA (ANN + DTW + BIO)
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

# Función centralizada para cálculo de TT (usada en Plot y en Dataframe)
def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base:
        return 0.0
    elif t <= t_opt:
        return t - t_base
    elif t < t_crit:
        # Ponderación lineal descendente
        factor = (t_crit - t) / (t_crit - t_opt)
        return (t - t_base) * factor
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
            if file_input.name.endswith('.csv'):
                df = pd.read_csv(file_input, parse_dates=["Fecha"])
            else:
                df = pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path = BASE / "meteo_daily.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=["Fecha"])
            else:
                return None
        
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {
            'FECHA': 'Fecha', 'DATE': 'Fecha', 
            'TMAX': 'TMAX', 'TMIN': 'TMIN', 
            'PREC': 'Prec', 'LLUVIA': 'Prec'
        }
        df = df.rename(columns=mapeo)
        return df
    except Exception as e:
        st.error(f"Error leyendo datos: {e}")
        return None

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Emergencia**")
umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Para detectar pico)", 0.05, 0.80, 0.50)

st.sidebar.divider()
st.sidebar.markdown("🌡️ **Fisiología Térmica (Bio-Limit)**")
st.sidebar.caption("Ajusta la respuesta biológica al calor.")

# --- CONTROLES TÉRMICOS ---
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2:
    t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)

t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Control", value=60, step=50)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=50)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    
    # A. Preprocesamiento
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # B. Predicción Neural
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 0, "EMERREL"] = 0.0 
    
    # C. CÁLCULO BIO-TÉRMICO
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    # Aplicamos la función centralizada
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    
    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM-BORDENAVE 2026")

    # --- GRÁFICO DE CALOR (HEATMAP) ---
    colorscale_hard = [[0.0, "green"], [0.49, "green"], [0.50, "yellow"], [0.79, "yellow"], [0.80, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
    ))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Intensidad de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True)

    # TABS PRINCIPALES
    tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    # --- TAB 1: MONITOR ---
    with tab1:
        col_main, col_gauge = st.columns([2, 1])
        
        # ==============================================================================
        # MODIFICACIÓN CLAVE: Lógica de Ventana basada en el PRIMER PICO
        # ==============================================================================
        # Obtenemos los índices donde la tasa diaria supera el umbral definido en el sidebar
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        
        fecha_inicio_ventana = None
        
        # Si existe al menos un índice que cumpla la condición...
        if indices_pulso:
            # ...tomamos la fecha del PRIMERO de ellos.
            first_peak_index = indices_pulso[0]
            fecha_inicio_ventana = df.loc[first_peak_index, "Fecha"]
        
        # Cálculo de acumulados si hay ventana activa
        dga_actual = 0.0
        dias_stress = 0
        if fecha_inicio_ventana:
            df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
            # Tomamos el último valor acumulado
            dga_actual = df_ventana["DGA_cum"].iloc[-1] if not df_ventana.empty else 0.0
            dias_stress = len(df_ventana[df_ventana["Tmedia"] > t_opt_max])
        # ==============================================================================

        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(
                x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria',
                line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
            ))
            # Línea del umbral para referencia visual
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral Pico ({umbral_er})")
            
            fig_emer.update_layout(title="Dinámica de Emergencia y Detección de Picos", height=350)
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico detectado)")
                if dias_stress > 0:
                    st.markdown(f"""<div class="bio-alert">🔥 <b>Estrés Térmico:</b> {dias_stress} días con T > {t_opt_max}°C desde el inicio.</div>""", unsafe_allow_html=True)
            else:
                st.warning(f"⏳ Esperando el primer pico de emergencia (Tasa diaria >= {umbral_er}).")

        with col_gauge:
            max_axis = dga_critico * 1.2
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta", value = dga_actual,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<b>ACUMULACIÓN TÉRMICA (°Cd)</b><br><span style='font-size:0.8em;color:gray'>Desde primer pico</span>"},
                gauge = {
                    'axis': {'range': [None, max_axis]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#4ade80"},
                        {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                        {'range': [dga_critico, max_axis], 'color': "#f87171"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 2: ANÁLISIS ---
    with tab2:
        st.header("🔍 Clasificación DTW")
        fecha_corte = pd.Timestamp("2026-05-01")
        df_obs = df[df["Fecha"] < fecha_corte].copy()

        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)

            dists = []
            for m in cluster_model["curves_interp"]:
                m_slice = m[JD_COM <= jd_corte]
                m_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
                dists.append(dtw_distance(obs_norm, m_norm))

            pred = int(np.argmin(dists))
            names = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
            cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}
            
            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(x=JD_COM, y=cluster_model["curves_interp"][pred], name="Patrón Histórico", line=dict(dash='dash', color=cols.get(pred))))
                fp.add_trace(go.Scatter(x=jd_grid, y=obs_norm * cluster_model["curves_interp"][pred].max(), name="2026", line=dict(color='black', width=3)))
                st.plotly_chart(fp, use_container_width=True)
            with c2:
                st.success(f"### {names.get(pred)}")
                st.metric("DTW Score", f"{min(dists):.2f}")
        else:
             st.info("Datos insuficientes para clasificación DTW (Se requiere actividad antes de Mayo).")

    # --- TAB 3: VISUALIZACIÓN DE CURVA (BIO) ---
    with tab3:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        st.markdown(f"Así se comporta la acumulación térmica según los parámetros definidos.")
        
        # Generar datos sintéticos para el gráfico
        x_temps = np.linspace(0, 45, 200)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]
        
        fig_bio = go.Figure()
        
        # Curva Principal
        fig_bio.add_trace(go.Scatter(
            x=x_temps, y=y_tt, mode='lines', name='Acumulación TT',
            line=dict(color='#2563eb', width=4),
            fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)'
        ))
        
        # Zonas
        fig_bio.add_vrect(x0=t_base_val, x1=t_opt_max, fillcolor="green", opacity=0.1, annotation_text="Óptimo", annotation_position="top left")
        fig_bio.add_vrect(x0=t_opt_max, x1=t_critica, fillcolor="orange", opacity=0.1, annotation_text="Estrés (Penalizado)", annotation_position="top right")
        fig_bio.add_vrect(x0=t_critica, x1=45, fillcolor="red", opacity=0.1, annotation_text="Inhibición", annotation_position="top right")
        
        fig_bio.update_layout(
            xaxis_title="Temperatura Media Diaria (°C)",
            yaxis_title="Tiempo Térmico Acumulado (°Cd)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bio, use_container_width=True)
        
        st.info(f"""
        **Interpretación:** * Hasta **{t_base_val}°C**: No pasa nada (Dormición/Inactividad).
        * Entre **{t_base_val}°C y {t_opt_max}°C**: Crecimiento lineal.
        * Entre **{t_opt_max}°C y {t_critica}°C**: La eficiencia cae rápidamente.
        * Más de **{t_critica}°C**: El sistema se detiene (TT = 0).
        """)

    # EXPORTACIÓN
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        pd.DataFrame({
            'Configuracion': ['T_Base', 'T_Optima', 'T_Critica', 'Umbral_Pico'],
            'Valor': [t_base_val, t_opt_max, t_critica, umbral_er]
        }).to_excel(writer, sheet_name='Bio_Params', index=False)
        
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Report.xlsx")

else:
    st.info("👋 **Bienvenido a PREDWEEM.** Cargue datos climáticos para comenzar.")
