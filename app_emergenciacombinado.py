# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM OPERATIVO vK4.9.8 — LOLIUM BORDENAVE 2026
# Actualización:
# - UI: "Datos del Lote" movido a st.expander en la página principal.
# - ADAPTACIÓN BORDENAVE: Coordenadas mantenidas estrictamente en -37.81 según indicación.
# - REVERSIÓN: Eliminada la Memoria Térmica. Restaurado el umbral manual estricto.
# - UNIFICACIÓN MECANÍSTICA 100%: 
#   * Eliminado el forzado empírico de 20 mm.
#   * Eliminada la restricción histórica de 21 días / 50 mm para enero.
# - NUEVO: Bypass de Ruptura de Dormición por Choque Hídrico.
# - NUEVO: Escudo Termofisiológico Dinámico (Media Móvil 10d) para inhibición estival.
# - NUEVO: Corte Hídrico Estricto (20% HR) acoplado a la sigmoide.
# - NUEVO: Bloqueo de emergencia (0%) hasta que una LLUVIA PUNTUAL supere la Capacidad de Campo.
# - NUEVO: Secado exponencial del suelo (Ke Dinámico / Factor Kr) en BHS.
# - Evapotranspiración (ET0) mediante Hargreaves-Samani (Latitud mantenida: -37.81)
# - MEJORA: Sensibilidad térmica e hídrica agresiva según nivel de rastrojo.
# - Gráfico dinámico de retención de agua en suelo vs Lluvias
# - AJUSTE: Umbral de alerta por defecto y salto visual calibrado en 0.30.
# - OPTIMIZACIÓN: Vectorización matricial pura en PracticalANNModel.predict.
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
    page_title="PREDWEEM BORDENAVE vK4.9.8", 
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

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN + DTW + BIO + HÍDRICO)
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
    if t <= t_base:
        return 0.0
    elif t <= t_opt:
        return t - t_base
    elif t < t_crit:
        factor = (t_crit - t) / (t_crit - t_opt)
        return (t - t_base) * factor
    else:
        return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-37.81):
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

# Secado dinámico con factor Kr
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
            github_url = "https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/meteo_daily.csv"
            try:
                df = pd.read_csv(github_url, parse_dates=["Fecha"])
            except Exception:
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

# --- HEADER PRINCIPAL ---
st.title("🌾 PREDWEEM LOLIUM - BORDENAVE 2026")

# --- MENÚ DESPLEGABLE: DATOS DEL LOTE (MAIN PAGE) ---
with st.expander("📂 1. Datos del Lote", expanded=True):
    col_upload, col_rastrojo = st.columns(2)
    
    with col_upload:
        archivo_usuario = st.file_uploader("Subir Clima Manual (BORDENAVE)", type=["xlsx", "csv"])
        df = get_data(archivo_usuario)
        
    with col_rastrojo:
        tipo_manejo = st.selectbox(
            "Nivel de Rastrojo",
            options=[
                "Cobertura Muy Densa (SD - Extra Rastrojo/CS)",
                "Alta Cobertura (SD - Rastrojo Trigo/Maíz)",
                "Cobertura Media (SD - Rastrojo Soja)",
                "Baja Cobertura / Labranza Convencional"
            ],
            index=1 
        )
        
        if "Muy Densa" in tipo_manejo:
            ke_val = 0.10      
            mod_termico = 0.80 
        elif "Alta" in tipo_manejo:
            ke_val = 0.25      
            mod_termico = 0.90 
        elif "Media" in tipo_manejo:
            ke_val = 0.50      
            mod_termico = 0.95 
        else:
            ke_val = 0.95      
            mod_termico = 1.00 
            
        st.caption(f"Coeficiente Ke interno aplicado: **{ke_val:.2f}** | Modulador Térmico Suelo: **{mod_termico:.2f}**")


# --- SIDEBAR ---
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")

umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Detección pico)", 0.05, 0.80, 0.30)

st.sidebar.markdown("**Ruptura de Dormición Estival (Escudo)**")
umbral_termoinhibicion = st.sidebar.number_input(
    "Umbral Termoinhibición (°C)", 
    min_value=15.0, max_value=35.0, value=24.0, step=0.5,
    help="Si la T° Media móvil de los últimos 10 días supera este valor, la emergencia se bloquea a 0%."
)

umbral_choque_hidrico = st.sidebar.slider(
    "Choque Hídrico 3 días (mm)", 
    min_value=20.0, max_value=100.0, value=45.0, 
    help="Desbloquea la emergencia temprana si se acumula esta lluvia antes de fines de abril."
)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2:
    t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)

t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Control", value=600, step=50)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=50)

st.sidebar.divider()
st.sidebar.markdown("## 💧 3. Balance Hídrico (Suelo)")
w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=20.0, step=1.0)


# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO (LÓGICA 100% MECANÍSTICA)
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    
    # --- A. PREPROCESAMIENTO ---
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # --- SIMULACIÓN TÉRMICA DEL SUELO ---
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
    
    df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * mod_termico)
    df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * mod_termico)

    # --- B. PREDICCIÓN NEURAL PURA (Usando la Temperatura del Suelo) ---
    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)
    
    # --- BYPASS AGRONÓMICO: RUPTURA DE DORMICIÓN TEMPRANA ---
    limite_juliano_temprano = 110 # Aprox. 20 de Abril
    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    
    mask_ruptura = (df["Julian_days"] <= limite_juliano_temprano) & (df["Prec_3d"] >= umbral_choque_hidrico)
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 0.75)

    # --- C. RESTRICCIÓN HÍDRICA Y TÉRMICA (MÓDULO MECANÍSTICO BHS) ---
    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-37.81)
    
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo_max=ke_val)
    
    humedad_relativa = df["W_superficial"] / w_max_val
    df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))
    
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]

    # 4. CORTE HÍDRICO ESTRICTO
    df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0

    # 5. TRIGGER DE RECARGA INICIAL (Lluvia puntual)
    df['Lluvia_Recarga'] = (df['Prec'] >= w_max_val).cummax()
    df.loc[~df['Lluvia_Recarga'], "EMERREL"] = 0.0

    # 6. ESCUDO TERMOFISIOLÓGICO DINÁMICO (Bloqueo Estival por T media 10d - del aire)
    df["Tmedia"] = df["Tmedia_aire"]
    df["Tmedia_10d"] = df["Tmedia"].rolling(window=10, min_periods=1).mean()
    mask_inhibicion = df["Tmedia_10d"] >= umbral_termoinhibicion
    df.loc[mask_inhibicion, "EMERREL"] = 0.0
    
    # --- D. CÁLCULO BIO-TÉRMICO (TT) ---
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    
    # --- E. DETECCIÓN DE VENTANA Y ACUMULADOS ---
    fecha_hoy = pd.Timestamp.now().normalize() 
    if fecha_hoy not in df['Fecha'].values:
        fecha_hoy = df['Fecha'].max()
    
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    
    dga_hoy = 0.0
    dga_7dias = 0.0
    fecha_inicio_ventana = None
    msg_estado = "Esperando pico de emergencia..."

    if indices_pulso:
        idx_primer_pico = indices_pulso[0]
        fecha_inicio_ventana = df.loc[idx_primer_pico, "Fecha"]
        
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
        
        mask_hoy = (df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy)
        dga_hoy = df.loc[mask_hoy, "DG"].sum()
        
        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        df_pronostico = df.iloc[idx_hoy + 1 : idx_hoy + 8]
        dga_7dias = dga_hoy + df_pronostico["DG"].sum()
        
        msg_estado = f"Pico detectado el {fecha_inicio_ventana.strftime('%d/%m')}"
        dias_stress = len(df_desde_pico[df_desde_pico["Tmedia"] > t_opt_max])
    
    # -----------------------------------------------------
    # VISUALIZACIÓN FRONT-END
    # -----------------------------------------------------
    colorscale_hard = [[0.0, "green"], [0.29, "green"], [0.30, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
    ))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Riesgo (Tasa Diaria)")
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DE DECISIÓN", "💧 PRECIPITACIONES Y SUELO", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        col_main, col_gauge = st.columns([2, 1])
        dga_actual = 0.0
        dias_stress = 0
        
        if fecha_inicio_ventana:
            df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
            dga_actual = df_ventana["DGA_cum"].iloc[-1] if not df_ventana.empty else 0.0
            dias_stress = len(df_ventana[df_ventana["Tmedia"] > t_opt_max])

        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(
                x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria Simulada',
                line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
            ))
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral Alerta ({umbral_er})")
            fig_emer.update_layout(title="Dinámica de Emergencia y Detección de Picos", height=350, hovermode="x unified")
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico detectado)")
                if dias_stress > 0:
                    st.markdown(f"""<div class="bio-alert">🔥 <b>Estrés Térmico:</b> {dias_stress} días con T > {t_opt_max}°C desde el inicio.</div>""", unsafe_allow_html=True)
            else:
                st.warning(f"⏳ Esperando primera alerta (Tasa diaria >= {umbral_er}).")

        with col_gauge:
            max_axis = dga_critico * 1.2
            fig_gauge = go.Figure()
            fig_gauge.add_trace(go.Indicator(
                mode = "gauge+number", 
                value = dga_hoy,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"<b>TT ACUMULADO (°Cd)</b>", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, max_axis]},
                    'bar': {'color': "#1e293b", 'thickness': 0.3},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#4ade80"},
                        {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                        {'range': [dga_critico, max_axis], 'color': "#f87171"}
                    ],
                    'threshold': {
                        'line': {'color': "#2563eb", 'width': 6},
                        'thickness': 0.8,
                        'value': dga_7dias
                    }
                }
            ))
            fig_gauge.add_annotation(
                x=0.5, y=-0.1,
                text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>",
                showarrow=False, font=dict(size=14, color="#1e3a8a"), align="center"
            )
            fig_gauge.update_layout(height=350, margin=dict(t=80, b=50, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.header("💧 Dinámica Hídrica del Suelo (Balance Superficial)")
        st.markdown("Visualización de las precipitaciones frente a la retención de agua en los primeros centímetros del suelo, considerando la evapotranspiración (ET0).")
        
        fig_hidrico = go.Figure()
        
        fig_hidrico.add_trace(go.Bar(
            x=df["Fecha"], 
            y=df["Prec"], 
            name='Lluvia Diaria (mm)', 
            marker_color='#93c5fd', 
            opacity=0.7
        ))
        
        fig_hidrico.add_trace(go.Scatter(
            x=df["Fecha"], 
            y=df["W_superficial"], 
            name='Agua en Suelo (0-10cm)', 
            mode='lines',
            line=dict(color='#0284c7', width=3),
            fill='tozeroy',
            fillcolor='rgba(2, 132, 199, 0.2)'
        ))

        fig_hidrico.add_hline(
            y=w_max_val, 
            line_dash="dot", 
            line_color="#334155", 
            annotation_text=f"Capacidad Máx. ({w_max_val} mm)", 
            annotation_position="top left"
        )

        fig_hidrico.update_layout(
            title="Precipitación vs. Retención Real de Humedad", 
            xaxis_title="Fecha", 
            yaxis_title="Milímetros (mm)", 
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
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
             st.info("Datos insuficientes para clasificación DTW.")

    with tab4:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]
        fig_bio = go.Figure()
        fig_bio.add_trace(go.Scatter(x=x_temps, y=y_tt, mode='lines', line=dict(color='#2563eb', width=4), fill='tozeroy'))
        st.plotly_chart(fig_bio, use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        pd.DataFrame({'Configuracion': ['T_Base', 'T_Optima', 'T_Critica', 'W_Max', 'Ke', 'Mod_Termico', 'Umbral_Termoinhibicion'], 'Valor': [t_base_val, t_opt_max, t_critica, w_max_val, ke_val, mod_termico, umbral_termoinhibicion]}).to_excel(writer, sheet_name='Bio_Params', index=False)
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Operativo_Bordenave_vK4_9_8.xlsx")

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue datos meteorológicos para comenzar.")
