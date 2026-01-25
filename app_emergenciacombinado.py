# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK3 — LOLIUM TRES ARROYOS 2026
# Fusión: Monitor de Ventana (App A) + Análisis de Patrones (App B)
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
    page_title="PREDWEEM INTEGRAL vK3", 
    layout="wide",
    page_icon="🌾"
)

# CSS para limpiar la interfaz y dar estilo
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ: GENERADOR DE ARCHIVOS MOCK (ANTI-CRASH)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    """Genera archivos base si no existen para evitar crash en primera ejecución."""
    # Archivos de la Red Neuronal (ANN)
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    # Archivo del Modelo de Clusters (Patrones)
    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        # Crear 3 patrones sintéticos matemáticos
        p1 = np.exp(-((jd - 100)**2)/600)  # Temprano
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200) # Bimodal
        p3 = np.exp(-((jd - 230)**2)/1500) # Tardío
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

    # Archivo de Clima Default
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
# 3. LÓGICA TÉCNICA (ANN + DTW)
# ---------------------------------------------------------
def dtw_distance(a, b):
    """Cálculo de distancia Dynamic Time Warping para comparar curvas."""
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
    """Carga ANN y Cluster Model."""
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
    """Procesa input de usuario o default."""
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

# Logo (si existe URL online o local, usamos la URL del repo del código A)
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Alerta**")
umbral_er = st.sidebar.slider("Umbral Emergencia Diaria", 0.05, 0.80, 0.50)

st.sidebar.markdown("**Ventana Térmica (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Óptimo", value=600, step=50)
dga_critico = st.sidebar.number_input("Límite Crítico", value=700, step=50)

st.sidebar.caption("PREDWEEM vK3 | Tres Arroyos 2026")

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    
    # A. Preprocesamiento Básico
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # B. Predicción Neural
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    
    # Limpieza
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 30, "EMERREL"] = 0.0 # Filtro biológico inicial
    
    # C. Cálculo de Grados Día (Base 2.0°C)
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0)
    
    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM-BORDENAVE 2026")

    # 1. HEATMAP (Diagnóstico Rápido)
    colorscale_hard = [
        [0.00, "green"], [0.49, "green"],  
        [0.49, "yellow"], [0.90, "yellow"], 
        [0.90, "red"], [1.00, "red"]     
    ]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False,
        hovertemplate="<b>%{x|%d-%b}</b><br>Tasa: %{z:.3f}<extra></extra>"
    ))
    fig_risk.update_layout(height=130, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Intensidad de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True)

    # TABS PARA ORGANIZAR LA INFORMACIÓN
    tab1, tab2 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS ESTRATÉGICO"])

    # --- TAB 1: MONITOR DE VENTANA (Lógica de App A) ---
    with tab1:
        col_main, col_gauge = st.columns([2, 1])
        
        # Detectar inicio de ventana (Primer pulso >= umbral)
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        fecha_inicio_ventana = None
        for i in range(len(indices_pulso) - 1):
            if (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days <= 5:
                fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
                break
        
        # Calcular Acumulados si hay ventana
        dga_actual = 0.0
        df_ventana = pd.DataFrame()
        if fecha_inicio_ventana:
            df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
            dga_actual = df_ventana["DGA_cum"].iloc[-1]

        with col_main:
            # Serie de Tiempo
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(
                x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria',
                line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
            ))
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral {umbral_er}")
            fig_emer.update_layout(title="Dinámica de Emergencia", height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.info(f"📅 **Inicio de Cohorte Detectado:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Acumulando Grados Día desde entonces)")
            else:
                st.warning("⏳ Esperando pulsos de emergencia significativos para iniciar conteo térmico.")

        with col_gauge:
            # Semáforo
            max_axis = dga_critico * 1.2
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta", value = dga_actual,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<b>ACUMULACIÓN TÉRMICA</b><br><span style='font-size:0.8em;color:gray'>Grados Días (°Cd)</span>"},
                delta = {'reference': dga_optimo, 'increasing': {'color': "gray"}},
                gauge = {
                    'axis': {'range': [None, max_axis]},
                    'bar': {'color': "black", 'thickness': 0.05},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#4ade80"},
                        {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                        {'range': [dga_critico, max_axis], 'color': "#f87171"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': dga_actual}
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Tabla Resumen Estado
            estado_texto = "ESPERA"
            if fecha_inicio_ventana:
                if dga_actual <= dga_optimo: estado_texto = "✅ VENTANA ÓPTIMA"
                elif dga_actual <= dga_critico: estado_texto = "⚠️ ALERTA AMARILLA"
                else: estado_texto = "🚫 FUERA DE VENTANA"
            st.metric("Estado Fenológico", estado_texto)

    # --- TAB 2: ANÁLISIS DE PATRONES (Lógica de App B) ---
    with tab2:
        st.header("🔍 Clasificación de Patrones (DTW)")
        st.markdown("Comparativa de la curva acumulada actual vs. Patrones históricos del Sur de Buenos Aires.")
        
        fecha_corte_analisis = pd.Timestamp("2026-05-01")
        df_obs = df[df["Fecha"] < fecha_corte_analisis].copy()

        if df_obs.empty:
            st.info("Datos insuficientes para análisis de patrones (Se requiere data previa a Mayo).")
        else:
            # 1. Preparar Curva Observada
            jd_corte = df_obs["Julian_days"].max()
            max_e_obs = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            
            JD_COMMON = cluster_model["JD_common"]
            jd_obs_grid = JD_COMMON[JD_COMMON <= jd_corte]
            # Interpolamos observaciones a la grilla común
            curva_obs_norm = np.interp(jd_obs_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e_obs)

            # 2. Calcular Distancias DTW contra Medoides
            dists = []
            meds = cluster_model["curves_interp"]
            for m in meds:
                m_slice = m[JD_COMMON <= jd_corte]
                m_slice_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
                dists.append(dtw_distance(curva_obs_norm, m_slice_norm))

            # 3. Resultado
            cluster_pred = int(np.argmin(dists))
            nombres = {0: "🌾 Intermedio / Bimodal", 1: "🌱 Temprano / Compacto", 2: "🍂 Tardío / Extendido"}
            colores = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}
            
            nombre_final = nombres.get(cluster_pred, f"Desconocido ({cluster_pred})")
            color_final = colores.get(cluster_pred, "gray")

            c1, c2 = st.columns([3, 1])
            with c1:
                fig_p = go.Figure()
                # Histórico
                fig_p.add_trace(go.Scatter(
                    x=JD_COMMON, y=meds[cluster_pred], mode='lines',
                    line=dict(color=color_final, width=2, dash='dash'),
                    name=f"Patrón Histórico: {nombre_final}"
                ))
                # Actual
                factor_escala = meds[cluster_pred].max() if meds[cluster_pred].max() > 0 else 1
                fig_p.add_trace(go.Scatter(
                    x=jd_obs_grid, y=curva_obs_norm * factor_escala, mode='lines',
                    line=dict(color='black', width=3), name="Campaña 2026"
                ))
                fig_p.add_vline(x=jd_corte, line_dash="dot", line_color="red", annotation_text="Hoy")
                fig_p.update_layout(title="Ajuste DTW: Observado vs Proyectado", xaxis_title="Día Juliano", height=350)
                st.plotly_chart(fig_p, use_container_width=True)
            
            with c2:
                st.success(f"### {nombre_final}")
                st.metric("Similitud (DTW)", f"{min(dists):.2f}", delta_color="inverse")
                st.caption("Menor valor indica mayor ajuste al patrón histórico.")

    # -----------------------------------------------------
    # EXPORTACIÓN
    # -----------------------------------------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        pd.DataFrame({
            'Parametro': ['Umbral Alerta', 'Optimo Termico', 'Critico Termico', 'Patron Detectado'],
            'Valor': [umbral_er, dga_optimo, dga_critico, nombres.get(cluster_pred, "N/A") if 'cluster_pred' in locals() else "N/A"]
        }).to_excel(writer, sheet_name='Reporte', index=False)
        
    st.sidebar.download_button(
        label="📥 Descargar Reporte Completo",
        data=output.getvalue(),
        file_name="PREDWEEM_Full_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👋 **Bienvenido a PREDWEEM.** El sistema está listo. Cargue datos climáticos para comenzar la simulación.")
