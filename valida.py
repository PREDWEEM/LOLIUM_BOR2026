# -*- coding: utf-8 -*-
# ===============================================================
# 📊 PREDWEEM — DASHBOARD DE VALIDACIÓN AGRONÓMICA A CAMPO
# Localidad: Bordenave | Lógica: vK4.4 (Shift +60d) | Multipeak > 0.5
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM Validación", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
    .peak-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. CLASE DEL MODELO (Arquitectura Corregida)
# ---------------------------------------------------------
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
        return emerrel

@st.cache_resource
def load_model():
    try:
        ann = PracticalANNModel(
            np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
        )
        return ann
    except Exception as e:
        st.error(f"Error cargando pesos de la red: {e}")
        return None

# ---------------------------------------------------------
# 3. CARGA DE DATOS (Soporte CSV y Excel)
# ---------------------------------------------------------
st.sidebar.header("📂 Carga de Datos")

file_meteo = st.sidebar.file_uploader("1. Clima (bordenave)", type=["csv", "xlsx", "xls"])
file_campo = st.sidebar.file_uploader("2. Campo (bordenave_campo)", type=["csv", "xlsx", "xls"])

def load_data(file_uploader, default_base_name):
    if file_uploader:
        if file_uploader.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_uploader)
        else:
            return pd.read_csv(file_uploader)
    else:
        if (BASE / f"{default_base_name}.csv").exists():
            return pd.read_csv(BASE / f"{default_base_name}.csv")
        elif (BASE / f"{default_base_name}.xlsx").exists():
            return pd.read_excel(BASE / f"{default_base_name}.xlsx")
    return None

df_meteo_raw = load_data(file_meteo, "bordenave")
df_campo_raw = load_data(file_campo, "bordenave_campo") 

# ---------------------------------------------------------
# 4. PROCESAMIENTO Y VALIDACIÓN
# ---------------------------------------------------------
modelo = load_model()

if df_meteo_raw is not None and df_campo_raw is not None and modelo is not None:
    
    # Preparar Clima
    df_meteo = df_meteo_raw.copy()
    df_meteo['Fecha'] = pd.to_datetime(df_meteo['Fecha'])
    df_meteo = df_meteo.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df_meteo['Julian_days'] = df_meteo['Fecha'].dt.dayofyear

    # Preparar Campo
    df_campo = df_campo_raw.copy()
    col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
    col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
    
    df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
    max_plm2 = df_campo[col_plm2].max()
    df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2 if max_plm2 > 0 else 0

    # Ejecutar Modelo (Lógica Bordenave vK4.4)
    df_meteo["JD_Shifted"] = (df_meteo["Julian_days"] + 60).clip(1, 300)
    X = df_meteo[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df_meteo["EMERREL"] = np.maximum(modelo.predict(X), 0.0)
    
    df_meteo["Prec_sum_21d"] = df_meteo["Prec"].rolling(window=21, min_periods=1).sum()
    df_meteo["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df_meteo["Prec_sum_21d"] - 15)))
    df_meteo["EMERREL"] = df_meteo["EMERREL"] * df_meteo["Hydric_Factor"]
    
    jd_thresholds = np.where(df_meteo["Prec_sum_21d"] > 50, 0, 15)
    df_meteo.loc[df_meteo["Julian_days"] <= jd_thresholds, "EMERREL"] = 0.0

    # Cruzar datos para métricas globales
    df_cruce = pd.merge(df_meteo[['Fecha', 'EMERREL']], 
                        df_campo[[col_fecha, col_plm2, 'Campo_Normalizado']], 
                        left_on='Fecha', right_on=col_fecha, how='inner')

    y_sim = df_cruce['EMERREL']
    y_obs = df_cruce['Campo_Normalizado']

    # --- CÁLCULOS ESTADÍSTICOS GLOBALES ---
    pearson_r = y_sim.corr(y_obs)
    rmse = np.sqrt(np.mean((y_sim - y_obs)**2))
    num = np.sum((y_sim - y_obs)**2)
    den = np.sum((np.abs(y_sim - np.mean(y_obs)) + np.abs(y_obs - np.mean(y_obs)))**2)
    willmott_d = 1 - (num / den) if den != 0 else 0

    # --- CÁLCULOS AGRONÓMICOS (MULTIPEAK > 0.5) ---
    margen_dias = st.sidebar.number_input("Margen Operativo (Días post-alerta)", min_value=0, max_value=20, value=7)
    
    # Detección de picos locales > 0.5
    # Condición: EMERREL > 0.5 Y es mayor que el día anterior Y mayor que el día siguiente
    es_pico = (
        (df_meteo['EMERREL'] > 0.5) & 
        (df_meteo['EMERREL'] > df_meteo['EMERREL'].shift(1)) & 
        (df_meteo['EMERREL'] > df_meteo['EMERREL'].shift(-1))
    )
    
    picos_detectados = df_meteo[es_pico].copy()
    
    # Si por alguna razón la curva es plana o no hay picos > 0.5, tomamos el máximo absoluto por seguridad
    if picos_detectados.empty:
        idx_max = df_meteo['EMERREL'].idxmax()
        picos_detectados = df_meteo.loc[[idx_max]]

    malezas_totales = df_campo[col_plm2].sum()
    resultados_aplicaciones = []

    for index, row in picos_detectados.iterrows():
        fecha_pico = row['Fecha']
        tasa_pico = row['EMERREL']
        fecha_app = fecha_pico + timedelta(days=margen_dias)
        
        # Calcular malezas controladas acumuladas hasta esa fecha de aplicación
        malezas_ctrl = df_campo.loc[df_campo[col_fecha] <= fecha_app, col_plm2].sum()
        pec = (malezas_ctrl / malezas_totales) * 100 if malezas_totales > 0 else 0
        
        resultados_aplicaciones.append({
            'fecha_pico': fecha_pico,
            'tasa': tasa_pico,
            'fecha_app': fecha_app,
            'pec': pec
        })

    # ---------------------------------------------------------
    # 5. INTERFAZ GRÁFICA (DASHBOARD)
    # ---------------------------------------------------------
    st.title("📊 Validación de Factibilidad - PREDWEEM")
    st.markdown("Evaluación del modelo frente a observaciones reales de *Lolium spp.*")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<p class='metric-header'>📈 MÉTRICAS ESTADÍSTICAS GLOBALES</p>", unsafe_allow_html=True)
        st.info("Evalúan la similitud matemática entre la curva simulada y la real.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson (r)", f"{pearson_r:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("Willmott (d)", f"{willmott_d:.3f}")

    with col2:
        st.markdown(f"<p class='metric-header'>🚜 EFICIENCIA POR PICOS DE ALERTA (> 0.5)</p>", unsafe_allow_html=True)
        st.caption(f"Margen logístico simulado: {margen_dias} días después de cada alerta.")
        
        # Mostrar los resultados de cada pico detectado
        for i, app in enumerate(resultados_aplicaciones):
            st.markdown(f"""
            <div class="peak-box">
                <b>Pulso #{i+1}</b> detectado el <b>{app['fecha_pico'].strftime('%d/%m/%Y')}</b> (Tasa: {app['tasa']:.2f})<br>
                ➔ Aplicación el {app['fecha_app'].strftime('%d/%m/%Y')} ➔ <b>Control Acumulado (PEC): {app['pec']:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

    # --- GRÁFICO INTERACTIVO ---
    st.markdown("---")
    fig = go.Figure()

    # Curva del Modelo
    fig.add_trace(go.Scatter(
        x=df_meteo['Fecha'], y=df_meteo['EMERREL'], 
        mode='lines', name='Predicción Modelo (0-1)', 
        line=dict(color='#166534', width=3), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
    ))

    # Puntos de Campo
    fig.add_trace(go.Scatter(
        x=df_campo[col_fecha], y=df_campo['Campo_Normalizado'], 
        mode='markers+lines', name='Datos Reales (Normalizados)', 
        marker=dict(color='#dc2626', size=10, symbol='circle', line=dict(color='white', width=2)),
        line=dict(color='rgba(220, 38, 38, 0.3)', width=2, dash='dot')
    ))

    # Líneas de Decisión para CADA pico
    colores_lineas = ['#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899']
    
    for i, app in enumerate(resultados_aplicaciones):
        color = colores_lineas[i % len(colores_lineas)]
        fig.add_vline(
            x=app['fecha_app'].timestamp() * 1000, 
            line_dash="dash", line_color=color, line_width=2.5,
            annotation_text=f"App #{i+1} ({app['pec']:.0f}%)", annotation_position="top left"
        )
        # Marcar también el pico que disparó la alerta
        fig.add_trace(go.Scatter(
            x=[app['fecha_pico']], y=[app['tasa']],
            mode='markers', showlegend=False,
            marker=dict(color=color, size=12, symbol='star', line=dict(color='black', width=1)),
            hoverinfo='text', hovertext=f"Alerta #{i+1}"
        ))

    fig.update_layout(
        title="Dinámica Poblacional y Oportunidades de Control Múltiple",
        xaxis_title="Fecha",
        yaxis_title="Tasa Relativa",
        hovermode="x unified",
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Faltan datos o los archivos de la red neuronal. Asegúrate de cargar el clima y los datos de campo en el panel lateral.")
