# -*- coding: utf-8 -*-
# ===============================================================
# 🚜 PREDWEEM — DASHBOARD DE VALIDACIÓN AGRONÓMICA Y ROI
# Enfoque: Toma de Decisiones a Campo (Bordenave - Shift +60d)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM ROI Agronómico", layout="wide", page_icon="🚜")

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
    .metric-roi { color: #166534; font-weight: bold; font-size: 1.1em;}
    .alert-box {
        background-color: #fef2f2; border-left: 4px solid #ef4444;
        padding: 10px; border-radius: 4px; margin-bottom: 10px;
    }
    .success-box {
        background-color: #f0fdf4; border-left: 4px solid #22c55e;
        padding: 10px; border-radius: 4px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. CLASE DEL MODELO (vK4.4 Corrección Matemática)
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
        st.error(f"Error cargando pesos: {e}")
        return None

# ---------------------------------------------------------
# 3. CARGA DE DATOS
# ---------------------------------------------------------
st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png", use_container_width=True)
st.sidebar.header("⚙️ Configuración Logística")

file_meteo = st.sidebar.file_uploader("1. Clima", type=["csv", "xlsx"])
file_campo = st.sidebar.file_uploader("2. Campo", type=["csv", "xlsx"])

def load_data(file_uploader, default_name):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file_uploader)
    elif (BASE / f"{default_name}.csv").exists():
        return pd.read_csv(BASE / f"{default_name}.csv")
    elif (BASE / f"{default_name}.xlsx").exists():
        return pd.read_excel(BASE / f"{default_name}.xlsx")
    return None

df_meteo_raw = load_data(file_meteo, "bordenave")
df_campo_raw = load_data(file_campo, "bordenave_campo") 

# Parámetros operativos
margen_dias = st.sidebar.number_input("Margen Operativo (Días post-pico para aplicar)", 0, 15, 7)
umbral_alerta = st.sidebar.slider("Umbral de Alerta Temprana (Tasa)", 0.05, 0.50, 0.15)
residualidad = st.sidebar.number_input("Días de residualidad del herbicida", 0, 60, 20)

# ---------------------------------------------------------
# 4. PROCESAMIENTO Y MÉTRICAS AGRONÓMICAS
# ---------------------------------------------------------
modelo = load_model()

if df_meteo_raw is not None and df_campo_raw is not None and modelo is not None:
    
    # 1. Preparar Datos
    df_meteo = df_meteo_raw.copy()
    df_meteo['Fecha'] = pd.to_datetime(df_meteo['Fecha'])
    df_meteo = df_meteo.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df_meteo['Julian_days'] = df_meteo['Fecha'].dt.dayofyear

    df_campo = df_campo_raw.copy()
    col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
    col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
    df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
    df_campo['Campo_Normalizado'] = df_campo[col_plm2] / df_campo[col_plm2].max() if df_campo[col_plm2].max() > 0 else 0

    # 2. Ejecutar Modelo (Shift +60d)
    df_meteo["JD_Shifted"] = (df_meteo["Julian_days"] + 60).clip(1, 300)
    X = df_meteo[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df_meteo["EMERREL"] = np.maximum(modelo.predict(X), 0.0)
    
    df_meteo["Prec_sum_21d"] = df_meteo["Prec"].rolling(window=21, min_periods=1).sum()
    df_meteo["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df_meteo["Prec_sum_21d"] - 15)))
    df_meteo["EMERREL"] = df_meteo["EMERREL"] * df_meteo["Hydric_Factor"]
    
    jd_thresholds = np.where(df_meteo["Prec_sum_21d"] > 50, 0, 15)
    df_meteo.loc[df_meteo["Julian_days"] <= jd_thresholds, "EMERREL"] = 0.0

    # 3. Cálculo de Hitos Temporales
    fecha_pico_modelo = df_meteo.loc[df_meteo['EMERREL'].idxmax(), 'Fecha']
    fecha_pico_campo = df_campo.loc[df_campo[col_plm2].idxmax(), col_fecha]
    fecha_aplicacion = fecha_pico_modelo + timedelta(days=margen_dias)
    fin_residualidad = fecha_aplicacion + timedelta(days=residualidad)
    
    alertas = df_meteo[df_meteo['EMERREL'] >= umbral_alerta]
    fecha_primera_alerta = alertas['Fecha'].iloc[0] if not alertas.empty else fecha_pico_modelo

    # 4. Cálculo de las 4 Métricas Clave
    # A. Peak Lag
    peak_lag = (fecha_pico_modelo - fecha_pico_campo).days
    
    # B. Lead Time (Anticipación)
    lead_time = (fecha_aplicacion - fecha_primera_alerta).days

    # C. PEC (Proporción Controlada - Incluye malezas nacidas hasta fin de residualidad)
    malezas_totales = df_campo[col_plm2].sum()
    malezas_controladas = df_campo.loc[df_campo[col_fecha] <= fin_residualidad, col_plm2].sum()
    pec = (malezas_controladas / malezas_totales) * 100 if malezas_totales > 0 else 0

    # D. Falsos Negativos (Escapes Críticos)
    # Malezas que nacieron cuando el modelo decía riesgo bajo (< 10%) antes de la aplicación
    fechas_bajo_riesgo = df_meteo[df_meteo['EMERREL'] < 0.10]['Fecha'].values
    escapes_antes_app = df_campo[(df_campo[col_fecha].isin(fechas_bajo_riesgo)) & (df_campo[col_fecha] <= fecha_aplicacion)][col_plm2].sum()
    tasa_escapes = (escapes_antes_app / malezas_totales) * 100 if malezas_totales > 0 else 0

    # ---------------------------------------------------------
    # 5. RENDERIZADO DEL DASHBOARD
    # ---------------------------------------------------------
    st.title("🚜 Validación Operativa de PREDWEEM")
    st.markdown("¿Es viable a campo? Evaluación basada en métricas de toma de decisión agronómica.")

    st.markdown("### 🏆 Los 4 Indicadores Clave de Desempeño (KPIs)")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. Control Efectivo (PEC)", f"{pec:.1f}%", f"Meta: >80%", delta_color="normal")
    c2.metric("2. Desfase de Pico (Lag)", f"{abs(peak_lag)} días", "Meta: < 7 días", delta_color="inverse" if abs(peak_lag) > 7 else "normal")
    c3.metric("3. Escapes Críticos", f"{tasa_escapes:.1f}%", "Falsos negativos", delta_color="inverse")
    c4.metric("4. Anticipación (Lead Time)", f"{lead_time} días", "Margen logístico", delta_color="normal")

    st.markdown("---")
    
    col_chart, col_summary = st.columns([2.5, 1])

    with col_chart:
        fig = go.Figure()
        
        # Curva de Riesgo
        fig.add_trace(go.Scatter(
            x=df_meteo['Fecha'], y=df_meteo['EMERREL'], 
            mode='lines', name='Riesgo de Nacimiento (Modelo)', 
            line=dict(color='#166534', width=3), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
        ))

        # Puntos de Campo
        fig.add_trace(go.Scatter(
            x=df_campo[col_fecha], y=df_campo['Campo_Normalizado'], 
            mode='markers+lines', name='Nacimientos Reales (Validación)', 
            marker=dict(color='#dc2626', size=10, symbol='diamond'),
            line=dict(color='rgba(220, 38, 38, 0.4)', dash='dot')
        ))

        # Alerta Temprana
        fig.add_vline(x=fecha_primera_alerta.timestamp() * 1000, line_dash="dash", line_color="orange", annotation_text="Alerta")
        
        # Momento de Aplicación
        fig.add_vline(x=fecha_aplicacion.timestamp() * 1000, line_color="blue", line_width=2, annotation_text="Día de Aplicación")
        
        # Sombra de Residualidad
        fig.add_vrect(
            x0=fecha_aplicacion, x1=fin_residualidad,
            fillcolor="blue", opacity=0.1, layer="below", line_width=0,
            annotation_text=f"Ventana Control ({pec:.0f}%)", annotation_position="top left"
        )

        fig.update_layout(
            title="Línea de Tiempo Operativa", xaxis_title="Fecha", yaxis_title="Tasa Relativa",
            hovermode="x unified", height=450, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_summary:
        st.markdown("### 📋 Diagnóstico Agronómico")
        
        if pec >= 80:
            st.markdown(f"""<div class="success-box"><b>✅ EXCELENTE CONTROL:</b> El momento sugerido cubrió el {pec:.1f}% de los nacimientos.</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="alert-box"><b>⚠️ CONTROL DEFICIENTE:</b> Se lograron controlar solo el {pec:.1f}%. Revisar residualidad.</div>""", unsafe_allow_html=True)
            
        if abs(peak_lag) <= 5:
            st.markdown(f"""<div class="success-box"><b>✅ ALTA PRECISIÓN:</b> El modelo erró el pico exacto por apenas {abs(peak_lag)} días.</div>""", unsafe_allow_html=True)
        
        if lead_time >= 7:
            st.markdown(f"""<div class="success-box"><b>✅ LOGÍSTICA CÓMODA:</b> El productor tuvo {lead_time} días para organizar la fumigación desde que se detectó la subida.</div>""", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 6. EXPORTAR REPORTE EJECUTIVO
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Exportar Reporte Ejecutivo")
    st.write("Genera un Excel con el resumen del ROI y los datos diarios cruzados para entregar al cliente o incluir en la publicación.")
    
    # Crear Excel en memoria
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Hoja 1: Resumen
        resumen_data = {
            'Métrica': ['Control Efectivo (PEC)', 'Desfase de Pico (Lag días)', 'Escapes (Falsos Negativos)', 'Anticipación (Lead Time días)', 'Fecha Aplicación', 'Días Residualidad'],
            'Valor': [f"{pec:.1f}%", peak_lag, f"{tasa_escapes:.1f}%", lead_time, fecha_aplicacion.strftime("%Y-%m-%d"), residualidad]
        }
        pd.DataFrame(resumen_data).to_excel(writer, sheet_name='Resumen_Ejecutivo', index=False)
        
        # Hoja 2: Datos diarios
        df_export = pd.merge(df_meteo[['Fecha', 'Prec', 'EMERREL']], df_campo[[col_fecha, col_plm2]], left_on='Fecha', right_on=col_fecha, how='outer').sort_values('Fecha')
        df_export.to_excel(writer, sheet_name='Datos_Cruzados', index=False)

    st.download_button(
        label="Descargar Reporte de Validación (Excel)",
        data=output.getvalue(),
        file_name="Reporte_PREDWEEM_Operativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.warning("⚠️ Faltan datos o pesos. Carga los archivos a la izquierda.")
