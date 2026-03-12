
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Validador de Sincronía", layout="wide", page_icon="🌾")

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ==========================================
# 1. FUNCIONES DE MÉTRICAS Y FRECUENCIA
# ==========================================
def categorizar_emergencia(valor):
    if valor >= 0.5:
        return "ALTA"
    elif 0.25 <= valor < 0.5:
        return "INTERMEDIA"
    else:
        return "BAJA/NULA"

def calcular_metricas_completas(df_v):
    if df_v.empty: return 0, 0, 0, 0, 0, [], []
    
    obs = df_v['Obs'].values
    pred = df_v['Pred'].values
    
    if np.std(obs) == 0: return 0, 0, 0, 0, 0, [], []
    
    # 1. NSE (Eficiencia de Nash-Sutcliffe)
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    
    # 2. PBIAS (Sesgo de Volumen)
    pbias = 100 * (np.sum(obs - pred) / np.sum(obs))
    
    # 3. KGE (Kling-Gupta Efficiency - Balance de tendencia)
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else 0
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else 1
    gamma = (np.std(pred)/np.mean(pred)) / (np.std(obs)/np.mean(obs)) if (np.mean(pred) > 0 and np.mean(obs) > 0) else 0
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    # 4. Error de Fase (Sincronía del Pico)
    fecha_pico_obs = df_v.loc[df_v['Obs'].idxmax(), 'Fecha']
    fecha_pico_pred = df_v.loc[df_v['Pred'].idxmax(), 'Fecha']
    desfase_dias = (fecha_pico_pred - fecha_pico_obs).days
    
    # 5. Accuracy por Categoría
    cat_obs = [categorizar_emergencia(v) for v in obs]
    cat_pred = [categorizar_emergencia(v) for v in pred]
    aciertos = sum(1 for o, p in zip(cat_obs, cat_pred) if o == p)
    accuracy_cat = (aciertos / len(obs)) * 100
    
    return nse, kge, pbias, accuracy_cat, desfase_dias, cat_obs, cat_pred

# ==========================================
# 2. ARQUITECTURA DE LA RED NEURONAL (Idéntica al modelo principal)
# ==========================================
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

# ==========================================
# 3. UI Y PROCESAMIENTO
# ==========================================
st.title("🌾 PREDWEEM: Análisis de Sincronía y Validación")

st.sidebar.header("⚙️ Parámetros de Calibración")
umbral_h = st.sidebar.slider("Umbral Hídrico (mm)", 0, 60, 20)
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

f_meteo = st.sidebar.file_uploader("Subir meteo_daily.csv", type=['csv'])
f_valida = st.sidebar.file_uploader("Subir VALIDA.xlsx", type=['xlsx'])

if f_meteo and f_valida:
    try:
        # Cargar datos
        df_clima = pd.read_csv(f_meteo)
        # Adaptar nombres de columnas por robustez
        df_clima.columns = [c.upper().strip() for c in df_clima.columns]
        df_clima = df_clima.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'PREC': 'Prec'})
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida)
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Cargar Pesos de la Red Neuronal
        iw, lw = np.load(BASE / 'IW.npy'), np.load(BASE / 'LW.npy')
        biw, blw = np.load(BASE / 'bias_IW.npy'), np.load(BASE / 'bias_out.npy')

        # Instanciar modelo y predecir
        model = PracticalANNModel(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        emerrel_raw, _ = model.predict(X_input)
        df_clima['EMERREL'] = np.maximum(emerrel_raw, 0.0)
        
        # --- FILTROS (Restricción Hídrica e Histórica) ---
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        df_clima.loc[df_clima['Prec_sum'] < umbral_h, 'EMERREL'] = 0.0
        df_clima.loc[df_clima['Julian_days'] <= 25, 'EMERREL'] = 0.0

        # --- NUEVO: REESCALADO MÁXIMO (Igual al modelo principal) ---
        max_emer = df_clima['EMERREL'].max()
        if max_emer > 0:
            df_clima['EMERREL'] = df_clima['EMERREL'] / max_emer

        # --- Sincronización para métricas ---
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        resultados = []
        radio = ventana_dias // 2

        for _, row in df_campo.iterrows():
            mask = (df_clima['Fecha'] >= row['FECHA'] - pd.Timedelta(days=radio)) & \
                   (df_clima['Fecha'] <= row['FECHA'] + pd.Timedelta(days=radio))
            max_p = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            resultados.append({'Fecha': row['FECHA'], 'Obs': row['ER_obs'], 'Pred': max_p})

        df_v = pd.DataFrame(resultados)
        nse, kge, pbias, acc_cat, desfase, c_obs, c_pred = calcular_metricas_completas(df_v)

        # --- Dashboard de Métricas ---
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("KGE (Ajuste global)", f"{kge:.2f}")
        m2.metric("NSE (Eficiencia)", f"{nse:.2f}")
        m3.metric("Error de Fase", f"{desfase} días")
        m4.metric("Sesgo PBIAS", f"{pbias:.1f}%")
        m5.metric("Acierto Riesgo", f"{acc_cat:.1f}%")

        # --- Gráfico de Validación ---
        st.subheader("Comparativa: Dinámica Simulada vs Observaciones de Campo")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhspan(0.5, 1.1, color='red', alpha=0.07, label='Riesgo Alto (>= 0.5)')
        ax.axhspan(0.15, 0.5, color='orange', alpha=0.07, label='Riesgo Medio (0.15 - 0.5)')
        
        # Curva continua del modelo
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], color='#166534', lw=2, label='Modelo (Diario)')
        
        # Puntos observados en campo
        ax.scatter(df_v['Fecha'], df_v['Obs'], color='black', s=60, zorder=5, label='Campo (Observado)')
        
        # Puntos capturados por la ventana de tolerancia
        ax.scatter(df_v['Fecha'], df_v['Pred'], color='orange', marker='X', s=80, zorder=6, label=f'Modelo (Ventana ±{radio}d)')
        
        ax.set_ylabel("Tasa Relativa de Emergencia")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False)
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

        # --- Análisis de Confusión ---
        with st.expander("📊 Matriz de Confusión: ¿Dónde falla el modelo?"):
            labels = ["BAJA/NULA", "INTERMEDIA", "ALTA"]
            cm = confusion_matrix(c_obs, c_pred, labels=labels)
            fig_cm, ax_cm = plt.subplots(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens', ax=ax_cm)
            ax_cm.set_xlabel('Predicción del Modelo')
            ax_cm.set_ylabel('Realidad del Campo')
            st.pyplot(fig_cm)
            st.info("Esta matriz clasifica el riesgo. Un valor alto en la diagonal indica que el modelo clasifica correctamente el nivel de alerta.")

    except Exception as e:
        st.error(f"Error en el proceso de validación: {e}")

else:
    st.info("👋 **Bienvenido al módulo de Validación.** Por favor, sube los archivos `meteo_daily.csv` y `VALIDA.xlsx` en la barra lateral para comenzar el análisis de sincronía.")
