
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Validador de Sincronía", layout="wide", page_icon="🌾")

# ==========================================
# 1. FUNCIONES DE MÉTRICAS Y FRECUENCIA
# ==========================================
def categorizar_emergencia(valor):
    if valor >= 0.5:
        return "ALTA"
    elif 0.15 <= valor < 0.5:
        return "INTERMEDIA"
    else:
        return "BAJA/NULA"

def calcular_metricas_completas(df_v):
    obs = df_v['Obs'].values
    pred = df_v['Pred'].values
    
    if np.std(obs) == 0: return 0, 0, 0, 0, 0, [], []
    
    # 1. NSE (Eficiencia de Nash-Sutcliffe)
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    
    # 2. PBIAS (Sesgo de Volumen)
    pbias = 100 * (np.sum(obs - pred) / np.sum(obs))
    
    # 3. KGE (Kling-Gupta Efficiency - Balance de tendencia)
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else 0
    beta = np.mean(pred) / np.mean(obs)
    gamma = (np.std(pred)/np.mean(pred)) / (np.std(obs)/np.mean(obs)) if np.mean(pred) > 0 else 0
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
# 2. ARQUITECTURA DE LA RED NEURONAL
# ==========================================
class PREDWEEM_ANN:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, X_raw):
        X_norm = self.normalize(X_raw)
        z1 = X_norm @ self.IW + self.bIW.T
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T) + self.bLW
        emer = (np.tanh(z2) + 1) / 2
        return np.maximum(emer.flatten(), 0)

# ==========================================
# 3. UI Y PROCESAMIENTO
# ==========================================
st.title("🌾 PREDWEEM: Análisis de Sincronía y Frecuencia")

st.sidebar.header("⚙️ Parámetros de Calibración")
umbral_h = st.sidebar.slider("Umbral Hídrico (mm)", 0, 60, 20)
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

f_meteo = st.file_uploader("Subir meteo_daily.csv", type=['csv'])
f_valida = st.file_uploader("Subir VALIDA.xlsx", type=['xlsx'])

if f_meteo and f_valida:
    try:
        df_clima = pd.read_csv(f_meteo)
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida)
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Cargar Pesos
        iw, lw = np.load('IW.npy'), np.load('LW.npy')
        biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')

        # Predicción
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = model.predict(X_input)
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        
        # Filtros
        df_clima.loc[df_clima['Prec_sum'] < umbral_h, 'EMERREL'] = 0.0
        df_clima.loc[df_clima['Julian_days'] <= 25, 'EMERREL'] = 0.0

        # Sincronización para métricas
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

        # Dashboard de Métricas
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("KGE (Ajuste)", f"{kge:.2f}")
        m2.metric("NSE (Eficiencia)", f"{nse:.2f}")
        m3.metric("Error de Fase", f"{desfase} días")
        m4.metric("Sesgo PBIAS", f"{pbias:.1f}%")
        m5.metric("Acierto Riesgo", f"{acc_cat:.1f}%")

        # Gráfico de Validación
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhspan(0.5, 1.1, color='red', alpha=0.07, label='Riesgo Alto')
        ax.axhspan(0.15, 0.5, color='orange', alpha=0.07, label='Riesgo Medio')
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], color='green', lw=1.5, label='Modelo')
        ax.scatter(df_v['Fecha'], df_v['Obs'], color='black', label='Campo')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
        st.pyplot(fig)

        # Análisis de Confusión
        with st.expander("📊 Matriz de Confusión: ¿Dónde falla el modelo?"):
            labels = ["BAJA/NULA", "INTERMEDIA", "ALTA"]
            cm = confusion_matrix(c_obs, c_pred, labels=labels)
            fig_cm, ax_cm = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='YlGn', ax=ax_cm)
            ax_cm.set_xlabel('Predicción del Modelo')
            ax_cm.set_ylabel('Realidad del Campo')
            st.pyplot(fig_cm)
            st.info("Esta matriz indica si el modelo tiende a dar falsas alarmas o si 'ignora' emergencias reales.")

    except Exception as e:
        st.error(f"Error en el proceso: {e}")
