import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Calibrador Dinámico", layout="wide")

# ==========================================
# 1. ARQUITECTURA DE LA RED NEURONAL
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
        emer_list = []
        for x in X_norm:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = np.dot(self.LW, a1) + self.bLW
            emer_list.append((np.tanh(z2) + 1) / 2)
        emer_cum = np.cumsum(np.array(emer_list).flatten())
        return np.diff(emer_cum, prepend=0)

# ==========================================
# 2. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🌾 PREDWEEM: Calibración y Validación de Campo")
st.markdown("Ajusta los parámetros en el panel lateral para optimizar las métricas de eficiencia (NSE).")

# --- SIDEBAR: CONTROLES DE AJUSTE ---
st.sidebar.header("⚙️ Parámetros de Calibración")

# Control 1: Umbral Hídrico (Lo que pediste añadir)
umbral_h = st.sidebar.slider(
    "Umbral Hídrico Acumulado (mm)", 
    min_value=0, max_value=50, value=20, 
    help="Cantidad mínima de lluvia en 21 días para activar la emergencia."
)

# Control 2: Ventana de Tolerancia
ventana_dias = st.sidebar.slider(
    "Ventana de Tolerancia (Días)", 
    1, 14, 7, 
    help="Compensa el desfase entre el pico real y el día de muestreo semanal."
)

st.subheader("📂 Carga de Datos")
col_a, col_b = st.columns(2)
with col_a:
    f_meteo = st.file_uploader("Subir meteo_daily.csv", type=['csv'])
with col_b:
    f_valida = st.file_uploader("Subir VALIDA.xlsx", type=['xlsx'])

# ==========================================
# 3. MOTOR DE CÁLCULO Y VALIDACIÓN
# ==========================================
if f_meteo and f_valida:
    try:
        # Carga de archivos
        df_clima = pd.read_csv(f_meteo)
        df_clima.columns = df_clima.columns.str.strip()
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida, engine='openpyxl')
        df_campo.columns = df_campo.columns.str.strip()
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Carga de Pesos
        iw, lw = np.load('IW.npy'), np.load('LW.npy')
        biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')

        # Predicción Base
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = np.maximum(model.predict(X_input), 0.0)

        # APLICACIÓN DEL UMBRAL HÍDRICO DINÁMICO
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        # Filtro: Si la lluvia acumulada es menor al umbral del slider, la emergencia es 0
        df_clima.loc[(df_clima['Prec_sum'] < umbral_h) | (df_clima['Julian_days'] <= 25), 'EMERREL'] = 0.0

        # Lógica de Ventana de Tolerancia
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        resultados_adj = []
        mitad = ventana_dias // 2

        for _, row in df_campo.iterrows():
            f_obs, v_obs = row['FECHA'], row['ER_obs']
            mask = (df_clima['Fecha'] >= f_obs - pd.Timedelta(days=mitad)) & \
                   (df_clima['Fecha'] <= f_obs + pd.Timedelta(days=mitad))
            max_p = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            resultados_adj.append({'Fecha': f_obs, 'Obs': v_obs, 'Pred_Adj': max_p})

        df_v = pd.DataFrame(resultados_adj)

        # Métricas
        y_o, y_p = df_v['Obs'].values, df_v['Pred_Adj'].values
        rmse = np.sqrt(np.mean((y_o - y_p)**2))
        nse = 1 - (np.sum((y_o - y_p)**2) / np.sum((y_o - np.mean(y_o))**2))

        # ==========================================
        # 4. DASHBOARD DE RESULTADOS
        # ==========================================
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE (Error)", f"{rmse:.3f}")
        m2.metric("NSE (Eficiencia)", f"{nse:.3f}")
        m3.metric("Umbral Activo", f"{umbral_h} mm")

        if nse > 0.5:
            st.success("✅ El modelo tiene una capacidad predictiva satisfactoria.")
        else:
            st.warning("⚠️ El modelo requiere más ajuste. Prueba bajar el umbral hídrico.")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], label='Modelo (Simulado)', color='#1b5e20', lw=2)
        ax.fill_between(df_clima['Fecha'], 0, df_clima['EMERREL'], color='#1b5e20', alpha=0.1)
        ax.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='#b71c1c', s=120, label='Campo (Real)', zorder=5)
        ax.set_ylabel("Emergencia Relativa")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Sube los archivos para comenzar la calibración del umbral.")
