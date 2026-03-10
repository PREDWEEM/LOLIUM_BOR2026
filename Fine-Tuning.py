import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Optimizador de Pesos", layout="wide")

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

    def predict_custom(self, X_raw, lw_custom, blw_custom):
        X_norm = self.normalize(X_raw)
        emer_list = []
        for x in X_norm:
            a1 = np.tanh(self.IW.T @ x + self.bIW)
            z2 = np.dot(lw_custom, a1) + blw_custom
            emer_list.append((np.tanh(z2) + 1) / 2)
        return np.diff(np.cumsum(np.array(emer_list).flatten()), prepend=0)

# ==========================================
# 2. INTERFAZ DE USUARIO
# ==========================================
st.title("🧪 PREDWEEM: Optimizador de Pesos para Nuevos Sitios")
st.markdown("""
Esta herramienta ajusta la **capa de salida** de la Red Neuronal para adaptar el modelo a las 
condiciones específicas de un nuevo lote donde el ajuste original es bajo.
""")

st.sidebar.header("⚙️ Hiperparámetros de Optimización")
max_iter = st.sidebar.number_input("Máximo de Iteraciones", value=50)
ventana_tol = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

# Carga de archivos
col_a, col_b = st.columns(2)
with col_a:
    f_meteo = st.file_uploader("1. Clima del Nuevo Sitio (CSV)", type=['csv'])
with col_b:
    f_valida = st.file_uploader("2. Verdad de Campo del Nuevo Sitio (Excel)", type=['xlsx'])

# ==========================================
# 3. PROCESO DE OPTIMIZACIÓN
# ==========================================
if f_meteo and f_valida:
    try:
        # Carga de datos
        df_m = pd.read_csv(f_meteo)
        df_m.columns = df_m.columns.str.strip()
        df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
        df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
        df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()
        
        df_c = pd.read_excel(f_valida, engine='openpyxl')
        df_c.columns = df_c.columns.str.strip()
        df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
        df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

        # Carga de Pesos Base
        iw, biw = np.load('IW.npy'), np.load('bias_IW.npy')
        lw_init, blw_init = np.load('LW.npy'), np.load('bias_out.npy')
        
        model = PREDWEEM_ANN(iw, biw, lw_init, blw_init)
        X_all = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)

        if st.button("🚀 Iniciar Optimización de Pesos"):
            
            # 
            
            with st.spinner("Calibrando neuronas para el nuevo sitio..."):
                
                def objective(params):
                    new_lw = params[:55].reshape(1, 55)
                    new_blw = params[55]
                    preds = model.predict_custom(X_all, new_lw, new_blw)
                    
                    # Filtros biológicos base
                    preds[(df_m['Prec_sum'] < 15) | (df_m['Julian_days'] <= 10)] = 0.0
                    
                    # Sincronización
                    y_p_adj = []
                    for f_o in df_c['FECHA']:
                        m = (df_m['Fecha'] >= f_o - pd.Timedelta(days=ventana_tol//2)) & \
                            (df_m['Fecha'] <= f_o + pd.Timedelta(days=ventana_tol//2))
                        y_p_adj.append(preds[m].max() if any(m) else 0)
                    
                    return np.mean((df_c['ER_obs'].values - np.array(y_p_adj))**2)

                # Ejecutar scipy.minimize
                x0 = np.append(lw_init.flatten(), blw_init)
                res = minimize(objective, x0, method='BFGS', options={'maxiter': max_iter})
                
                # Extraer resultados
                lw_opt = res.x[:55].reshape(1, 55)
                blw_opt = res.x[55]
                
                st.success(f"✅ Optimización finalizada. MSE final: {res.fun:.4f}")

                # --- COMPARACIÓN VISUAL ---
                # 
                
                preds_old = model.predict_custom(X_all, lw_init, blw_init)
                preds_new = model.predict_custom(X_all, lw_opt, blw_opt)
                
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_m['Fecha'], preds_old, label='Original (Mal ajuste)', color='gray', linestyle='--')
                ax.plot(df_m['Fecha'], preds_new, label='Optimizado (Nuevo Sitio)', color='blue', lw=2)
                ax.scatter(df_c['FECHA'], df_c['ER_obs'], color='red', s=100, label='Campo Real', zorder=5)
                ax.legend()
                st.pyplot(fig)

                # --- DESCARGA DE PESOS ---
                st.subheader("💾 Descargar Nuevos Pesos")
                
                col1, col2 = st.columns(2)
                
                # Guardar en buffer para descarga
                lw_buf = io.BytesIO()
                np.save(lw_buf, lw_opt)
                col1.download_button("Descargar LW_SITIO2.npy", lw_buf.getvalue(), "LW_SITIO2.npy")
                
                blw_buf = io.BytesIO()
                np.save(blw_buf, blw_opt)
                col2.download_button("Descargar bias_out_SITIO2.npy", blw_buf.getvalue(), "bias_out_SITIO2.npy")

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("Sube los datos del nuevo sitio para comenzar el proceso de Fine-Tuning.")
