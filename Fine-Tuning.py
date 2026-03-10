import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Optimización Global", layout="wide")

# ==========================================
# 1. MOTOR DE LA RED NEURONAL (LÓGICA DE PESOS)
# ==========================================
def run_ann(X_input, iw, biw, lw, blw):
    # Normalización manual del modelo PREDWEEM
    in_min = np.array([1, 0, -7, 0])
    in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
    
    emer_rel = []
    for x in X_n:
        # Capa Oculta
        z1 = iw.T @ x + biw
        a1 = np.tanh(z1)
        # Capa de Salida
        z2 = np.dot(lw, a1) + blw
        emer_rel.append((np.tanh(z2) + 1) / 2)
    
    # Tasa diaria (Diferencial del acumulado)
    return np.diff(np.cumsum(np.array(emer_rel).flatten()), prepend=0)

# ==========================================
# 2. INTERFAZ DE USUARIO
# ==========================================
st.title("🧠 PREDWEEM: Re-entrenamiento Global de la Red")
st.warning("⚠️ Estás por optimizar los 331 parámetros del modelo. Esto ajustará la inteligencia del modelo al nuevo sitio.")

# Carga de archivos
col_a, col_b = st.columns(2)
with col_a:
    f_meteo = st.file_uploader("1. Clima del Sitio (CSV)", type=['csv'])
with col_b:
    f_valida = st.file_uploader("2. Campo del Sitio (Excel/CSV)", type=['xlsx', 'csv'])

# ==========================================
# 3. LÓGICA DE OPTIMIZACIÓN GLOBAL
# ==========================================
if f_meteo and f_valida:
    try:
        # Procesar Clima
        df_m = pd.read_csv(f_meteo)
        df_m.columns = df_m.columns.str.strip()
        df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
        df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
        df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()
        
        # Procesar Campo
        if f_valida.name.endswith('.csv'):
            df_c = pd.read_csv(f_valida)
        else:
            df_c = pd.read_excel(f_valida, engine='openpyxl')
        df_c.columns = df_c.columns.str.strip()
        df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
        df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

        # Cargar Pesos Originales como Semilla
        IW_orig = np.load('IW.npy')        # (4, 55)
        bIW_orig = np.load('bias_IW.npy')  # (55,)
        LW_orig = np.load('LW.npy')        # (1, 55)
        bLW_orig = np.load('bias_out.npy') # (SCALAR)

        X_matrix = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)

        if st.button("🔥 Iniciar Optimización Global (331 Parámetros)"):
            with st.spinner("Ejecutando algoritmos de optimización..."):
                
                # Función para aplanar y reconstruir pesos
                def objective(params):
                    # Reconstrucción del vector plano a matrices
                    ptr = 0
                    new_iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
                    new_biw = params[ptr:ptr+55]; ptr += 55
                    new_lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
                    new_blw = params[ptr]
                    
                    # Ejecutar modelo
                    preds = run_ann(X_matrix, new_iw, new_biw, new_lw, new_blw)
                    
                    # Filtros Biológicos (Se mantienen para coherencia del proyecto)
                    preds[(df_m['Prec_sum'] < 10) | (df_m['Julian_days'] <= 5)] = 0.0
                    
                    # Sincronización (Ventana 7 días)
                    y_p_adj = []
                    for f_o in df_c['FECHA']:
                        mask = (df_m['Fecha'] >= f_o - pd.Timedelta(days=3)) & \
                               (df_m['Fecha'] <= f_o + pd.Timedelta(days=3))
                        y_p_adj.append(preds[mask].max() if any(mask) else 0)
                    
                    # Error (MSE)
                    return np.mean((df_c['ER_obs'].values - np.array(y_p_adj))**2)

                # Vector inicial de 331 parámetros
                x0 = np.concatenate([IW_orig.flatten(), bIW_orig, LW_orig.flatten(), [bLW_orig]])
                
                # Optimización (Limitamos iteraciones para evitar degradación total)
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 30})
                
                # Reconstruir los pesos optimizados
                ptr = 0
                opt_iw = res.x[ptr:ptr+220].reshape(4, 55); ptr += 220
                opt_biw = res.x[ptr:ptr+55]; ptr += 55
                opt_lw = res.x[ptr:ptr+55].reshape(1, 55); ptr += 55
                opt_blw = res.x[ptr]

                st.success(f"✅ Calibración terminada. Error final: {res.fun:.4f}")

                # Gráfico de Resultados
                p_final = run_ann(X_matrix, opt_iw, opt_biw, opt_lw, opt_blw)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_m['Fecha'], p_final, label='Modelo Optimizado', color='blue')
                ax.scatter(df_c['FECHA'], df_c['ER_obs'], color='red', label='Campo')
                ax.legend()
                st.pyplot(fig)

                # DESCARGA DE PESOS OPTIMIZADOS
                st.subheader("📦 Descargar Pack de Pesos Calibrados")
                c1, c2, c3, c4 = st.columns(4)
                
                # Helper para descargar .npy
                def get_btn(label, data, name):
                    buf = io.BytesIO()
                    np.save(buf, data)
                    return st.download_button(label, buf.getvalue(), name)

                with c1: get_btn("IW.npy", opt_iw, "IW_global.npy")
                with c2: get_btn("bias_IW.npy", opt_biw, "bias_IW_global.npy")
                with c3: get_btn("LW.npy", opt_lw, "LW_global.npy")
                with c4: get_btn("bias_out.npy", opt_blw, "bias_out_global.npy")

    except Exception as e:
        st.error(f"Error en la optimización: {e}")
