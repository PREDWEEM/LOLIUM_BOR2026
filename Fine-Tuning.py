import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Optimizador Global", layout="wide")

# ==========================================
# 1. MOTOR DE LA RED NEURONAL (RNA)
# ==========================================
def run_ann(X_input, iw, biw, lw, blw):
    # Rangos de normalización originales del proyecto
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
    
    return np.diff(np.cumsum(np.array(emer_rel).flatten()), prepend=0)

# ==========================================
# 2. INTERFAZ DE USUARIO (ST)
# ==========================================
st.title("🧠 PREDWEEM: Optimizador Global de Parámetros")
st.markdown("""
Esta herramienta ajusta los **331 pesos y desvíos** de la red para que el modelo 
simule correctamente la emergencia en sitios con dinámicas atípicas.
""")

st.sidebar.header("⚙️ Hiperparámetros de Ajuste")
max_iter = st.sidebar.slider("Máximo de Iteraciones", 10, 200, 50)
u_hidrico = st.sidebar.slider("Umbral Hídrico (mm)", 5, 30, 15)
d_inicio = st.sidebar.slider("Día de Inicio (Julian)", 1, 31, 10)

# --- CARGA DE ARCHIVOS (RESUELVE EL FILENOTFOUNDERROR) ---
st.subheader("📂 Carga de Datos del Nuevo Sitio")
col_a, col_b = st.columns(2)

with col_a:
    f_meteo = st.file_uploader("1. Subir Clima (meteo_daily (2).csv)", type=['csv'])
with col_b:
    f_valida = st.file_uploader("2. Subir Campo (VALIDA (1).xlsx)", type=['xlsx', 'csv'])

# ==========================================
# 3. LÓGICA DE OPTIMIZACIÓN
# ==========================================
if f_meteo and f_valida:
    try:
        # Cargar clima desde el objeto del uploader
        df_m = pd.read_csv(f_meteo)
        df_m.columns = df_m.columns.str.strip()
        df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
        df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
        df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()
        
        # Cargar campo (manejo de XLSX o CSV)
        if f_valida.name.endswith('.csv'):
            df_c = pd.read_csv(f_valida)
        else:
            df_c = pd.read_excel(f_valida, engine='openpyxl')
        
        df_c.columns = df_c.columns.str.strip()
        df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
        df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

        # Cargar Pesos Base (Deben estar en el repo de GitHub)
        IW_orig = np.load('IW.npy')
        bIW_orig = np.load('bias_IW.npy')
        LW_orig = np.load('LW.npy')
        bLW_orig = np.load('bias_out.npy')

        X_matrix = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)

        if st.button("🔥 Ejecutar Calibración Global"):
            with st.spinner("Optimizando 331 parámetros neuronales..."):
                
                def objective(params):
                    # Reconstruir la arquitectura de la RNA
                    ptr = 0
                    niw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
                    nbiw = params[ptr:ptr+55]; ptr += 55
                    nlw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
                    nblw = params[ptr]
                    
                    preds = run_ann(X_matrix, niw, nbiw, nlw, nblw)
                    
                    # Filtros de sitio
                    preds[(df_m['Prec_sum'] < u_hidrico) | (df_m['Julian_days'] <= d_inicio)] = 0.0
                    
                    # Sincronización (Ventana 7 días)
                    y_p_adj = []
                    for f_o in df_c['FECHA']:
                        mask = (df_m['Fecha'] >= f_o - pd.Timedelta(days=3)) & \
                               (df_m['Fecha'] <= f_o + pd.Timedelta(days=3))
                        y_p_adj.append(preds[mask].max() if any(mask) else 0)
                    
                    # Error cuadrático medio
                    return np.mean((df_c['ER_obs'].values - np.array(y_p_adj))**2)

                # Punto de partida (pesos originales)
                x0 = np.concatenate([IW_orig.flatten(), bIW_orig, LW_orig.flatten(), [bLW_orig]])
                
                # Optimización L-BFGS-B (ideal para muchos parámetros)
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter})
                
                # Reconstruir resultados optimizados
                ptr = 0
                opt_iw = res.x[ptr:ptr+220].reshape(4, 55); ptr += 220
                opt_biw = res.x[ptr:ptr+55]; ptr += 55
                opt_lw = res.x[ptr:ptr+55].reshape(1, 55); ptr += 55
                opt_blw = res.x[ptr]

                st.success(f"✅ Optimización Terminada. MSE: {res.fun:.5f}")

                # --- VISUALIZACIÓN ---
                p_final = run_ann(X_matrix, opt_iw, opt_biw, opt_lw, opt_blw)
                p_final[(df_m['Prec_sum'] < u_hidrico) | (df_m['Julian_days'] <= d_inicio)] = 0.0
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_m['Fecha'], p_final, label='Modelo Calibrado', color='#1e3a8a', lw=2)
                ax.scatter(df_c['FECHA'], df_c['ER_obs'], color='#b91c1c', s=100, label='Verdad de Campo', zorder=5)
                ax.set_ylabel("Emergencia Relativa")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # --- DESCARGA ---
                st.subheader("💾 Guardar Nuevos Pesos")
                c1, c2, c3, c4 = st.columns(4)
                
                def get_btn(label, data, name):
                    buf = io.BytesIO()
                    np.save(buf, data)
                    return st.download_button(label, buf.getvalue(), name)

                with c1: get_btn("IW.npy", opt_iw, "IW_global.npy")
                with c2: get_btn("bias_IW.npy", opt_biw, "bias_IW_global.npy")
                with c3: get_btn("LW.npy", opt_lw, "LW_global.npy")
                with c4: get_btn("bias_out.npy", opt_blw, "bias_out_global.npy")

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
else:
    st.info("Sube los archivos meteorológicos y de campo para comenzar el Fine-Tuning.")
