import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Optimizador Global", layout="wide")

# ==========================================
# 1. ARQUITECTURA DE LA RED NEURONAL (LÓGICA DE PESOS)
# ==========================================
def run_ann(X_input, iw, biw, lw, blw):
    # Rangos de normalización originales del proyecto
    in_min = np.array([1, 0, -7, 0])
    in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
    
    emer_rel = []
    for x in X_n:
        # Capa Oculta (4 neuronas entrada -> 55 neuronas ocultas)
        z1 = iw.T @ x + biw
        a1 = np.tanh(z1)
        # Capa de Salida (55 ocultas -> 1 salida)
        z2 = np.dot(lw, a1) + blw
        emer_rel.append((np.tanh(z2) + 1) / 2)
    
    # Cálculo de tasa diaria a partir del acumulado
    return np.diff(np.cumsum(np.array(emer_rel).flatten()), prepend=0)

# ==========================================
# 2. INTERFAZ DE USUARIO
# ==========================================
st.title("🧠 PREDWEEM: Re-entrenamiento Global de la Red")
st.warning("⚠️ Esta herramienta optimiza los 331 parámetros del modelo. Recomendado para sitios con dinámicas muy diferentes al original.")

st.sidebar.header("⚙️ Hiperparámetros")
max_iter = st.sidebar.number_input("Máximo de Iteraciones", value=100)
u_hidrico = st.sidebar.slider("Umbral Hídrico Sugerido (mm)", 5, 30, 15)
d_inicio = st.sidebar.slider("Día de Inicio Sugerido (Julian)", 1, 31, 10)

# Carga de archivos
col_a, col_b = st.columns(2)
with col_a:
    f_meteo = st.file_uploader("1. Clima del Nuevo Sitio (CSV)", type=['csv'])
with col_b:
    f_valida = st.file_uploader("2. Campo del Nuevo Sitio (Excel/CSV)", type=['xlsx', 'csv'])

# ==========================================
# 3. LÓGICA DE OPTIMIZACIÓN
# ==========================================
if f_meteo and f_valida:
    try:
        # Procesamiento de Clima
        df_m = pd.read_csv(f_meteo)
        df_m.columns = df_m.columns.str.strip()
        df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
        df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
        df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()
        
        # Procesamiento de Campo
        if f_valida.name.endswith('.csv'):
            df_c = pd.read_csv(f_valida)
        else:
            df_c = pd.read_excel(f_valida, engine='openpyxl')
        df_c.columns = df_c.columns.str.strip()
        df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
        df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

        # Cargar Pesos Originales como Semilla de Optimización
        IW_orig = np.load('IW.npy')
        bIW_orig = np.load('bias_IW.npy')
        LW_orig = np.load('LW.npy')
        bLW_orig = np.load('bias_out.npy')

        X_matrix = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)

        if st.button("🔥 Iniciar Optimización Global"):
            with st.spinner("Calibrando 331 neuronas..."):
                
                def objective(params):
                    # Reconstrucción del vector plano a matrices de la RNA
                    ptr = 0
                    new_iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
                    new_biw = params[ptr:ptr+55]; ptr += 55
                    new_lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
                    new_blw = params[ptr]
                    
                    preds = run_ann(X_matrix, new_iw, new_biw, new_lw, new_blw)
                    
                    # Filtros biológicos dinámicos
                    preds[(df_m['Prec_sum'] < u_hidrico) | (df_m['Julian_days'] <= d_inicio)] = 0.0
                    
                    # Sincronización (Ventana 7 días para muestreos semanales)
                    y_p_adj = []
                    for f_o in df_c['FECHA']:
                        mask = (df_m['Fecha'] >= f_o - pd.Timedelta(days=3)) & (df_m['Fecha'] <= f_o + pd.Timedelta(days=3))
                        y_p_adj.append(preds[mask].max() if any(mask) else 0)
                    
                    # Loss: Error Cuadrático Medio
                    return np.mean((df_c['ER_obs'].values - np.array(y_p_adj))**2)

                # Vector inicial plano
                x0 = np.concatenate([IW_orig.flatten(), bIW_orig, LW_orig.flatten(), [bLW_orig]])
                
                # Algoritmo de optimización (L-BFGS-B para gran cantidad de parámetros)
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter})
                
                # Reconstrucción de resultados finales
                ptr = 0
                opt_iw = res.x[ptr:ptr+220].reshape(4, 55); ptr += 220
                opt_biw = res.x[ptr:ptr+55]; ptr += 55
                opt_lw = res.x[ptr:ptr+55].reshape(1, 55); ptr += 55
                opt_blw = res.x[ptr]

                st.success(f"✅ Calibración terminada. Error (MSE): {res.fun:.4f}")

                # Gráfico Comparativo
                p_final = run_ann(X_matrix, opt_iw, opt_biw, opt_lw, opt_blw)
                p_final[(df_m['Prec_sum'] < u_hidrico) | (df_m['Julian_days'] <= d_inicio)] = 0.0
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_m['Fecha'], p_final, label='Modelo Optimizado', color='#1e3a8a', lw=2)
                ax.scatter(df_c['FECHA'], df_c['ER_obs'], color='#b91c1c', s=100, label='Campo', zorder=5)
                ax.set_title("Ajuste Global Finalizado")
                ax.legend()
                st.pyplot(fig)

                # DESCARGA DE RESULTADOS
                st.subheader("📦 Descargar Nuevos Pesos (.npy)")
                cols = st.columns(4)
                def get_download_btn(label, data, fname):
                    buf = io.BytesIO()
                    np.save(buf, data)
                    return st.download_button(label, buf.getvalue(), fname)

                with cols[0]: get_download_btn("IW_opt.npy", opt_iw, "IW_opt.npy")
                with cols[1]: get_download_btn("bias_IW_opt.npy", opt_biw, "bias_IW_opt.npy")
                with cols[2]: get_download_btn("LW_opt.npy", opt_lw, "LW_opt.npy")
                with cols[3]: get_download_btn("bias_out_opt.npy", opt_blw, "bias_out_opt.npy")

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("Sube los archivos del nuevo sitio para iniciar la optimización global.")
