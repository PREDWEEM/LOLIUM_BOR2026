import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import io
import time

# Configuración de la interfaz
st.set_page_config(page_title="PREDWEEM - Sistema de Calibración Global", layout="wide")

# ==========================================
# 1. MOTOR RNA VECTORIZADO (ALTA VELOCIDAD)
# ==========================================
def run_ann_fast(X_n, params):
    """
    Ejecuta la red neuronal de forma matricial (vectorizada).
    Procesa todo el año de una sola operación sin bucles 'for'.
    """
    ptr = 0
    # Reconstrucción de Pesos (4 entradas -> 55 ocultas -> 1 salida = 331 parámetros)
    iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
    biw = params[ptr:ptr+55]; ptr += 55
    lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
    blw = params[ptr]
    
    # Capa Oculta
    a1 = np.tanh(X_n @ iw + biw)
    # Capa de Salida
    z2 = a1 @ lw.T + blw
    # Escalamiento de emergencia relativa [0, 1]
    return (np.tanh(z2).flatten() + 1) / 2

# ==========================================
# 2. FUNCIONES DE DETECCIÓN ROBUSTA
# ==========================================
def safe_get_col(df, keywords, default_idx=1):
    """Busca columnas por palabras clave para evitar IndexError."""
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return df.columns[default_idx] if len(df.columns) > default_idx else df.columns[0]

# ==========================================
# 3. INTERFAZ DE USUARIO
# ==========================================
st.title("🧬 PREDWEEM: Sistema de Optimización Evolutiva")
st.markdown("""
Esta aplicación utiliza **Evolución Diferencial** para encontrar los pesos universales de la Red Neuronal 
a través de múltiples años, considerando ventanas de tolerancia biológica.
""")

# Sidebar: Configuración del algoritmo
st.sidebar.header("⚙️ Configuración del Modelo")
ventana_tol = st.sidebar.slider("Ventana de Tolerancia (días)", 0, 10, 7, help="Tolerancia para desfases entre predicción y muestreo.")
u_lluvia = st.sidebar.slider("Umbral Hídrico Activo (mm)", 5, 30, 20, help="Precipitación acumulada en 21 días para activar emergencia.")

st.sidebar.header("🧬 Algoritmo Evolutivo")
max_gen = st.sidebar.slider("Generaciones Máximas", 5, 100, 20)
pop_size = st.sidebar.slider("Tamaño de Población", 5, 20, 8)

# Carga de Archivos
c1, c2 = st.columns(2)
f_meteo = c1.file_uploader("📂 Excel Meteorología (Multi-hoja)", type=['xlsx'])
f_campo = c2.file_uploader("📂 Excel Verdad de Campo (Multi-hoja)", type=['xlsx'])

if f_meteo and f_campo:
    # Carga de datos (Cacheado para fluidez)
    @st.cache_data
    def load_master_data(fm, fc):
        return pd.read_excel(fm, sheet_name=None), pd.read_excel(fc, sheet_name=None)
    
    dict_m, dict_c = load_master_data(f_meteo, f_campo)
    años_comunes = sorted(list(set(dict_m.keys()) & set(dict_c.keys())))
    
    if not años_comunes:
        st.error("No se encontraron nombres de hojas coincidentes entre Clima y Campo.")
    else:
        st.success(f"✅ {len(años_comunes)} campañas detectadas.")
        
        train_years = st.multiselect("Seleccionar años para ENTRENAMIENTO", años_comunes, default=años_comunes[:-1])
        val_year = st.selectbox("Seleccionar año para VALIDACIÓN CIEGA", [y for y in años_comunes if y not in train_years])

        if st.button("🔥 Iniciar Optimización Global"):
            status = st.status("Preparando Tensores...", expanded=True)
            
            # --- PRE-PROCESAMIENTO VECTORIZADO ---
            data_train = []
            in_min = np.array([1, 0, -7, 0]); in_max = np.array([300, 41, 25.5, 84])
            
            for y in train_years:
                m = dict_m[y].copy().dropna(how='all')
                m.columns = [str(c).strip() for c in m.columns]
                
                # Normalizar Clima
                tmax_c = safe_get_col(m, ['tmax'], 1)
                tmin_c = safe_get_col(m, ['tmin'], 2)
                prec_c = safe_get_col(m, ['prec'], 3)
                
                m['julian'] = pd.to_datetime(m.iloc[:,0]).dt.dayofyear
                X_input = m[['julian', tmax_c, tmin_c, prec_c]].to_numpy(float)
                X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
                
                # Filtros biológicos (Máscara)
                p21 = m[prec_c].rolling(21, min_periods=1).sum().values
                mask_bio = (p21 >= u_lluvia) & (m['julian'] > 10)
                
                # Datos de Campo y Ventanas
                c = dict_c[y].copy().dropna(how='all')
                p_col = safe_get_col(c, ['plant', 'prom', 'pl.m-2', 'plm2'], 1)
                
                obs_dates = pd.to_datetime(c.iloc[:, 0])
                obs_vals = (c[p_col] / c[p_col].max()).values
                meteo_dates = pd.to_datetime(m.iloc[:, 0]).values
                
                # Pre-calcular índices de ventana para velocidad
                win_idx = []
                for d in obs_dates:
                    idxs = np.where((meteo_dates >= (d - pd.Timedelta(days=ventana_tol)).to_datetime64()) & 
                                    (meteo_dates <= (d + pd.Timedelta(days=ventana_tol)).to_datetime64()))[0]
                    win_idx.append(idxs)
                
                data_train.append({'X_n': X_n, 'mask': mask_bio, 'obs': obs_vals, 'idx': win_idx})

            status.update(label="🧬 Ejecutando Evolución Diferencial...", state="running")
            
            # --- FUNCIÓN OBJETIVO ---
            def objective(p):
                mse = 0
                for ds in data_train:
                    preds = run_ann_fast(ds['X_n'], p)
                    preds[~ds['mask']] = 0
                    y_p = np.array([preds[i].max() if len(i)>0 else 0 for i in ds['idx']])
                    mse += np.mean((ds['obs'] - y_p)**2)
                return mse / len(data_train)

            # Optimización
            progress_bar = st.progress(0)
            res = differential_evolution(
                objective, [(-1.5, 1.5)]*331, 
                maxiter=max_gen, popsize=pop_size, 
                callback=lambda x, c: progress_bar.progress(min(c, 1.0)),
                polish=True
            )
            
            status.update(label="✅ Optimización Finalizada", state="complete")

            # ==========================================
            # 4. RESULTADOS Y VALIDACIÓN CIEGA
            # ==========================================
            st.divider()
            st.header(f"📊 Evaluación en Año Ciego: {val_year}")
            
            # Preparar Año de Validación
            mv = dict_m[val_year].copy(); mv.columns = [str(c).strip() for c in mv.columns]
            tmax_v = safe_get_col(mv, ['tmax'], 1); tmin_v = safe_get_col(mv, ['tmin'], 2); prec_v = safe_get_col(mv, ['prec'], 3)
            mv['julian'] = pd.to_datetime(mv.iloc[:,0]).dt.dayofyear
            X_v_n = 2 * (mv[['julian', tmax_v, tmin_v, prec_v]].to_numpy(float) - in_min) / (in_max - in_min) - 1
            
            p_blind = run_ann_fast(X_v_n, res.x)
            p_blind[(mv[prec_v].rolling(21).sum().values < u_lluvia) | (mv['julian'] <= 10)] = 0
            
            # Datos de Campo Validación
            cv = dict_c[val_year].copy(); p_cv = safe_get_col(cv, ['plant', 'prom', 'pl.m-2'], 1)
            cv['ER_obs'] = cv[p_cv] / cv[p_cv].max()
            cv['FECHA'] = pd.to_datetime(cv.iloc[:, 0])
            
            # NSE
            y_p_v = []
            for d in cv['FECHA']:
                m_mask = (pd.to_datetime(mv.iloc[:,0]) >= d - pd.Timedelta(days=ventana_tol)) & \
                         (pd.to_datetime(mv.iloc[:,0]) <= d + pd.Timedelta(days=ventana_tol))
                y_p_v.append(p_blind[m_mask].max() if any(m_mask) else 0)
            
            nse = 1 - (np.sum((cv['ER_obs'] - y_p_v)**2) / np.sum((cv['ER_obs'] - cv['ER_obs'].mean())**2))
            
            m1, m2 = st.columns(2)
            m1.metric("NSE (Confiabilidad)", f"{nse:.3f}")
            m2.metric("Error MSE Final", f"{res.fun:.5f}")

            # Gráfico Final
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(pd.to_datetime(mv.iloc[:,0]), p_blind, label='Modelo PREDWEEM (Ciego)', color='#6d28d9', lw=2)
            ax.scatter(cv['FECHA'], cv['ER_obs'], color='#ef4444', s=100, label='Campo (Realidad)')
            ax.set_ylabel("Emergencia Relativa")
            ax.legend()
            st.pyplot(fig)

            if nse > 0.5:
                st.balloons()
                st.success("🚀 El modelo es altamente confiable para predecir el futuro.")
