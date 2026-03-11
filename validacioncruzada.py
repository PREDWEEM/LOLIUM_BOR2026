import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import io
import time

st.set_page_config(page_title="PREDWEEM - Fast Optimizer", layout="wide")

# ==========================================
# 1. MOTOR RNA VECTORIZADO (ALTA VELOCIDAD)
# ==========================================
def run_ann_fast(X_n, params):
    """Procesamiento matricial completo sin bucles for"""
    ptr = 0
    iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
    biw = params[ptr:ptr+55]; ptr += 55
    lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
    blw = params[ptr]
    
    # Capa Oculta (N, 55)
    a1 = np.tanh(X_n @ iw + biw)
    # Capa Salida (N, 1)
    z2 = a1 @ lw.T + blw
    # Activación y escalamiento [0, 1]
    return (np.tanh(z2).flatten() + 1) / 2

# ==========================================
# 2. INTERFAZ Y CARGA
# ==========================================
st.title("🚀 PREDWEEM: Optimizador Ultra-Rápido")
st.markdown("Optimización vectorizada para evitar bloqueos en la App.")

col1, col2 = st.columns(2)
f_meteo = col1.file_uploader("Excel Clima", type=['xlsx'])
f_campo = col2.file_uploader("Excel Campo", type=['xlsx'])

if f_meteo and f_campo:
    # Cacheamos la carga para no repetir proceso al tocar sliders
    @st.cache_data
    def load_data(fm, fc):
        dm = pd.read_excel(fm, sheet_name=None)
        dc = pd.read_excel(fc, sheet_name=None)
        return dm, dc

    dict_m, dict_c = load_data(f_meteo, f_campo)
    años = sorted(list(set(dict_m.keys()) & set(dict_c.keys())))
    
    st.sidebar.header("⚙️ Ajustes de Velocidad")
    # Ajustes más bajos para que responda rápido en la web
    max_gen = st.sidebar.slider("Generaciones Máximas", 5, 50, 15)
    pop = st.sidebar.slider("Tamaño Población", 5, 15, 8)
    ventana = st.sidebar.slider("Ventana Tolerancia (días)", 0, 10, 7)

    train_years = st.multiselect("Años Entrenamiento", años, default=años[:-1])
    val_year = st.selectbox("Año Validación Ciega", [y for y in años if y not in train_years])

    if st.button("🔥 Iniciar Optimización"):
        status = st.status("Preparando datos...", expanded=True)
        
        # PRE-PROCESAMIENTO: Normalizar y calcular P21 una sola vez
        data_train = []
        in_min = np.array([1, 0, -7, 0]); in_max = np.array([300, 41, 25.5, 84])
        
        for y in train_years:
            m = dict_m[y].copy()
            m.columns = [str(c).strip().lower() for c in m.columns]
            m = m.rename(columns={'tmax':'TMAX', 'tmin':'TMIN', 'prec':'Prec'})
            
            # Preparar matriz de entrada normalizada (X_n)
            X_raw = m[['fecha', 'TMAX', 'TMIN', 'Prec']].copy()
            X_raw['julian'] = pd.to_datetime(X_raw.iloc[:,0]).dt.dayofyear
            X_input = X_raw[['julian', 'TMAX', 'TMIN', 'Prec']].to_numpy(float)
            X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
            
            # Máscara de filtros
            p21 = m['Prec'].rolling(21, min_periods=1).sum().values
            mask_bio = (p21 >= 15) & (X_raw['julian'] > 10)
            
            # Datos de campo e índices de ventana
            c = dict_c[y].copy()
            c_col = [col for col in c.columns if any(k in str(col).lower() for k in ['plant', 'prom', 'pl.m-2'])][0]
            obs_dates = pd.to_datetime(c.iloc[:, 0])
            obs_vals = (c[c_col] / c[c_col].max()).values
            meteo_dates = pd.to_datetime(m.iloc[:, 0]).values
            
            # Pre-calcular los índices de la ventana para cada observación
            window_indices = []
            for d in obs_dates:
                idx = np.where((meteo_dates >= (d - pd.Timedelta(days=ventana)).to_datetime64()) & 
                               (meteo_dates <= (d + pd.Timedelta(days=ventana)).to_datetime64()))[0]
                window_indices.append(idx)
            
            data_train.append({
                'X_n': X_n, 'mask': mask_bio, 
                'obs_vals': obs_vals, 'win_idx': window_indices
            })

        status.update(label="🚀 Ejecutando Algoritmo Evolutivo...", state="running")
        
        # FUNCIÓN OBJETIVO ULTRA-RÁPIDA
        def objective(p):
            mse_total = 0
            for ds in data_train:
                preds = run_ann_fast(ds['X_n'], p)
                preds[~ds['mask']] = 0 # Aplicar filtros vectorizados
                
                # Extraer máximos en ventanas pre-calculadas
                y_p = np.array([preds[idx].max() if len(idx) > 0 else 0 for idx in ds['win_idx']])
                mse_total += np.mean((ds['obs_vals'] - y_p)**2)
            return mse_total / len(data_train)

        # Barra de progreso personalizada
        progress_bar = st.progress(0)
        start_time = time.time()

        def callback(xk, convergence):
            # Esta función se llama en cada generación
            progress_bar.progress(min(convergence, 1.0))

        # Evolución Diferencial
        bounds = [(-1.5, 1.5)] * 331
        res = differential_evolution(
            objective, bounds, maxiter=max_gen, popsize=pop, 
            callback=callback, polish=True
        )

        status.update(label=f"✅ ¡Completado en {time.time()-start_time:.1f}s!", state="complete", expanded=False)

        # --- VALIDACIÓN FINAL (Igual que antes pero rápido) ---
        st.subheader(f"📊 Resultado Validación Ciega: {val_year}")
        # (Aquí va el código de visualización usando run_ann_fast)
        # ... [omitido por brevedad, usa la misma lógica de run_ann_fast] ...
        st.success(f"Optimización finalizada. Error mínimo hallado: {res.fun:.5f}")
