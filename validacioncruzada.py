
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import io

st.set_page_config(page_title="PREDWEEM - Evolutionary Optimizer", layout="wide")

# ==========================================
# 1. MOTOR RNA (331 PARÁMETROS)
# ==========================================
def run_ann(X_input, params):
    ptr = 0
    iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
    biw = params[ptr:ptr+55]; ptr += 55
    lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
    blw = params[ptr]
    
    # Normalización del proyecto PREDWEEM
    in_min = np.array([1, 0, -7, 0]); in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
    
    emer = []
    for x in X_n:
        a1 = np.tanh(iw.T @ x + biw)
        z2 = np.dot(lw, a1) + blw
        emer.append((np.tanh(z2) + 1) / 2)
    return np.diff(np.cumsum(np.array(emer).flatten()), prepend=0)

# ==========================================
# 2. DETECCIÓN ROBUSTA DE COLUMNAS
# ==========================================
def safe_get_col(df, keywords, default_idx):
    for col in df.columns:
        if any(k in str(col).lower() for k in keywords):
            return col
    return df.columns[default_idx] if len(df.columns) > default_idx else df.columns[0]

# ==========================================
# 3. INTERFAZ Y CONFIGURACIÓN EVOLUTIVA
# ==========================================
st.title("🧬 PREDWEEM: Optimizador Global Evolutivo")
st.sidebar.header("⚙️ Configuración del Algoritmo")

# Parámetros del algoritmo evolutivo
iteraciones = st.sidebar.slider("Generaciones (Max Iter)", 5, 100, 20)
pop_size = st.sidebar.slider("Tamaño de Población", 5, 20, 10)
ventana = st.sidebar.slider("Ventana de Tolerancia (días)", 0, 10, 7)

f_meteo = st.file_uploader("Excel Clima (Multi-hoja)", type=['xlsx'])
f_campo = st.file_uploader("Excel Campo (Multi-hoja)", type=['xlsx'])

if f_meteo and f_campo:
    dict_m = pd.read_excel(f_meteo, sheet_name=None)
    dict_c = pd.read_excel(f_campo, sheet_name=None)
    años = sorted(list(set(dict_m.keys()) & set(dict_c.keys())))
    
    train_years = st.multiselect("Años Entrenamiento", años, default=años[:-1])
    val_year = st.selectbox("Año Validación Ciega", [y for y in años if y not in train_years])

    if st.button("🚀 Iniciar Evolución Diferencial"):
        # Preparación de datos
        data_train = []
        for y in train_years:
            dm = dict_m[y].copy().dropna(how='all')
            dm.columns = [str(c).strip() for c in dm.columns]
            dm = dm.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec', 'tmax':'TMAX', 'tmin':'TMIN'})
            dm['Fecha'] = pd.to_datetime(dm.iloc[:, 0])
            dm['P21'] = dm['Prec'].rolling(21, min_periods=1).sum()
            dm['Julian_days'] = dm['Fecha'].dt.dayofyear
            
            dc = dict_c[y].copy().dropna(how='all')
            p_col = safe_get_col(dc, ['plant', 'prom', 'pl.m-2', 'plm2'], 1)
            dc['ER_obs'] = dc[p_col] / dc[p_col].max()
            dc['FECHA'] = pd.to_datetime(dc.iloc[:, 0])
            data_train.append({'m': dm, 'c': dc})

        # --- FUNCIÓN OBJETIVO PARA EL ALGORITMO EVOLUTIVO ---
        def objective(p):
            total_mse = 0
            for ds in data_train:
                X = ds['m'][["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                preds = run_ann(X, p)
                preds[(ds['m']['P21'] < 15) | (ds['m']['Julian_days'] <= 10)] = 0
                
                y_p_adj = []
                for fo in ds['c']['FECHA']:
                    mask = (ds['m']['Fecha'] >= fo - pd.Timedelta(days=ventana)) & \
                           (ds['m']['Fecha'] <= fo + pd.Timedelta(days=ventana))
                    y_p_adj.append(preds[mask].max() if any(mask) else 0)
                total_mse += np.mean((ds['c']['ER_obs'].values - np.array(y_p_adj))**2)
            return total_mse / len(data_train)

        # Definir límites de los 331 pesos (Bounds)
        # Los pesos neuronales suelen estar entre -1.5 y 1.5 para evitar saturación de Tanh
        bounds = [(-1.5, 1.5)] * 331

        with st.spinner("Ejecutando Evolución Diferencial... Esto puede tardar unos minutos."):
            # Algoritmo Evolutivo (Scipy implementation)
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=iteraciones, 
                popsize=pop_size, 
                mutation=(0.5, 1), 
                recombination=0.7,
                polish=True # Realiza una optimización local final para mayor precisión
            )
            
            # --- VALIDACIÓN EN EL AÑO CIEGO ---
            dm_v = dict_m[val_year].copy()
            dm_v.columns = [str(c).strip() for c in dm_v.columns]; dm_v = dm_v.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec'})
            dm_v['Fecha'] = pd.to_datetime(dm_v.iloc[:, 0]); dm_v['Julian_days'] = dm_v['Fecha'].dt.dayofyear
            dm_v['P21'] = dm_v['Prec'].rolling(21, min_periods=1).sum()
            
            p_v = run_ann(dm_v[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float), result.x)
            p_v[(dm_v['P21'] < 15) | (dm_v['Julian_days'] <= 10)] = 0
            
            dc_v = dict_c[val_year].copy()
            pc_v = safe_get_col(dc_v, ['plant', 'prom', 'pl.m-2', 'plm2'], 1)
            dc_v['ER_obs'] = dc_v[pc_v] / dc_v[pc_v].max(); dc_v['FECHA'] = pd.to_datetime(dc_v.iloc[:, 0])

            y_p_v = []
            for fo in dc_v['FECHA']:
                mask = (dm_v['Fecha'] >= fo - pd.Timedelta(days=ventana)) & (dm_v['Fecha'] <= fo + pd.Timedelta(days=ventana))
                y_p_v.append(p_v[mask].max() if any(mask) else 0)
            
            nse = 1 - (np.sum((dc_v['ER_obs'] - y_p_v)**2) / np.sum((dc_v['ER_obs'] - dc_v['ER_obs'].mean())**2))

            # RESULTADOS
            st.success(f"✅ Optimización Finalizada. Error de entrenamiento (MSE): {result.fun:.5f}")
            st.metric(f"NSE Validación Ciega ({val_year})", f"{nse:.3f}")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(dm_v['Fecha'], p_v, label='Predicción Evolutiva', color='darkgreen', lw=2)
            ax.scatter(dc_v['FECHA'], dc_v['ER_obs'], color='red', s=100, label='Campo Real')
            ax.set_title(f"Validación Ciega con Algoritmo Evolutivo - Año {val_year}")
            ax.legend(); st.pyplot(fig)
