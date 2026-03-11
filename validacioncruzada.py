import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# FUNCIONES DE SEGURIDAD (PREVENCIÓN DE INDEXERROR)
# ==========================================
def buscar_columna_plantas(df):
    """Detecta la columna de densidad sin fallar si no hay coincidencias."""
    keywords = ['plant', 'prom', 'pl.m-2', 'plm2', 'densidad']
    encontradas = [c for c in df.columns if any(k in str(c).lower() for k in keywords)]
    
    if encontradas:
        return encontradas[0]
    # Si no hay coincidencias, devolvemos la segunda columna (asumiendo que la 1ra es Fecha)
    if len(df.columns) >= 2:
        return df.columns[1]
    return None

# ==========================================
# MOTOR RNA
# ==========================================
def run_ann(X_input, params):
    try:
        ptr = 0
        iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
        biw = params[ptr:ptr+55]; ptr += 55
        lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
        blw = params[ptr]
        
        in_min = np.array([1, 0, -7, 0]); in_max = np.array([300, 41, 25.5, 84])
        X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
        
        emer = []
        for x in X_n:
            a1 = np.tanh(iw.T @ x + biw)
            z2 = np.dot(lw, a1) + blw
            emer.append((np.tanh(z2) + 1) / 2)
        return np.diff(np.cumsum(np.array(emer).flatten()), prepend=0)
    except Exception as e:
        st.error(f"Error en arquitectura RNA: {e}")
        return np.zeros(len(X_input))

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("🧪 PREDWEEM: Validador Seguro")
st.sidebar.header("Parámetros")
ventana = st.sidebar.slider("Ventana (días)", 0, 10, 7)

f_meteo = st.file_uploader("Excel Clima", type=['xlsx'])
f_campo = st.file_uploader("Excel Campo", type=['xlsx'])

if f_meteo and f_campo:
    try:
        dict_m = pd.read_excel(f_meteo, sheet_name=None)
        dict_c = pd.read_excel(f_campo, sheet_name=None)
        años = sorted(list(set(dict_m.keys()) & set(dict_c.keys())))
        
        train_years = st.multiselect("Años Entrenamiento", años, default=años[:-1] if len(años)>1 else años)
        val_year = st.selectbox("Año Validación Ciega", [y for y in años if y not in train_years] if len(años)>1 else [años[0]])

        if st.button("🚀 Iniciar"):
            data_train = []
            for y in train_years:
                dm = dict_m[y].copy().dropna(how='all')
                dm.columns = [str(c).strip() for c in dm.columns]
                dm = dm.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec', 'tmax':'TMAX', 'tmin':'TMIN', 'Prec':'Prec'})
                
                dc = dict_c[y].copy().dropna(how='all')
                p_col = buscar_columna_plantas(dc)
                
                if p_col is None:
                    st.warning(f"⚠️ No se encontró columna de datos en el año {y}. Saltando...")
                    continue
                
                dm['Fecha'] = pd.to_datetime(dm.iloc[:, 0])
                dm['Julian_days'] = dm['Fecha'].dt.dayofyear
                dm['P21'] = dm['Prec'].rolling(21, min_periods=1).sum()
                
                dc['FECHA'] = pd.to_datetime(dc.iloc[:, 0])
                dc['ER_obs'] = dc[p_col] / dc[p_col].max()
                data_train.append({'m': dm, 'c': dc})

            # Optimización (BFGS)
            x0 = np.random.uniform(-0.1, 0.1, 331) # Semilla aleatoria si no hay archivos .npy

            def objective(p):
                err = 0
                for ds in data_train:
                    X = ds['m'][["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                    preds = run_ann(X, p)
                    preds[(ds['m']['P21'] < 15) | (ds['m']['Julian_days'] <= 10)] = 0
                    
                    y_p_adj = []
                    for fo in ds['c']['FECHA']:
                        mask = (ds['m']['Fecha'] >= fo - pd.Timedelta(days=ventana)) & \
                               (ds['m']['Fecha'] <= fo + pd.Timedelta(days=ventana))
                        y_p_adj.append(preds[mask].max() if any(mask) else 0)
                    err += np.mean((ds['c']['ER_obs'].values - np.array(y_p_adj))**2)
                return err / len(data_train)

            with st.spinner("Calibrando..."):
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 30})
                
                # --- TEST CIEGO ---
                dm_v = dict_m[val_year].copy()
                dm_v.columns = [str(c).strip() for c in dm_v.columns]
                dm_v = dm_v.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec', 'tmax':'TMAX', 'tmin':'TMIN'})
                dm_v['Fecha'] = pd.to_datetime(dm_v.iloc[:, 0])
                dm_v['Julian_days'] = dm_v['Fecha'].dt.dayofyear
                dm_v['P21'] = dm_v['Prec'].rolling(21, min_periods=1).sum()
                
                p_v = run_ann(dm_v[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float), res.x)
                p_v[(dm_v['P21'] < 15) | (dm_v['Julian_days'] <= 10)] = 0
                
                dc_v = dict_c[val_year].copy()
                pc_v = buscar_columna_plantas(dc_v)
                dc_v['ER_obs'] = dc_v[pc_v] / dc_v[pc_v].max()
                dc_v['FECHA'] = pd.to_datetime(dc_v.iloc[:, 0])

                y_p_v = []
                for fo in dc_v['FECHA']:
                    mask = (dm_v['Fecha'] >= fo - pd.Timedelta(days=ventana)) & \
                           (dm_v['Fecha'] <= fo + pd.Timedelta(days=ventana))
                    y_p_v.append(p_v[mask].max() if any(mask) else 0)
                
                nse = 1 - (np.sum((dc_v['ER_obs'] - y_p_v)**2) / np.sum((dc_v['ER_obs'] - dc_v['ER_obs'].mean())**2))
                st.metric("NSE (Fiabilidad)", f"{nse:.3f}")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(dm_v['Fecha'], p_v, color='purple', label='Predicción')
                ax.scatter(dc_v['FECHA'], dc_v['ER_obs'], color='red', label='Campo')
                ax.legend(); st.pyplot(fig)

    except Exception as e:
        st.error(f"Error general: {e}")
