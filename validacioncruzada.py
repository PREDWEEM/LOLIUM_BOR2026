import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

st.set_page_config(page_title="PREDWEEM - Master Cross Validator", layout="wide")

# ==========================================
# 1. MOTOR DE LA RED NEURONAL (RNA)
# ==========================================
def run_ann(X_input, params):
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

# ==========================================
# 2. FUNCIONES DE APOYO (SAFE DETECT)
# ==========================================
def get_plant_column(df):
    """Busca la columna de densidad de forma segura."""
    keywords = ['plant', 'prom', 'pl.m-2', 'plm2', 'densidad', 'emergencia']
    # 1. Intento por nombre
    for col in df.columns:
        if any(key in str(col).lower() for key in keywords):
            return col
    # 2. Si falla, intentar la segunda columna (asumiendo Fecha es la 1ra)
    if len(df.columns) >= 2:
        return df.columns[1]
    return None

# ==========================================
# 3. INTERFAZ
# ==========================================
st.title("🧪 PREDWEEM: Master Cross Validator (Versión Robusta)")
st.info("Esta versión previene el error 'Index out of range' validando cada hoja de Excel.")

c1, c2 = st.columns(2)
with c1: f_meteo = st.file_uploader("Excel de Clima", type=['xlsx'])
with c2: f_campo = st.file_uploader("Excel de Campo", type=['xlsx'])

if f_meteo and f_campo:
    try:
        dict_meteo = pd.read_excel(f_meteo, sheet_name=None)
        dict_campo = pd.read_excel(f_campo, sheet_name=None)
        
        años_comunes = sorted(list(set(dict_meteo.keys()) & set(dict_campo.keys())))
        
        if not años_comunes:
            st.error("No hay años coincidentes entre los dos archivos.")
        else:
            train_years = st.multiselect("Años Entrenamiento", años_comunes, default=años_comunes[:-1])
            opc_val = [y for y in años_comunes if y not in train_years]
            val_year = st.selectbox("Año Validación Ciega", opc_val) if opc_val else None

            if st.button("🔥 Iniciar Proceso") and val_year:
                data_train = []
                for y in train_years:
                    # PROCESAR CLIMA
                    dm = dict_meteo[y].copy().dropna(how='all')
                    dm.columns = [str(c).strip() for c in dm.columns]
                    dm = dm.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec', 'Prec':'Prec', 'tmax':'TMAX', 'tmin':'TMIN'})
                    
                    # PROCESAR CAMPO
                    dc = dict_campo[y].copy().dropna(how='all')
                    dc.columns = [str(c).strip() for c in dc.columns]
                    
                    p_col = get_plant_column(dc)
                    
                    if p_col is None:
                        st.warning(f"⚠️ Saltando año {y}: No se encontró columna de plantas.")
                        continue
                    
                    # Sincronización de Fechas
                    dm['Fecha'] = pd.to_datetime(dm.iloc[:, 0])
                    dm['Julian_days'] = dm['Fecha'].dt.dayofyear
                    dm['P21'] = dm['Prec'].rolling(21, min_periods=1).sum()
                    
                    dc['FECHA'] = pd.to_datetime(dc.iloc[:, 0])
                    dc['ER_obs'] = dc[p_col] / dc[p_col].max()
                    
                    data_train.append({'m': dm, 'c': dc, 'year': y})

                if not data_train:
                    st.error("No se pudo preparar ningún año para el entrenamiento.")
                else:
                    # OPTIMIZACIÓN
                    try:
                        x0 = np.concatenate([np.load('IW.npy').flatten(), np.load('bias_IW.npy'), 
                                             np.load('LW.npy').flatten(), [np.load('bias_out.npy')]])
                    except:
                        x0 = np.zeros(331)

                    def objective(p):
                        err = 0
                        for ds in data_train:
                            X = ds['m'][["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                            preds = run_ann(X, p)
                            preds[(ds['m']['P21'] < 15) | (ds['m']['Julian_days'] <= 10)] = 0
                            
                            vals = []
                            for fo in ds['c']['FECHA']:
                                mask = (ds['m']['Fecha'] >= fo - pd.Timedelta(days=3)) & (ds['m']['Fecha'] <= fo + pd.Timedelta(days=3))
                                vals.append(preds[mask].max() if any(mask) else 0)
                            err += np.mean((ds['c']['ER_obs'].values - np.array(vals))**2)
                        return err / len(data_train)

                    with st.spinner("Optimizando pesos universales..."):
                        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 35})
                        
                        # --- VALIDACIÓN CIEGA ---
                        # (Repetir lógica de limpieza para el año val_year)
                        dm_v = dict_meteo[val_year].copy()
                        dm_v.columns = [str(c).strip() for c in dm_v.columns]
                        dm_v = dm_v.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec', 'tmax':'TMAX', 'tmin':'TMIN'})
                        dm_v['Fecha'] = pd.to_datetime(dm_v.iloc[:, 0])
                        dm_v['Julian_days'] = dm_v['Fecha'].dt.dayofyear
                        dm_v['P21'] = dm_v['Prec'].rolling(21, min_periods=1).sum()
                        
                        X_v = dm_v[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                        p_v = run_ann(X_v, res.x)
                        p_v[(dm_v['P21'] < 15) | (dm_v['Julian_days'] <= 10)] = 0
                        
                        dc_v = dict_campo[val_year].copy()
                        pc_v = get_plant_column(dc_v)
                        dc_v['ER_obs'] = dc_v[pc_v] / dc_v[pc_v].max()
                        dc_v['FECHA'] = pd.to_datetime(dc_v.iloc[:, 0])
                        
                        # Cálculo NSE
                        y_p = []
                        for fo in dc_v['FECHA']:
                            mask = (dm_v['Fecha'] >= fo - pd.Timedelta(days=3)) & (dm_v['Fecha'] <= fo + pd.Timedelta(days=3))
                            y_p.append(p_v[mask].max() if any(mask) else 0)
                        
                        nse = 1 - (np.sum((dc_v['ER_obs'] - y_p)**2) / np.sum((dc_v['ER_obs'] - dc_v['ER_obs'].mean())**2))
                        
                        st.header(f"📊 Resultado Test Ciego: {val_year}")
                        st.metric("NSE (Confiabilidad)", f"{nse:.3f}")
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(dm_v['Fecha'], p_v, color='purple', label='Predicción')
                        ax.scatter(dc_v['FECHA'], dc_v['ER_obs'], color='red', label='Campo')
                        ax.legend(); st.pyplot(fig)

    except Exception as e:
        st.error(f"Error general: {e}")
