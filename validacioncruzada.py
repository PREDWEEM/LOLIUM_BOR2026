
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="PREDWEEM - Validador con Tolerancia", layout="wide")

# ==========================================
# 1. MOTOR RNA
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
# 2. CONFIGURACIÓN EN SIDEBAR
# ==========================================
st.sidebar.header("🛠️ Parámetros de Validación")
ventana = st.sidebar.slider("Ventana de Tolerancia (días)", 0, 10, 7, 
                            help="Busca el máximo predicho en ± N días alrededor de la observación.")
u_lluvia = st.sidebar.slider("Umbral Hídrico (mm)", 5, 30, 15)

# ==========================================
# 3. INTERFAZ DE CARGA
# ==========================================
st.title("🧪 PREDWEEM: Validación con Tolerancia Temporal")
st.markdown(f"Considerando un desfase aceptable de **±{ventana} días**.")

c1, c2 = st.columns(2)
f_meteo = c1.file_uploader("Excel Clima (Multi-hoja)", type=['xlsx'])
f_campo = c2.file_uploader("Excel Campo (Multi-hoja)", type=['xlsx'])

if f_meteo and f_campo:
    try:
        dict_m = pd.read_excel(f_meteo, sheet_name=None)
        dict_c = pd.read_excel(f_campo, sheet_name=None)
        años = sorted(list(set(dict_m.keys()) & set(dict_c.keys())))
        
        train_years = st.multiselect("Años Entrenamiento", años, default=años[:-1])
        val_year = st.selectbox("Año Validación Ciega", [y for y in años if y not in train_years])

        if st.button("🚀 Ejecutar Validación Cruzada"):
            # Preparar entrenamiento
            data_train = []
            for y in train_years:
                dm = dict_m[y].copy()
                dm.columns = [str(c).strip() for c in dm.columns]
                dm = dm.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec', 'tmax':'TMAX', 'tmin':'TMIN'})
                dm['Fecha'] = pd.to_datetime(dm.iloc[:, 0])
                dm['P21'] = dm['Prec'].rolling(21, min_periods=1).sum()
                dm['Julian_days'] = dm['Fecha'].dt.dayofyear
                
                dc = dict_c[y].copy()
                p_col = [c for c in dc.columns if any(x in str(c).lower() for x in ['plant', 'prom', 'pl.m-2'])][0]
                dc['ER_obs'] = dc[p_col] / dc[p_col].max()
                dc['FECHA'] = pd.to_datetime(dc.iloc[:, 0])
                data_train.append({'m': dm, 'c': dc})

            # Optimización
            try: x0 = np.concatenate([np.load('IW.npy').flatten(), np.load('bias_IW.npy'), np.load('LW.npy').flatten(), [np.load('bias_out.npy')]])
            except: x0 = np.random.uniform(-0.1, 0.1, 331)

            def objective(p):
                err = 0
                for ds in data_train:
                    X = ds['m'][["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                    preds = run_ann(X, p)
                    preds[(ds['m']['P21'] < u_lluvia) | (ds['m']['Julian_days'] <= 10)] = 0
                    
                    y_p_adj = []
                    for fo in ds['c']['FECHA']:
                        # Aplicar ventana de tolerancia dinámica
                        mask = (ds['m']['Fecha'] >= fo - pd.Timedelta(days=ventana)) & \
                               (ds['m']['Fecha'] <= fo + pd.Timedelta(days=ventana))
                        y_p_adj.append(preds[mask].max() if any(mask) else 0)
                    err += np.mean((ds['c']['ER_obs'].values - np.array(y_p_adj))**2)
                return err / len(data_train)

            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 30})

            # --- VALIDACIÓN CIEGA CON VENTANA ---
            dm_v = dict_m[val_year].copy()
            dm_v.columns = [str(c).strip() for c in dm_v.columns]
            dm_v = dm_v.rename(columns={'Tmax':'TMAX', 'Tmin':'TMIN', 'prec':'Prec', 'tmax':'TMAX', 'tmin':'TMIN'})
            dm_v['Fecha'] = pd.to_datetime(dm_v.iloc[:, 0])
            dm_v['Julian_days'] = dm_v['Fecha'].dt.dayofyear
            dm_v['P21'] = dm_v['Prec'].rolling(21, min_periods=1).sum()
            
            X_v = dm_v[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
            p_v = run_ann(X_v, res.x)
            p_v[(dm_v['P21'] < u_lluvia) | (dm_v['Julian_days'] <= 10)] = 0
            
            dc_v = dict_c[val_year].copy()
            p_col_v = [c for c in dc_v.columns if any(x in str(c).lower() for x in ['plant', 'prom', 'pl.m-2'])][0]
            dc_v['ER_obs'] = dc_v[p_col_v] / dc_v[p_col_v].max()
            dc_v['FECHA'] = pd.to_datetime(dc_v.iloc[:, 0])

            y_p_v = []
            for fo in dc_v['FECHA']:
                mask = (dm_v['Fecha'] >= fo - pd.Timedelta(days=ventana)) & \
                       (dm_v['Fecha'] <= fo + pd.Timedelta(days=ventana))
                y_p_v.append(p_v[mask].max() if any(mask) else 0)
            
            y_o_v = dc_v['ER_obs'].values
            nse = 1 - (np.sum((y_o_v - y_p_v)**2) / np.sum((y_o_v - np.mean(y_o_v))**2))

            # RESULTADOS
            st.header(f"📊 Año de Test: {val_year} (Ventana ±{ventana} días)")
            st.metric("NSE Corregido por Fase", f"{nse:.3f}")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(dm_v['Fecha'], p_v, label='Predicción RNA', color='purple', alpha=0.6)
            ax.scatter(dc_v['FECHA'], dc_v['ER_obs'], color='red', label='Campo (Real)')
            # Dibujar el "ajuste de ventana" para visualización
            ax.scatter(dc_v['FECHA'], y_p_v, color='blue', marker='x', s=100, label='Mejor ajuste en ventana')
            ax.set_title(f"Validación con Tolerancia de {ventana} días")
            ax.legend(); st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
