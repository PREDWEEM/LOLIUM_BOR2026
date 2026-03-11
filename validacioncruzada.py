import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

st.set_page_config(page_title="PREDWEEM - Cross Validator", layout="wide")

# ==========================================
# 1. MOTOR RNA Y PROCESAMIENTO
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
# 2. INTERFAZ: CARGA DE MÚLTIPLES AÑOS
# ==========================================
st.title("🧬 PREDWEEM: Validador Cruzado Multi-Año")
st.markdown("""
Sube pares de archivos (Clima + Campo) para diferentes años. 
El sistema buscará los pesos que mejor funcionen para **todos** los años simultáneamente.
""")

if 'datasets' not in st.session_state:
    st.session_state.datasets = []

with st.expander("➕ Añadir Año al Estudio", expanded=True):
    c1, c2, c3 = st.columns([2, 2, 1])
    year_label = c1.text_input("Etiqueta (ej: 2025)", placeholder="Año")
    f_m = c2.file_uploader(f"Clima {year_label}", type=['csv'], key=f"m_{year_label}")
    f_c = c3.file_uploader(f"Campo {year_label}", type=['csv', 'xlsx'], key=f"c_{year_label}")
    
    if st.button("Registrar Año"):
        if f_m and f_c and year_label:
            st.session_state.datasets.append({
                'year': year_label, 'meteo': f_m, 'campo': f_c
            })
            st.rerun()

# Listar años cargados
if st.session_state.datasets:
    st.subheader("🗓️ Años Cargados para Validación")
    for i, ds in enumerate(st.session_state.datasets):
        st.write(f"- **{ds['year']}**: {ds['meteo'].name} | {ds['campo'].name}")
    if st.button("Limpiar todo"):
        st.session_state.datasets = []
        st.rerun()

# ==========================================
# 3. LÓGICA DE OPTIMIZACIÓN CRUZADA
# ==========================================
if len(st.session_state.datasets) >= 2:
    st.divider()
    train_years = st.multiselect("Seleccionar años para ENTRENAR", 
                                  [d['year'] for d in st.session_state.datasets])
    val_year = st.selectbox("Seleccionar año para VALIDACIÓN CIEGA", 
                             [d['year'] for d in st.session_state.datasets if d['year'] not in train_years])

    if st.button("🔥 Iniciar Validación Cruzada"):
        # Preparar datos de entrenamiento
        data_train = []
        for ds in st.session_state.datasets:
            if ds['year'] in train_years:
                df_m = pd.read_csv(ds['meteo'])
                df_m['Julian_days'] = pd.to_datetime(df_m['Fecha']).dt.dayofyear
                df_m['P21'] = df_m['Prec'].rolling(21, min_periods=1).sum()
                
                df_c = pd.read_excel(ds['campo']) if ds['campo'].name.endswith('xlsx') else pd.read_csv(ds['campo'])
                df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()
                df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
                
                data_train.append({'m': df_m, 'c': df_c})

        # Cargar pesos semilla
        try:
            x0 = np.concatenate([np.load('IW.npy').flatten(), np.load('bias_IW.npy'), 
                                 np.load('LW.npy').flatten(), [np.load('bias_out.npy')]])
        except:
            x0 = np.random.uniform(-0.1, 0.1, 331)

        def objective(p):
            total_error = 0
            for ds in data_train:
                X = ds['m'][["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                preds = run_ann(X, p)
                preds[(ds['m']['P21'] < 15) | (ds['m']['Julian_days'] <= 10)] = 0
                
                y_p_adj = []
                for f_o in ds['c']['FECHA']:
                    mask = (pd.to_datetime(ds['m']['Fecha']) >= f_o - pd.Timedelta(days=3)) & \
                           (pd.to_datetime(ds['m']['Fecha']) <= f_o + pd.Timedelta(days=3))
                    y_p_adj.append(preds[mask].max() if any(mask) else 0)
                
                total_error += np.mean((ds['c']['ER_obs'].values - np.array(y_p_adj))**2)
            return total_error / len(data_train)

        with st.spinner("Buscando pesos universales..."):
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 40})
            st.success("✅ Pesos Universales Encontrados")

            # --- TEST EN AÑO DE VALIDACIÓN ---
            ds_val = [d for d in st.session_state.datasets if d['year'] == val_year][0]
            df_mv = pd.read_csv(ds_val['meteo'])
            df_mv['Julian_days'] = pd.to_datetime(df_mv['Fecha']).dt.dayofyear
            df_mv['P21'] = df_mv['Prec'].rolling(21, min_periods=1).sum()
            df_cv = pd.read_excel(ds_val['campo']) if ds_val['campo'].name.endswith('xlsx') else pd.read_csv(ds_val['campo'])
            df_cv['ER_obs'] = df_cv['PLM2'] / df_cv['PLM2'].max()
            
            X_val = df_mv[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
            p_val = run_ann(X_matrix=X_val, params=res.x)
            p_val[(df_mv['P21'] < 15) | (df_mv['Julian_days'] <= 10)] = 0

            # Calcular NSE de Validación
            y_p_v = []
            for f_o in pd.to_datetime(df_cv['FECHA']):
                mask = (pd.to_datetime(df_mv['Fecha']) >= f_o - pd.Timedelta(days=3)) & \
                       (pd.to_datetime(df_mv['Fecha']) <= f_o + pd.Timedelta(days=3))
                y_p_v.append(p_val[mask].max() if any(mask) else 0)
            
            y_o_v = df_cv['ER_obs'].values
            nse_val = 1 - (np.sum((y_o_v - y_p_v)**2) / np.sum((y_o_v - np.mean(y_o_v))**2))

            # Visualización
            st.subheader(f"📊 Validación en Año Ciego: {val_year}")
            st.metric("NSE de Validación (Confiabilidad)", f"{nse_val:.3f}")
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(pd.to_datetime(df_mv['Fecha']), p_val, label='Predicción Ciega', color='purple')
            ax.scatter(pd.to_datetime(df_cv['FECHA']), df_cv['ER_obs'], color='red', label='Realidad')
            ax.legend()
            st.pyplot(fig)

            if nse_val > 0.5:
                st.balloons()
                st.success("🚀 El modelo es ALTAMENTE CONFIABLE para predecir nuevos años.")
            else:
                st.warning("⚠️ El modelo aún tiene dificultades para generalizar. Prueba añadir más años.")
