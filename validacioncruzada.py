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
    """
    Ejecuta el paso hacia adelante de la red neuronal.
    X_input: Matriz de entrada (Julian_days, TMAX, TMIN, Prec)
    params: Vector de 331 pesos y desvíos
    """
    ptr = 0
    # Reconstrucción de la arquitectura
    iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
    biw = params[ptr:ptr+55]; ptr += 55
    lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
    blw = params[ptr]
    
    # Escalamiento de sensores (Normalización original)
    in_min = np.array([1, 0, -7, 0])
    in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
    
    emer = []
    for x in X_n:
        # Capa Oculta
        a1 = np.tanh(iw.T @ x + biw)
        # Capa de Salida
        z2 = np.dot(lw, a1) + blw
        emer.append((np.tanh(z2) + 1) / 2)
        
    return np.diff(np.cumsum(np.array(emer).flatten()), prepend=0)

# 

# ==========================================
# 2. INTERFAZ Y CARGA DE DATOS
# ==========================================
st.title("🧪 PREDWEEM: Validador Cruzado Multi-Año (.xlsx)")
st.markdown("""
Esta herramienta sincroniza automáticamente tus archivos de clima y campo por nombre de hoja (Año).
Permite entrenar con varios años y validar la confiabilidad en un 'año ciego'.
""")

col1, col2 = st.columns(2)
with col1:
    f_meteo = st.file_uploader("1. Excel de Clima (Multi-hoja)", type=['xlsx'])
with col2:
    f_campo = st.file_uploader("2. Excel de Patrones (Multi-hoja)", type=['xlsx'])

if f_meteo and f_campo:
    try:
        # Leer libros completos
        dict_meteo = pd.read_excel(f_meteo, sheet_name=None)
        dict_campo = pd.read_excel(f_campo, sheet_name=None)
        
        # Sincronización por nombre de hoja
        años_comunes = sorted(list(set(dict_meteo.keys()) & set(dict_campo.keys())))
        
        if not años_comunes:
            st.error("No se encontraron nombres de hojas coincidentes (ej: '2025') en ambos archivos.")
        else:
            st.success(f"✅ Años sincronizados: {', '.join(años_comunes)}")

            # Selección de Años para el estudio
            st.divider()
            c_a, c_b = st.columns(2)
            train_years = c_a.multiselect("Años para ENTRENAMIENTO", años_comunes, default=años_comunes[:-1])
            
            opciones_val = [y for y in años_comunes if y not in train_years]
            if opciones_val:
                val_year = c_b.selectbox("Año para VALIDACIÓN CIEGA (Test)", opciones_val)
            else:
                st.warning("Selecciona al menos un año para entrenamiento diferente al de validación.")
                val_year = None

            # ==========================================
            # 3. PROCESO DE OPTIMIZACIÓN CRUZADA
            # ==========================================
            if st.button("🚀 Iniciar Validación Cruzada") and val_year:
                
                # Preparar datos de entrenamiento (Data Augmentation local)
                data_train = []
                for y in train_years:
                    # Limpieza Clima
                    dm = dict_meteo[y].copy().dropna(how='all')
                    dm.columns = dm.columns.astype(str).str.strip()
                    # Normalización de nombres de columnas
                    dm = dm.rename(columns={'Tmax': 'TMAX', 'Tmin': 'TMIN', 'prec': 'Prec', 'Prec': 'Prec'})
                    dm['Fecha'] = pd.to_datetime(dm.iloc[:, 0])
                    dm['Julian_days'] = dm['Fecha'].dt.dayofyear
                    dm['P21'] = dm['Prec'].rolling(21, min_periods=1).sum()
                    
                    # Limpieza Campo
                    dc = dict_campo[y].copy().dropna(how='all')
                    dc.columns = dc.columns.astype(str).str.strip()
                    # Buscar columna de plantas/densidad
                    p_col = [c for c in dc.columns if any(x in c.lower() for x in ['plant', 'prom', 'pl.m-2'])][0]
                    dc['ER_obs'] = dc[p_col] / dc[p_col].max()
                    dc['FECHA'] = pd.to_datetime(dc.iloc[:, 0])
                    
                    data_train.append({'m': dm, 'c': dc})

                # 

                # Cargar Pesos Originales (Semilla)
                try:
                    x0 = np.concatenate([np.load('IW.npy').flatten(), np.load('bias_IW.npy'), 
                                         np.load('LW.npy').flatten(), [np.load('bias_out.npy')]])
                except:
                    x0 = np.random.uniform(-0.1, 0.1, 331)

                def objective(p):
                    total_error = 0
                    for ds in data_train:
                        X = ds['m'][["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                        # LLAMADA CORREGIDA (Uso de argumentos por posición para evitar TypeError)
                        preds = run_ann(X, p)
                        # Filtros biológicos base
                        preds[(ds['m']['P21'] < 15) | (ds['m']['Julian_days'] <= 10)] = 0
                        
                        y_p_adj = []
                        for f_o in ds['c']['FECHA']:
                            mask = (ds['m']['Fecha'] >= f_o - pd.Timedelta(days=3)) & \
                                   (ds['m']['Fecha'] <= f_o + pd.Timedelta(days=3))
                            y_p_adj.append(preds[mask].max() if any(mask) else 0)
                        
                        total_error += np.mean((ds['c']['ER_obs'].values - np.array(y_p_adj))**2)
                    return total_error / len(data_train)

                with st.spinner(f"Entrenando modelo con {len(train_years)} campañas..."):
                    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50})
                    st.success("✅ Entrenamiento multi-año completado.")

                    # --- TEST EN AÑO CIEGO (VALIDACIÓN) ---
                    dm_v = dict_meteo[val_year].copy()
                    dm_v.columns = dm_v.columns.astype(str).str.strip()
                    dm_v = dm_v.rename(columns={'Tmax': 'TMAX', 'Tmin': 'TMIN', 'prec': 'Prec', 'Prec': 'Prec'})
                    dm_v['Fecha'] = pd.to_datetime(dm_v.iloc[:, 0])
                    dm_v['Julian_days'] = dm_v['Fecha'].dt.dayofyear
                    dm_v['P21'] = dm_v['Prec'].rolling(21, min_periods=1).sum()
                    
                    X_v = dm_v[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
                    
                    # CORRECCIÓN FINAL DEL TYPEERROR:
                    # Se eliminó X_matrix=X_val para usar X_v por posición
                    p_blind = run_ann(X_v, res.x)
                    p_blind[(dm_v['P21'] < 15) | (dm_v['Julian_days'] <= 10)] = 0
                    
                    # Preparar datos de campo de validación
                    dc_v = dict_campo[val_year].copy()
                    dc_v.columns = dc_v.columns.astype(str).str.strip()
                    p_col_v = [c for c in dc_v.columns if any(x in c.lower() for x in ['plant', 'prom', 'pl.m-2'])][0]
                    dc_v['ER_obs'] = dc_v[p_col_v] / dc_v[p_col_v].max()
                    dc_v['FECHA'] = pd.to_datetime(dc_v.iloc[:, 0])
                    
                    y_p_v = []
                    for f_o in dc_v['FECHA']:
                        mask = (dm_v['Fecha'] >= f_o - pd.Timedelta(days=3)) & \
                               (dm_v['Fecha'] <= f_o + pd.Timedelta(days=3))
                        y_p_v.append(p_blind[mask].max() if any(mask) else 0)
                    
                    # Métrica de Eficiencia (NSE)
                    y_o_v = dc_v['ER_obs'].values
                    y_p_v = np.array(y_p_v)
                    nse = 1 - (np.sum((y_o_v - y_p_v)**2) / np.sum((y_o_v - np.mean(y_o_v))**2))

                    # VISUALIZACIÓN DE RESULTADOS
                    st.divider()
                    st.header(f"📊 Resultado Validación en Año Ciego: {val_year}")
                    st.metric("NSE (Capacidad Predictiva)", f"{nse:.3f}")

                    # 
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(dm_v['Fecha'], p_blind, label='Modelo (Predicción Ciega)', color='#7e22ce', lw=2)
                    ax.scatter(dc_v['FECHA'], dc_v['ER_obs'], color='#ef4444', s=100, label='Campo (Realidad)')
                    ax.set_title(f"Validación Cruzada - Año de Prueba: {val_year}")
                    ax.set_ylabel("Emergencia Relativa (0-1)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    if nse > 0.5:
                        st.balloons()
                        st.success("🚀 El modelo demuestra una alta confiabilidad para predecir años futuros.")
                    else:
                        st.warning("⚠️ El ajuste es bajo para este año. Considera revisar la calidad de los datos climáticos.")

    except Exception as e:
        st.error(f"Error técnico durante el proceso: {e}")
