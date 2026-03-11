import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="PREDWEEM - Analizador de Desempeño", layout="wide")

# ==========================================
# 1. MOTOR RNA VECTORIZADO
# ==========================================
def run_ann_fast(X_n, params):
    ptr = 0
    iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
    biw = params[ptr:ptr+55]; ptr += 55
    lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
    blw = params[ptr]
    
    a1 = np.tanh(X_n @ iw + biw)
    z2 = a1 @ lw.T + blw
    return (np.tanh(z2).flatten() + 1) / 2

# ==========================================
# 2. FUNCIONES DE MÉTRICAS CIENTÍFICAS
# ==========================================
def calcular_metricas(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((obs - pred)**2))
    return nse, rmse

def safe_get_col(df, keywords, default_idx=1):
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in df.columns:
        if any(k in col for k in keywords):
            return col
    return df.columns[default_idx]

# ==========================================
# 3. INTERFAZ Y CARGA DE PESOS
# ==========================================
st.title("📊 PREDWEEM: Analizador de Calidad del Modelo")
st.markdown("Simulación masiva utilizando los pesos y desvíos actuales (archivos .npy).")

# Intentar cargar pesos actuales
try:
    x_actual = np.concatenate([
        np.load('IW.npy').flatten(), 
        np.load('bias_IW.npy'), 
        np.load('LW.npy').flatten(), 
        [np.load('bias_out.npy')]
    ])
    st.sidebar.success("✅ Pesos actuales cargados correctamente.")
except:
    st.sidebar.error("❌ No se encontraron los archivos .npy. Usando pesos aleatorios.")
    x_actual = np.random.uniform(-0.1, 0.1, 331)

ventana = st.sidebar.slider("Ventana de Tolerancia (días)", 0, 10, 7)
u_lluvia = st.sidebar.slider("Umbral Hídrico (mm)", 5, 30, 15)

# Carga de archivos maestros
c1, c2 = st.columns(2)
f_meteo = c1.file_uploader("Excel Clima (Multi-hoja)", type=['xlsx'])
f_campo = c2.file_uploader("Excel Campo (Multi-hoja)", type=['xlsx'])

if f_meteo and f_campo:
    dict_m = pd.read_excel(f_meteo, sheet_name=None)
    dict_c = pd.read_excel(f_campo, sheet_name=None)
    años = sorted(list(set(dict_m.keys()) & set(dict_c.keys())))

    if st.button("▶️ Ejecutar Simulación Histórica"):
        resumen = []
        in_min = np.array([1, 0, -7, 0]); in_max = np.array([300, 41, 25.5, 84])
        
        # Procesar cada año
        for y in años:
            # 1. Preparar Clima
            m = dict_m[y].copy().dropna(how='all')
            tmax_c = safe_get_col(m, ['tmax'], 1)
            tmin_c = safe_get_col(m, ['tmin'], 2)
            prec_c = safe_get_col(m, ['prec'], 3)
            
            m['fecha_dt'] = pd.to_datetime(m.iloc[:,0])
            m['julian'] = m['fecha_dt'].dt.dayofyear
            X_input = m[['julian', tmax_c, tmin_c, prec_c]].to_numpy(float)
            X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
            
            # 2. Simulación
            preds = run_ann_fast(X_n, x_actual)
            p21 = m[prec_c].rolling(21, min_periods=1).sum().values
            preds[(p21 < u_lluvia) | (m['julian'] <= 25)] = 0
            
            # 3. Preparar Campo
            c = dict_c[y].copy().dropna(how='all')
            p_col = safe_get_col(c, ['plant', 'prom', 'pl.m-2'], 1)
            obs_dates = pd.to_datetime(c.iloc[:, 0])
            obs_vals = (c[p_col] / c[p_col].max()).values
            meteo_dates = m['fecha_dt'].values
            
            # 4. Ajuste por ventana para métricas
            y_p_adj = []
            for d in obs_dates:
                mask = (meteo_dates >= (d - pd.Timedelta(days=ventana)).to_datetime64()) & \
                       (meteo_dates <= (d + pd.Timedelta(days=ventana)).to_datetime64())
                y_p_adj.append(preds[mask].max() if any(mask) else 0)
            
            # 5. Calcular Métricas
            nse, rmse = calcular_metricas(obs_vals, y_p_adj)
            resumen.append({"Año": y, "NSE": nse, "RMSE": rmse})
            
            # 6. Gráfico Individual
            with st.expander(f"📈 Gráfico Campaña {y}"):
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(m['fecha_dt'], preds, label='Simulación', color='purple', alpha=0.7)
                ax.scatter(obs_dates, obs_vals, color='red', label='Observado')
                ax.set_title(f"Ajuste {y} (NSE: {nse:.3f})")
                ax.legend()
                st.pyplot(fig)

        # TABLA FINAL DE RESULTADOS
        st.divider()
        st.subheader("📋 Resumen de Métricas de Ajuste")
        df_res = pd.DataFrame(resumen)
        
        col_res1, col_res2 = st.columns([2, 1])
        col_res1.dataframe(df_res.style.background_gradient(subset=['NSE'], cmap='RdYlGn', vmin=0, vmax=1))
        
        # Métrica promedio global
        nse_avg = df_res['NSE'].mean()
        col_res2.metric("NSE Promedio Global", f"{nse_avg:.3f}")
        
        if nse_avg > 0.6:
            st.balloons()
            st.success("El modelo tiene un desempeño robusto a través de los años.")
        else:
            st.warning("El ajuste promedio es bajo. Se recomienda una re-calibración global.")
