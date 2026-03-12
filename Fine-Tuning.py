
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import io
import zipfile  # <-- Agregado para empaquetar en ZIP

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="PREDWEEM - Optimizador Robusto", layout="wide")

# ==========================================
# 1. ARQUITECTURA DE LA RED NEURONAL
# ==========================================
def run_ann(X_input, params):
    # Reconstrucción de la arquitectura (331 pesos + 1 factor de escala)
    ptr = 0
    iw = params[ptr:ptr+220].reshape(4, 55); ptr += 220
    biw = params[ptr:ptr+55]; ptr += 55
    lw = params[ptr:ptr+55].reshape(1, 55); ptr += 55
    blw = params[ptr]; ptr += 1
    escala = params[ptr] # Factor de ajuste de magnitud

    # Escalamiento original
    in_min = np.array([1, 0, -7, 0]); in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
    
    emer = []
    for x in X_n:
        a1 = np.tanh(iw.T @ x + biw)
        z2 = np.dot(lw, a1) + blw
        emer.append((np.tanh(z2) + 1) / 2)
    
    return np.diff(np.cumsum(np.array(emer).flatten()), prepend=0) * escala

# ==========================================
# 2. INTERFAZ DE USUARIO
# ==========================================
st.title("🔬 PREDWEEM: Optimizador Global Robusto")
st.sidebar.header("🛠️ Configuración del Algoritmo")
iteraciones = st.sidebar.slider("Intensidad de Búsqueda", 5, 50, 15, help="Más alto = mejor ajuste pero más lento.")

# Cargadores de archivos
col1, col2 = st.columns(2)
with col1: f_meteo = st.file_uploader("Clima (CSV)", type=['csv'])
with col2: f_valida = st.file_uploader("Campo (Excel/CSV)", type=['xlsx', 'csv'])

# ==========================================
# 3. LÓGICA DE OPTIMIZACIÓN
# ==========================================
if f_meteo and f_valida:
    try:
        # Cargar datos con limpieza de errores previos
        df_m = pd.read_csv(f_meteo)
        df_m.columns = df_m.columns.str.strip()
        df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
        df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
        df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()
        
        df_c = pd.read_excel(f_valida) if f_valida.name.endswith('.xlsx') else pd.read_csv(f_valida)
        df_c.columns = df_c.columns.str.strip()
        df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
        df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

        # MANEJO SEGURO DE ARCHIVOS .NPY
        try:
            x0 = np.concatenate([np.load('IW.npy').flatten(), np.load('bias_IW.npy'), 
                                 np.load('LW.npy').flatten(), [np.load('bias_out.npy')], [1.0]])
        except:
            st.warning("⚠️ No se encontraron pesos originales. Inicializando pesos neutros.")
            x0 = np.zeros(332) # 331 pesos + 1 factor escala

        X_mat = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)

        if st.button("🚀 Ejecutar Optimización (Evolución Diferencial)"):
            with st.spinner("Buscando la mejor configuración global... esto puede demorar 1-2 minutos."):
                
                # Definir límites (bounds) para evitar que los pesos exploten
                bounds = [(-2, 2)] * 331 + [(0.1, 5.0)] # Pesos entre -2 y 2, escala entre 0.1 y 5

                def objective(p):
                    preds = run_ann(X_mat, p)
                    # Filtros dinámicos (Clave para el sitio 2)
                    preds[(df_m['Prec_sum'] < 10) | (df_m['Julian_days'] <= 25)] = 0.0
                    
                    y_p_adj = []
                    for f_o in df_c['FECHA']:
                        mask = (df_m['Fecha'] >= f_o - pd.Timedelta(days=3)) & (df_m['Fecha'] <= f_o + pd.Timedelta(days=3))
                        y_p_adj.append(preds[mask].max() if any(mask) else 0)
                    
                    return np.mean((df_c['ER_obs'].values - np.array(y_p_adj))**2)

                # El optimizador diferencial es mucho más robusto que BFGS
                res = differential_evolution(objective, bounds, maxiter=iteraciones, popsize=5, polish=True)
                
                # Resultados
                st.success(f"✅ Optimización terminada con éxito. Error (MSE): {res.fun:.5f}")
                
                # Visualización
                p_final = run_ann(X_mat, res.x)
                p_final[(df_m['Prec_sum'] < 10) | (df_m['Julian_days'] <= 25)] = 0.0
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_m['Fecha'], p_final, label='Modelo Calibrado', color='blue', lw=2)
                ax.scatter(df_c['FECHA'], df_c['ER_obs'], color='red', s=100, label='Campo Real')
                ax.set_title("Ajuste Global Finalizado (Algoritmo Evolutivo)")
                ax.legend()
                st.pyplot(fig)

                # --- NUEVA SECCIÓN DE DESCARGA EN ZIP ---
                st.subheader("💾 Descargar Pesos Finales")
                
                # Crear un buffer en memoria para el ZIP
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    ptr = 0
                    
                    # Función auxiliar para guardar arrays como .npy dentro del ZIP
                    def add_to_zip(name, data):
                        buf = io.BytesIO()
                        np.save(buf, data)
                        zip_file.writestr(name, buf.getvalue())

                    # Extraer parámetros optimizados y empaquetarlos
                    add_to_zip("IW_opt.npy", res.x[ptr:ptr+220].reshape(4, 55)); ptr += 220
                    add_to_zip("bIW_opt.npy", res.x[ptr:ptr+55]); ptr += 55
                    add_to_zip("LW_opt.npy", res.x[ptr:ptr+55].reshape(1, 55)); ptr += 55
                    add_to_zip("bOut_opt.npy", np.array(res.x[ptr])) # Se guarda como array por seguridad
                
                # Botón único de descarga del archivo ZIP
                st.download_button(
                    label="📦 Descargar todos los Pesos y Desvíos (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="pesos_optimizados_predweem.zip",
                    mime="application/zip"
                )

    except Exception as e:
        st.error(f"Error en el proceso: {e}")
