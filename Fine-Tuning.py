
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="PREDWEEM - Optimizador Maestro", layout="wide")

# ==========================================
# 1. ARQUITECTURA DE LA RED NEURONAL
# ==========================================
def run_ann(X_input, iw, biw, lw, blw):
    # Escalamiento original del proyecto
    in_min = np.array([1, 0, -7, 0])
    in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
    
    emer_rel = []
    for x in X_n:
        a1 = np.tanh(iw.T @ x + biw)
        z2 = np.dot(lw, a1) + blw
        # Salida Tanh escalada a [0, 1]
        emer_rel.append((np.tanh(z2) + 1) / 2)
    
    return np.diff(np.cumsum(np.array(emer_rel).flatten()), prepend=0)

# ==========================================
# 2. INTERFAZ DE USUARIO
# ==========================================
st.title("🧠 PREDWEEM: Optimizador Global y Diagnóstico")
st.markdown("Si la emergencia es 0, ajusta los **Filtros de Seguridad** en el panel lateral.")

# --- SIDEBAR: CONTROLES CRÍTICOS ---
st.sidebar.header("🛡️ Filtros de Seguridad (Biología)")
u_hidrico = st.sidebar.slider("Umbral Hídrico (mm en 21d)", 0, 40, 15, 
                               help="Si la lluvia es menor a esto, el modelo fuerza 0.")
d_inicio = st.sidebar.slider("Día de Inicio (Julian)", 1, 60, 10,
                              help="Día del año antes del cual no hay emergencia (Dormición).")

st.sidebar.header("⚙️ Configuración de Optimización")
max_iter = st.sidebar.number_input("Iteraciones (BFGS)", value=30)

# Carga de archivos
st.subheader("📂 Entrada de Datos")
c1, c2 = st.columns(2)
with c1:
    f_meteo = st.file_uploader("Clima (CSV)", type=['csv'])
with c2:
    f_valida = st.file_uploader("Campo (Excel/CSV)", type=['xlsx', 'csv'])

# ==========================================
# 3. PROCESAMIENTO Y DIAGNÓSTICO
# ==========================================
if f_meteo and f_valida:
    try:
        # Cargar datos
        df_m = pd.read_csv(f_meteo)
        df_m.columns = df_m.columns.str.strip()
        df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
        df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
        df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()
        
        if f_valida.name.endswith('.csv'):
            df_c = pd.read_csv(f_valida)
        else:
            df_c = pd.read_excel(f_valida, engine='openpyxl')
        df_c.columns = df_c.columns.str.strip()
        df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
        df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

        # Cargar pesos originales (Deben estar en el repo de GitHub)
        IW = np.load('IW.npy'); bIW = np.load('bias_IW.npy')
        LW = np.load('LW.npy'); bLW = np.load('bias_out.npy')
        
        X_mat = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)

        # --- DIAGNÓSTICO INICIAL ---
        preds_raw = run_ann(X_mat, IW, bIW, LW, bLW)
        
        # Aplicar filtros del sidebar
        preds_filt = preds_raw.copy()
        preds_filt[(df_m['Prec_sum'] < u_hidrico) | (df_m['Julian_days'] <= d_inicio)] = 0.0

        st.subheader("🔍 Diagnóstico de Emergencia")
        if preds_filt.max() == 0:
            st.error(f"🚨 LA EMERGENCIA SIGUE EN 0. Motivo: La lluvia máxima detectada es {df_m['Prec_sum'].max():.1f} mm y tu umbral es {u_hidrico} mm.")
        else:
            st.success(f"✅ Emergencia activa. Pico detectado: {preds_filt.max():.4f}")

        # Gráfico de Diagnóstico
        fig_diag, ax_diag = plt.subplots(figsize=(10, 3))
        ax_diag.plot(df_m['Fecha'], preds_raw, label="Salida Bruta RNA", color='orange', alpha=0.5)
        ax_diag.plot(df_m['Fecha'], preds_filt, label="Salida con Filtros", color='green')
        ax_diag.set_title("Comparación: RNA Bruta vs Filtros de Seguridad")
        ax_diag.legend()
        st.pyplot(fig_diag)

        # ==========================================
        # 4. BOTÓN DE OPTIMIZACIÓN GLOBAL
        # ==========================================
        if st.button("🚀 Optimizar todos los Pesos y Desvíos"):
            with st.spinner("Calibrando 331 parámetros..."):
                
                def objective(p):
                    # Reconstrucción del vector de parámetros
                    ptr = 0
                    niw = p[ptr:ptr+220].reshape(4, 55); ptr += 220
                    nbiw = p[ptr:ptr+55]; ptr += 55
                    nlw = p[ptr:ptr+55].reshape(1, 55); ptr += 55
                    nblw = p[ptr]
                    
                    preds = run_ann(X_mat, niw, nbiw, nlw, nblw)
                    preds[(df_m['Prec_sum'] < u_hidrico) | (df_m['Julian_days'] <= d_inicio)] = 0.0
                    
                    # Sincronización con ventana de 7 días
                    y_p_adj = []
                    for f_o in df_c['FECHA']:
                        mask = (df_m['Fecha'] >= f_o - pd.Timedelta(days=3)) & \
                               (df_m['Fecha'] <= f_o + pd.Timedelta(days=3))
                        y_p_adj.append(preds[mask].max() if any(mask) else 0)
                    
                    return np.mean((df_c['ER_obs'].values - np.array(y_p_adj))**2)

                x0 = np.concatenate([IW.flatten(), bIW, LW.flatten(), [bLW]])
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter})
                
                # Reconstruir y Guardar
                ptr = 0
                opt_iw = res.x[ptr:ptr+220].reshape(4, 55); ptr += 220
                opt_biw = res.x[ptr:ptr+55]; ptr += 55
                opt_lw = res.x[ptr:ptr+55].reshape(1, 55); ptr += 55
                opt_blw = res.x[ptr]

                st.success(f"✅ Calibración terminada. MSE final: {res.fun:.5f}")

                # Gráfico final
                p_final = run_ann(X_mat, opt_iw, opt_biw, opt_lw, opt_blw)
                p_final[(df_m['Prec_sum'] < u_hidrico) | (df_m['Julian_days'] <= d_inicio)] = 0.0
                
                fig_fin, ax_fin = plt.subplots(figsize=(10, 4))
                ax_fin.plot(df_m['Fecha'], p_final, label='Optimizado', color='blue')
                ax_fin.scatter(df_c['FECHA'], df_c['ER_obs'], color='red', label='Campo')
                ax_fin.legend()
                st.pyplot(fig_fin)

                # BOTONES DE DESCARGA
                st.subheader("💾 Descargar Nuevos Pesos")
                col_d = st.columns(4)
                def d_btn(label, data, fname):
                    buf = io.BytesIO(); np.save(buf, data)
                    return st.download_button(label, buf.getvalue(), fname)
                
                with col_d[0]: d_btn("IW", opt_iw, "IW_global.npy")
                with col_d[1]: d_btn("bIW", opt_biw, "bias_IW_global.npy")
                with col_d[2]: d_btn("LW", opt_lw, "LW_global.npy")
                with col_d[3]: d_btn("bOut", opt_blw, "bias_out_global.npy")

    except Exception as e:
        st.error(f"Error: {e}")
