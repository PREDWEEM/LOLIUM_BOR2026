
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.7 — LOLIUM BORDENAVE 2026
# Actualización:
# - Pearson por intervalos de monitoreo
# - Desfase temporal automático admisible hasta ±10 días
# - PEC calculado estrictamente hasta el día de control
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM BORDENAVE vK4.7",
    layout="wide",
    page_icon="🌾"
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] {
        background-color: #dcfce7;
        border-right: 1px solid #bbf7d0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #166534 !important;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bio-alert {
        padding: 10px;
        border-radius: 5px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ Y ARCHIVOS (MOCKS)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100)**2)/600)
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200)
        p3 = np.exp(-((jd - 230)**2)/1500)
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN + BIO)
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base:
        return 0.0
    elif t <= t_opt:
        return t - t_base
    elif t < t_crit:
        return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else:
        return 0.0

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        z1 = Xn @ self.IW + self.bIW
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T).flatten() + self.bLW
        emerrel = (np.tanh(z2) + 1) / 2
        emer_ac = np.cumsum(emerrel)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(
            np.load(BASE / "IW.npy"),
            np.load(BASE / "bias_IW.npy"),
            np.load(BASE / "LW.npy"),
            np.load(BASE / "bias_out.npy")
        )
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def load_data(file_uploader, default_name):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file_uploader)
    elif (BASE / f"{default_name}.csv").exists():
        return pd.read_csv(BASE / f"{default_name}.csv")
    elif (BASE / f"{default_name}.xlsx").exists():
        return pd.read_excel(BASE / f"{default_name}.xlsx")
    return None

def build_shifted_interval_series(df_sim, df_campo, col_fecha, shift_days):
    """
    Integra la emergencia simulada por intervalos de monitoreo
    aplicando un desfase temporal a la simulación.
    """
    sim_intervals = []
    last_date = df_sim["Fecha"].min() - pd.Timedelta(days=1)

    for _, row in df_campo.iterrows():
        current_date = row[col_fecha]

        start_shifted = last_date + pd.Timedelta(days=shift_days)
        end_shifted = current_date + pd.Timedelta(days=shift_days)

        mask_intervalo = (df_sim["Fecha"] > start_shifted) & (df_sim["Fecha"] <= end_shifted)
        suma_simulada = df_sim.loc[mask_intervalo, "EMERREL"].sum()
        sim_intervals.append(suma_simulada)

        last_date = current_date

    return np.array(sim_intervals, dtype=float)

def evaluate_shifted_validation(df_sim, df_campo, col_fecha, col_plm2, max_shift_days=10):
    """
    Busca el mejor desfase temporal entre -max_shift_days y +max_shift_days.
    Criterio:
      1) máximo Pearson
      2) si empata, mínimo valor absoluto del shift
    """
    obs = df_campo[col_plm2].to_numpy(dtype=float)

    best = {
        "shift_days": 0,
        "pearson_r": -np.inf,
        "sim_intervalo": np.zeros(len(df_campo))
    }

    for shift in range(-max_shift_days, max_shift_days + 1):
        sim_vals = build_shifted_interval_series(df_sim, df_campo, col_fecha, shift)

        pearson_r = pd.Series(obs).corr(pd.Series(sim_vals))
        if pd.isna(pearson_r):
            pearson_r = -1.0

        is_better = False
        if pearson_r > best["pearson_r"]:
            is_better = True
        elif np.isclose(pearson_r, best["pearson_r"], atol=1e-9) and abs(shift) < abs(best["shift_days"]):
            is_better = True

        if is_better:
            best = {
                "shift_days": shift,
                "pearson_r": float(pearson_r),
                "sim_intervalo": sim_vals.copy()
            }

    if best["pearson_r"] == -np.inf:
        best["pearson_r"] = 0.0

    return best

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_BOR2026/main/logo.png", use_container_width=True)
st.sidebar.markdown("## 📂 1. Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima (bordenave)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

df_meteo_raw = load_data(archivo_meteo, "bordenave")
df_campo_raw = load_data(archivo_campo, "bordenave_campo")

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 20)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2:
    t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)

t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input(
    "TT Control Post-emergente (°Cd)",
    value=600,
    step=10,
    help="Grados-día a acumular desde el primer pico."
)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800, step=10)

st.sidebar.markdown("## 🧪 3. Validación")
max_desfase_validacion = st.sidebar.slider(
    "Desfase máximo admisible (días)",
    min_value=0,
    max_value=10,
    value=10,
    help="Permite buscar automáticamente el mejor corrimiento temporal entre simulación y campo."
)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO (BORDENAVE vK4.7)
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:

    # --- PREPROCESAMIENTO CLIMA ---
    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={
        'FECHA': 'Fecha',
        'DATE': 'Fecha',
        'TMAX': 'TMAX',
        'TMIN': 'TMIN',
        'PREC': 'Prec',
        'LLUVIA': 'Prec'
    })
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # --- PREPROCESAMIENTO CAMPO ---
    df_campo = None
    col_fecha = None
    col_plm2 = None

    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)

        max_plm2 = df_campo[col_plm2].max()
        df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2 if max_plm2 > 0 else 0

    # --- PREDICCIÓN NEURAL (SHIFT +60D) ---
    df["JD_Shifted"] = (df["Julian_days"] + 60).clip(1, 300)
    X = df[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    # --- RESTRICCIÓN HÍDRICA Y RELAJACIÓN ---
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]

    # Ajuste robusto del bloqueo temprano
    df.loc[(df["Julian_days"] <= 15) & (df["Prec_sum_21d"] <= 50), "EMERREL"] = 0.0

    # --- BIO-TÉRMICO Y VENTANA DE CONTROL ---
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values:
        fecha_hoy = df['Fecha'].max()

    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()

    dga_hoy, dga_7dias = 0.0, 0.0
    fecha_inicio_ventana, fecha_control = None, None
    msg_estado = "Esperando pico de emergencia..."
    dias_stress = 0

    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()

        df_control = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_control.empty:
            fecha_control = df_control.iloc[0]["Fecha"]

        dga_hoy = df.loc[
            (df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy),
            "DG"
        ].sum()

        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        if idx_hoy + 8 <= len(df):
            dga_7dias = dga_hoy + df.iloc[idx_hoy + 1: idx_hoy + 8]["DG"].sum()
        else:
            dga_7dias = dga_hoy

        msg_estado = f"Pico detectado el {fecha_inicio_ventana.strftime('%d/%m')}"
        dias_stress = len(df_desde_pico[df_desde_pico["Tmedia"] > t_opt_max])

    # --- MÉTRICAS DE VALIDACIÓN SOBRE DATOS REALES DE CAMPO ---
    pearson_r = 0.0
    best_shift_days = 0
    pec, peak_lag, lead_time = 0.0, 0, 0

    if df_campo is not None:
        best_val = evaluate_shifted_validation(
            df_sim=df,
            df_campo=df_campo,
            col_fecha=col_fecha,
            col_plm2=col_plm2,
            max_shift_days=max_desfase_validacion
        )

        best_shift_days = best_val["shift_days"]
        pearson_r = best_val["pearson_r"]
        df_campo["Sim_Intervalo"] = best_val["sim_intervalo"]

        if fecha_control:
            malezas_totales_campo = df_campo[col_plm2].sum()

            # PEC estrictamente hasta el día de control
            malezas_controladas_efectivamente = df_campo.loc[
                df_campo[col_fecha] <= fecha_control,
                col_plm2
            ].sum()

            pec = (
                (malezas_controladas_efectivamente / malezas_totales_campo) * 100
                if malezas_totales_campo > 0 else 0
            )

            idx_pico_campo = df_campo[col_plm2].idxmax()
            fecha_pico_campo = df_campo.loc[idx_pico_campo, col_fecha]
            peak_lag = (fecha_control - fecha_pico_campo).days

            df_alertas = df[df['EMERREL'] >= umbral_er]
            fecha_primera_alerta = df_alertas['Fecha'].iloc[0] if not df_alertas.empty else fecha_inicio_ventana
            lead_time = (fecha_control - fecha_primera_alerta).days

    # -----------------------------------------------------
    # VISUALIZACIÓN FRONT-END
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - BORDENAVE 2026")

    colorscale_hard = [
        [0.0, "green"],
        [0.14, "green"],
        [0.15, "yellow"],
        [0.34, "yellow"],
        [0.35, "red"],
        [1.0, "red"]
    ]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values],
        x=df["Fecha"],
        y=["Emergencia"],
        colorscale=colorscale_hard,
        zmin=0,
        zmax=1,
        showscale=False
    ))
    fig_risk.update_layout(
        height=120,
        margin=dict(t=30, b=0, l=10, r=10),
        title="Mapa de Intensidad (Shifted +60d)"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 MONITOR DE DECISIÓN",
        "🌧️ PRECIPITACIONES",
        "📈 ANÁLISIS ESTRATÉGICO",
        "🧪 BIO-CALIBRACIÓN"
    ])

    with tab1:
        if df_campo is not None and fecha_control:
            st.markdown("<p class='metric-header'>🚜 DIAGNÓSTICO DE CONTROL A CAMPO (Recuentos Reales)</p>", unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Control Efectivo (PEC)", f"{pec:.1f}%", "A la fecha de aplicación", delta_color="normal")
            k2.metric("Lag (Desfase)", f"{peak_lag} días", "Vs Pico de Campo", delta_color="off")
            k3.metric("Anticipación", f"{lead_time} días", "Lead Time Logístico", delta_color="normal")
            k4.metric("Pearson (r)", f"{pearson_r:.3f}", "Sincronía por Intervalos")
            k5.metric("Shift óptimo", f"{best_shift_days:+d} d", "Ajuste campo-simulación")
            st.markdown("---")
        elif df_campo is not None:
            st.markdown("<p class='metric-header'>🚜 VALIDACIÓN DE MODELO CON DATOS DE CAMPO</p>", unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            k1.metric("Pearson (r)", f"{pearson_r:.3f}", "Sincronía por Intervalos")
            k2.metric("Shift óptimo", f"{best_shift_days:+d} d", "Ajuste campo-simulación")
            st.markdown("---")

        col_main, col_gauge = st.columns([2, 1])

        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(
                x=df["Fecha"],
                y=df["EMERREL"],
                mode='lines',
                name='Tasa Diaria',
                line=dict(color='#166534', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(22, 101, 52, 0.1)'
            ))
            fig_emer.add_hline(
                y=umbral_er,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Umbral Alerta ({umbral_er})"
            )

            if df_campo is not None:
                fig_emer.add_trace(go.Scatter(
                    x=df_campo[col_fecha],
                    y=df_campo['Campo_Normalizado'],
                    mode='markers+lines',
                    name='Recuentos a Campo',
                    marker=dict(color='#dc2626', size=10, symbol='diamond'),
                    line=dict(color='rgba(220, 38, 38, 0.4)', dash='dot')
                ))

            if fecha_control:
                fig_emer.add_vline(
                    x=fecha_control.timestamp() * 1000,
                    line_dash="dot",
                    line_color="red",
                    line_width=3,
                    annotation_text=f"Control ({dga_optimo}°Cd)",
                    annotation_position="top left",
                    annotation_font=dict(color="red", size=12)
                )
                fin_res = fecha_control + timedelta(days=residualidad)
                fig_emer.add_vrect(
                    x0=fecha_control.timestamp() * 1000,
                    x1=fin_res.timestamp() * 1000,
                    fillcolor="blue",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    annotation_text=f"Protección ({residualidad}d)",
                    annotation_position="top left"
                )

            fig_emer.update_layout(
                title="Dinámica de Emergencia y Momento Crítico",
                height=400,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico detectado)")
                if fecha_control:
                    st.error(f"🎯 **MOMENTO CRÍTICO DE CONTROL:** {fecha_control.strftime('%d-%m-%Y')}. Se acumularon **{dga_optimo} °Cd** post-emergencia.")
                else:
                    st.info(f"⏳ **En Progreso:** Aún no se han acumulado los {dga_optimo} °Cd requeridos para el control.")
            else:
                st.warning(f"⏳ Esperando primera alerta (Tasa diaria >= {umbral_er}).")

        with col_gauge:
            max_axis = dga_critico * 1.2
            fig_gauge = go.Figure()
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=dga_hoy,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>TT ACUMULADO (°Cd)</b>", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [None, max_axis]},
                    'bar': {'color': "#1e293b", 'thickness': 0.3},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#4ade80"},
                        {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                        {'range': [dga_critico, max_axis], 'color': "#f87171"}
                    ],
                    'threshold': {
                        'line': {'color': "#2563eb", 'width': 6},
                        'thickness': 0.8,
                        'value': dga_7dias
                    }
                }
            ))
            fig_gauge.add_annotation(
                x=0.5,
                y=-0.1,
                text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>",
                showarrow=False,
                font=dict(size=14, color="#1e3a8a"),
                align="center"
            )
            fig_gauge.update_layout(height=350, margin=dict(t=80, b=50, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.header("🌧️ Dinámica de Precipitaciones Diarias")
        fig_prec = go.Figure()
        fig_prec.add_trace(go.Bar(
            x=df["Fecha"],
            y=df["Prec"],
            name='Lluvia Diaria (mm)',
            marker_color='#60a5fa',
            opacity=0.8
        ))
        fig_prec.update_layout(
            title="Precipitación Diaria Registrada",
            xaxis_title="Fecha",
            yaxis_title="Milímetros (mm)",
            height=400
        )
        st.plotly_chart(fig_prec, use_container_width=True)

    with tab3:
        st.header("🔍 Clasificación DTW")
        fecha_corte = pd.Timestamp("2026-05-01")
        df_obs = df[df["Fecha"] < fecha_corte].copy()
        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)
            dists = []
            for m in cluster_model["curves_interp"]:
                m_slice = m[JD_COM <= jd_corte]
                m_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
                dists.append(dtw_distance(obs_norm, m_norm))
            pred = int(np.argmin(dists))
            cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}
            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(
                    x=JD_COM,
                    y=cluster_model["curves_interp"][pred],
                    name="Patrón Histórico",
                    line=dict(dash='dash', color=cols.get(pred))
                ))
                fp.add_trace(go.Scatter(
                    x=jd_grid,
                    y=obs_norm * cluster_model["curves_interp"][pred].max(),
                    name="2026",
                    line=dict(color='black', width=3)
                ))
                st.plotly_chart(fp, use_container_width=True)
            with c2:
                nombres_patrones = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
                st.success(f"### {nombres_patrones.get(pred, 'Desconocido')}")
                st.metric("DTW Score", f"{min(dists):.2f}")
        else:
            st.info("Datos insuficientes para clasificación DTW.")

    with tab4:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]
        fig_bio = go.Figure()
        fig_bio.add_trace(go.Scatter(
            x=x_temps,
            y=y_tt,
            mode='lines',
            line=dict(color='#2563eb', width=4),
            fill='tozeroy'
        ))
        st.plotly_chart(fig_bio, use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')

        if df_campo is not None:
            df_campo.to_excel(writer, index=False, sheet_name='Campo_Validacion')

            resumen_val = {
                'Métrica': [
                    'PEC (%)',
                    'Lag (días)',
                    'Lead Time (días)',
                    'Pearson (r)',
                    'Shift óptimo (días)'
                ],
                'Valor': [
                    pec,
                    peak_lag,
                    lead_time,
                    pearson_r,
                    best_shift_days
                ]
            }
            pd.DataFrame(resumen_val).to_excel(writer, sheet_name='Validacion_Campo', index=False)

    st.sidebar.download_button(
        "📥 Descargar Reporte Completo",
        output.getvalue(),
        "PREDWEEM_Integral_Bordenave_vK4_7.xlsx"
    )

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue datos climáticos para comenzar.")
