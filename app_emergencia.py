# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026
# - ANN → EMERREL diaria
# - Post-proceso: recorte negativos, suavizado opcional, acumulado
# - Riesgo diario + animación
# - Clasificación funcional K=3 (DTW + K-Medoids) sobre EMERREL
# - Interpretación agronómica detallada por patrón (Temprano / Bimodal / Tardío)
# - Fuente de datos FIJA: meteo_daily.csv
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, requests, xml.etree.ElementTree as ET
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIG STREAMLIT + ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM vK3 – LOLIUM BORDENAVE 2026",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 FUNCIONES SEGURAS
# ===============================================================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ===============================================================
# 🔧 API METEOBAHIA (7 días) — OPCIONAL (no usada en esta versión)
# ===============================================================
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"

def _to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

@st.cache_data(ttl=900)
def fetch_forecast():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    for d in root.findall(".//forecast/tabular/day"):
        fecha  = d.find("fecha").get("value")
        tmax   = d.find("tmax").get("value")
        tmin   = d.find("tmin").get("value")
        prec   = d.find("precip").get("value")
        rows.append({
            "Fecha": pd.to_datetime(fecha),
            "TMAX": _to_float(tmax),
            "TMIN": _to_float(tmin),
            "Prec": _to_float(prec),
        })

    df = pd.DataFrame(rows).sort_values("Fecha").head(7)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df

# ===============================================================
# 🔧 ANN — Modelo de predicción emergencia
# ===============================================================
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        # rango de entrenamiento original
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        """
        Devuelve EMERREL cruda de la ANN y EMERAC cruda (cumsum).
        El post-procesamiento se hace por fuera.
        """
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1) / 2    # 0–1 (diario, crudo)
        emer_ac = np.cumsum(emer)          # acumulada cruda
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_ann():
    IW  = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW  = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

modelo_ann = safe(lambda: load_ann(), "Error cargando pesos ANN")
if modelo_ann is None:
    st.stop()

# ===============================================================
# 🔧 POST-PROCESO EMERGENCIA (suavizado + recorte, SIN reescalar a 1)
# ===============================================================
def postprocess_emergence(emerrel_raw,
                          smooth=True,
                          window=3,
                          clip_zero=True):
    """
    Toma EMERREL cruda de la ANN y devuelve:
    - emerrel_proc: EMERREL suavizada / recortada
    - emerac_proc : EMERAC acumulada (no forzada a terminar en 1)
    """
    emer = np.array(emerrel_raw, dtype=float)

    # 1) Recortar posibles negativos
    if clip_zero:
        emer = np.maximum(emer, 0.0)

    # 2) Suavizado por media móvil
    if smooth and len(emer) > 1 and window > 1:
        window = int(window)
        window = max(1, min(window, len(emer)))
        if window > 1:
            kernel = np.ones(window, dtype=float) / window
            emer = np.convolve(emer, kernel, mode="same")

    # 3) EMERAC acumulada
    emerac = np.cumsum(emer)

    return emer, emerac

# ===============================================================
# 🔧 CARGA FIJA DESDE meteo_daily.csv
# ===============================================================
st.title("🌾 PREDWEEM vK3 — LOLIUM BORDENAVE 2026")

with st.sidebar:
    st.header("Ajustes de emergencia")
    use_smoothing = st.checkbox("Suavizar EMERREL", value=True)
    window_size   = st.slider("Ventana de suavizado (días)", min_value=1, max_value=9, value=3, step=1)
    clip_zero     = st.checkbox("Recortar negativos a 0", value=True)

path_daily = BASE / "meteo_daily.csv"
if not path_daily.exists():
    st.error("❌ No se encontró meteo_daily.csv en el directorio de la app.")
    st.stop()

df = pd.read_csv(path_daily, parse_dates=["Fecha"])

# Aseguramos orden y Julian_days
df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
df["Julian_days"] = df["Fecha"].dt.dayofyear

# ---------------------------------------------------------------
# ANN → EMERREL
# ---------------------------------------------------------------
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel_raw, emerac_raw = modelo_ann.predict(X)

emerrel, emerac = postprocess_emergence(
    emerrel_raw,
    smooth=use_smoothing,
    window=window_size,
    clip_zero=clip_zero,
)

df["EMERREL"] = emerrel
df["EMERAC"]  = emerac

dias   = df["Julian_days"].to_numpy()
fechas = df["Fecha"].to_numpy()

# ===============================================================
# 🔥 MAPA DE RIESGO — VERSIÓN MODERNA E INTERACTIVA
# ===============================================================
st.subheader("🔥 Mapa moderno e interactivo de riesgo de emergencia")

# Cálculo del riesgo (0–1 normalizado)
if "Riesgo" not in df.columns:
    max_emerrel = df["EMERREL"].max()
    if max_emerrel > 0:
        df["Riesgo"] = df["EMERREL"] / max_emerrel
    else:
        df["Riesgo"] = 0.0

# Clasificación del nivel de riesgo
if "Nivel_riesgo" not in df.columns:
    def clasificar_riesgo(r):
        if r <= 0.10:
            return "Nulo"
        elif r <= 0.33:
            return "Bajo"
        elif r <= 0.66:
            return "Medio"
        else:
            return "Alto"
    df["Nivel_riesgo"] = df["Riesgo"].apply(clasificar_riesgo)

df_risk = df.copy()
df_risk["Fecha_str"] = df_risk["Fecha"].dt.strftime("%d-%b")

# Día de riesgo máximo
if df_risk["Riesgo"].max() > 0:
    idx_max_riesgo   = df_risk["Riesgo"].idxmax()
    fecha_max_riesgo = df_risk.loc[idx_max_riesgo, "Fecha"]
    valor_max_riesgo = df_risk.loc[idx_max_riesgo, "Riesgo"]
else:
    fecha_max_riesgo = None
    valor_max_riesgo = None

with st.sidebar:
    st.markdown("### 🎨 Estilo del mapa de riesgo")
    cmap = st.selectbox(
        "Mapa de colores",
        ["viridis", "plasma", "cividis", "turbo", "magma", "inferno", "cool", "warm"],
        index=0
    )
    tipo_barra = st.radio(
        "Modo de visualización",
        ["Rectángulo suave (recomendado)", "Barras finas tipo timeline"],
        index=0
    )

# Gráfico principal
if tipo_barra == "Rectángulo suave (recomendado)":
    fig = go.Figure(
        data=go.Heatmap(
            z=[df_risk["Riesgo"].values],
            x=df_risk["Fecha"],
            y=["Riesgo"],
            colorscale=cmap,
            zmin=0, zmax=1,
            showscale=True,
            hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(showticklabels=False)
else:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_risk["Fecha"],
            y=df_risk["Riesgo"],
            marker=dict(color=df_risk["Riesgo"], colorscale=cmap, cmin=0, cmax=1),
            hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1], title="Riesgo")

if fecha_max_riesgo is not None:
    fig.add_annotation(
        x=fecha_max_riesgo,
        y=1.05 if tipo_barra != "Rectángulo suave (recomendado)" else 0.6,
        text=f"⬆ Máximo riesgo ({valor_max_riesgo:.2f})",
        showarrow=False,
        font=dict(size=12, color="red")
    )

fig.update_layout(
    height=250,
    margin=dict(l=30, r=30, t=40, b=20),
    title="Mapa interactivo de riesgo diario de emergencia (0–1)",
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Tabla detallada de riesgo diario"):
    st.dataframe(
        df_risk[["Fecha", "EMERREL", "Riesgo", "Nivel_riesgo"]],
        use_container_width=True
    )

# ===============================================================
# 🎬 ANIMACIÓN DEL RIESGO DE EMERGENCIA DÍA A DÍA
# ===============================================================
st.subheader("🎬 Animación temporal del riesgo de emergencia (día por día)")

df_anim = df.copy()
df_anim["Fecha_str"] = df_anim["Fecha"].dt.strftime("%d-%b")

with st.sidebar:
    cmap_anim = st.selectbox(
        "Mapa de colores para la animación",
        ["viridis", "plasma", "cividis", "turbo", "magma", "inferno", "icefire", "rdbu"],
        index=0,
        key="anim_cmap"
    )

fig_anim = px.scatter(
    df_anim,
    x="Fecha",
    y="Riesgo",
    animation_frame="Fecha_str",
    range_y=[0, 1],
    color="Riesgo",
    color_continuous_scale=cmap_anim,
    size=[12]*len(df_anim),
    hover_data={"Fecha_str": True, "Riesgo": ":.2f"},
    labels={"Fecha": "Fecha calendario", "Riesgo": "Riesgo de emergencia (0–1)"}
)

# Línea base
fig_anim.add_trace(
    go.Scatter(
        x=df_anim["Fecha"],
        y=df_anim["Riesgo"],
        mode="lines",
        line=dict(color="gray", width=1.5),
        name="Riesgo observado"
    )
)

fig_anim.update_layout(
    title="Evolución diaria del riesgo de emergencia",
    height=450,
    margin=dict(l=20, r=20, t=50, b=20),
)

# Control de velocidad
if fig_anim.layout.updatemenus and len(fig_anim.layout.updatemenus) > 0:
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300

st.plotly_chart(fig_anim, use_container_width=True)

# ===============================================================
# 🔍 GRÁFICOS EMERREL / EMERAC (Fecha real)
# ===============================================================
st.subheader("🔍 EMERGENCIA diaria y acumulada — ANN vs post-proceso")

col_er, col_ac = st.columns(2)

# EMERREL: cruda vs procesada
with col_er:
    fig_er, ax_er = plt.subplots(figsize=(5,4))
    ax_er.plot(fechas, emerrel_raw, label="EMERREL cruda (ANN)", color="red", alpha=0.6)
    ax_er.plot(fechas, emerrel,     label="EMERREL procesada",   color="blue", linewidth=2)
    ax_er.set_xlabel("Fecha calendario real")
    ax_er.set_ylabel("EMERREL (fracción diaria)")
    ax_er.set_title("EMERREL: ANN vs post-proceso")
    ax_er.legend()
    fig_er.autofmt_xdate()
    st.pyplot(fig_er)

# EMERAC: cruda vs procesada
with col_ac:
    fig_ac, ax_ac = plt.subplots(figsize=(5,4))

    if emerac_raw[-1] > 0:
        ax_ac.plot(fechas, emerac_raw/emerac_raw[-1],
                   label="EMERAC cruda (normalizada)", color="orange", alpha=0.6)
    else:
        ax_ac.plot(fechas, emerac_raw,
                   label="EMERAC cruda", color="orange", alpha=0.6)

    if emerac[-1] > 0:
        ax_ac.plot(fechas, emerac/emerac[-1],
                   label="EMERAC procesada (normalizada)", color="green", linewidth=2)
    else:
        ax_ac.plot(fechas, emerac,
                   label="EMERAC procesada", color="green", linewidth=2)

    ax_ac.set_xlabel("Fecha calendario real")
    ax_ac.set_ylabel("EMERAC (0–1 relativo al período)")
    ax_ac.set_title("EMERAC: ANN vs post-proceso")
    ax_ac.legend()
    fig_ac.autofmt_xdate()
    st.pyplot(fig_ac)

# ===============================================================
# 🔥 CLASIFICADOR FUNCIONAL K=3 (DTW + K-Medoids)
# ===============================================================
st.header("🌾 Clasificación funcional K=3 basada en curvas EMERREL (DTW)")

# ---------------------------------------------------------------
# Cargar modelo_clusters_k3.pkl
# ---------------------------------------------------------------
def load_k3_model():
    local_path = BASE/"modelo_clusters_k3.pkl"
    alt_path   = Path("/mnt/data/modelo_clusters_k3.pkl")

    if local_path.exists():
        path = local_path
    elif alt_path.exists():
        path = alt_path
    else:
        raise FileNotFoundError("modelo_clusters_k3.pkl no encontrado")

    with open(path, "rb") as f:
        return pickle.load(f)

cluster_model = safe(lambda: load_k3_model(), "Error cargando modelo_clusters_k3.pkl")
if cluster_model is None:
    st.stop()

names_k3      = cluster_model["names"]
labels_k3     = np.array(cluster_model["labels_k3"])
medoids_k3    = cluster_model["medoids_k3"]
DTW_hist      = np.array(cluster_model["DTW_matrix"])
JD_COMMON     = np.array(cluster_model["JD_common"])
curves_interp = np.array(cluster_model["curves_interp"])   # matriz (N, T)

# ---------------------------------------------------------------
# DTW + funciones auxiliares
# ---------------------------------------------------------------
def dtw_distance(a, b):
    """DTW simple para comparar la forma de dos curvas 1D."""
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

def interpolate_curve(jd, y, jd_common):
    """Interpola la curva EMERREL a la grilla JD_COMMON usada en el clustering."""
    return np.interp(jd_common, jd, y)

# Normalizar EMERREL simulada a la misma escala (0–1 por máximo)
if emerrel.max() > 0:
    emerrel_norm = emerrel / emerrel.max()
else:
    emerrel_norm = emerrel.copy()

curve_interp_year = interpolate_curve(dias, emerrel_norm, JD_COMMON)

# Medoides (curvas representativas de cada patrón)
med0 = curves_interp[medoids_k3[0]]   # Patrón 0 — Intermedio/Bimodal
med1 = curves_interp[medoids_k3[1]]   # Patrón 1 — Temprano/Compacto
med2 = curves_interp[medoids_k3[2]]   # Patrón 2 — Tardío/Extendido

# Distancias DTW a cada patrón
d0 = dtw_distance(curve_interp_year, med0)
d1 = dtw_distance(curve_interp_year, med1)
d2 = dtw_distance(curve_interp_year, med2)

dist_vector  = np.array([d0, d1, d2])
cluster_pred = int(np.argmin(dist_vector))

# Mapeo de nombres y colores
cluster_names = {
    0: "🌾 Intermedio / Bimodal",
    1: "🌱 Temprano / Compacto",
    2: "🍂 Tardío / Extendido"
}

cluster_colors = {
    0: "blue",
    1: "green",     # temprano
    2: "orange"     # tardío
}

cluster_desc = {
    0: "Patrón mixto con dos pulsos bien diferenciados: uno temprano moderado y uno otoñal fuerte.",
    1: "Patrón temprano y concentrado, con emergencia dominante en feb–mar y pico marcado antes de abril.",
    2: "Patrón tardío/extenso con emergencia sostenida abril–junio y fuerte cola otoñal."
}

# Resultado principal
st.markdown(f"""
## 🎯 Patrón asignado por análisis funcional K=3:
### <span style='color:{cluster_colors[cluster_pred]}; font-size:30px;'>
{cluster_names[cluster_pred]}
</span>
""", unsafe_allow_html=True)

st.info(cluster_desc[cluster_pred])

# ===============================================================
# 🌱 Descripción agronómica ampliada del patrón
# ===============================================================
st.subheader("🌱 Descripción agronómica ampliada del patrón asignado")

descripcion_agronomica_detallada = {
    1: """
### 🟢 Patrón 1 — Temprano / Compacto
#### Dinámica de emergencia
- Emergencia muy concentrada en 20–35 días.
- Pico marcado entre fines de febrero y mediados de marzo.
- Casi nula emergencia posterior a abril.

#### Implicancias de manejo
- Ventana crítica **muy temprana**.
- Clave el uso de **residuales pre-siembra / pre-emergentes** activos desde fines de febrero.
- Postemergentes pierden eficacia si se aplican después del pico principal.
- Requiere monitoreo intensivo en la primera quincena de marzo.
""",
    0: """
### 🔵 Patrón 0 — Intermedio / Bimodal
#### Dinámica de emergencia
- Dos picos bien definidos: uno temprano (marzo) y otro otoñal (mayo–junio).
- Entre ambos aparece una meseta de baja emergencia.
- Alta variabilidad dentro del grupo.

#### Implicancias de manejo
- Demanda **estrategia en dos tiempos**:
  - Residual o control temprano para el primer pulso.
  - Refuerzo (postemergente o residual de segunda ventana) para el pulso tardío.
- Alta probabilidad de “sobreconfianza” después del primer pico si no se monitorea el segundo.
""",
    2: """
### 🟠 Patrón 2 — Tardío / Extendido
#### Dinámica de emergencia
- Emergencia principal a partir de abril.
- Pico en mayo (a veces junio).
- Cola prolongada hasta julio.

#### Implicancias de manejo
- Los residuales aplicados en febrero–marzo pueden no cubrir la ventana efectiva.
- Requiere **postemergentes escalonados** y monitoreo sostenido en otoño–invierno.
- Aumenta costos de control y presión tardía sobre cultivos de fina tardíos y verdeos.
"""
}

st.markdown(descripcion_agronomica_detallada.get(
    cluster_pred,
    "No hay descripción disponible para este patrón."
))

# ===============================================================
# 🔍 Análisis fino de intensidad de emergencia
# ===============================================================
st.subheader("🔍 Evaluación fina de intensidad emergente")

peak = emerrel.max() if len(emerrel) > 0 else 0
if len(emerrel) > 0:
    idx_peak = int(np.argmax(emerrel))
    fecha_peak = fechas[idx_peak]
else:
    fecha_peak = None

def safe_to_date(x):
    if x is None:
        return "No definido"
    try:
        return str(pd.to_datetime(x).date())
    except:
        return str(x)

fecha_pico_segura = safe_to_date(fecha_peak)

if emerrel.sum() > 0:
    frac_temprana = emerrel[dias < 90].sum()  / emerrel.sum()
    frac_tardia   = emerrel[dias > 120].sum() / emerrel.sum()
else:
    frac_temprana = 0
    frac_tardia   = 0

st.write({
    "Pico máximo (EMERREL)": float(peak),
    "Fecha del pico": fecha_pico_segura,
    "Proporción temprana (< JD 90)": round(frac_temprana, 3),
    "Proporción tardía (> JD 120)": round(frac_tardia, 3),
})

# Interpretación automática según patrón + proporciones
st.subheader("🧠 Interpretación automática del año")

if cluster_pred == 1:
    # Temprano / Compacto
    if frac_temprana > 0.60:
        st.success("🌱 Año muy temprano: >60% de la emergencia ocurre antes de JD 90.")
    else:
        st.warning("🌱 Año temprano, pero con una cola algo más extendida que el patrón típico.")
elif cluster_pred == 2:
    # Tardío / Extendido
    if frac_tardia > 0.40:
        st.error("🍂 Año altamente tardío: gran parte de la emergencia ocurre después de JD 120.")
    else:
        st.warning("🍂 Año tardío, aunque con menor cola de lo habitual.")
elif cluster_pred == 0:
    # Intermedio / Bimodal
    if frac_temprana > 0.40 and frac_tardia > 0.25:
        st.info("🌾 Año bimodal clásico, con pulsos temprano y tardío bien marcados.")
    else:
        st.info("🌾 Patrón intermedio con menor dominancia de uno de los pulsos.")

# ===============================================================
# 📈 Gráficos comparativos con medoides
# ===============================================================
st.subheader("📈 Curva del año vs medoide asignado")

fig_cmp, ax_cmp = plt.subplots(figsize=(9,5))

ax_cmp.plot(JD_COMMON, curve_interp_year,
            label="Año evaluado (normalizado)",
            color="black", linewidth=3)

med_dict = {0: med0, 1: med1, 2: med2}
ax_cmp.plot(JD_COMMON, med_dict[cluster_pred],
            label=f"Medoide del patrón asignado ({cluster_pred})",
            color=cluster_colors[cluster_pred],
            linewidth=3, linestyle="--")

ax_cmp.set_xlabel("Día Juliano (grilla unificada)")
ax_cmp.set_ylabel("EMERREL normalizada (0–1)")
ax_cmp.legend()
st.pyplot(fig_cmp)

st.subheader("🌈 Los tres patrones funcionales (medoides)")

fig_all, ax_all = plt.subplots(figsize=(9,5))
ax_all.plot(JD_COMMON, med0, label="Patrón 0 — Intermedio/Bimodal", color="blue")
ax_all.plot(JD_COMMON, med1, label="Patrón 1 — Temprano/Compacto",   color="green")
ax_all.plot(JD_COMMON, med2, label="Patrón 2 — Tardío/Extendido",    color="orange")
ax_all.plot(JD_COMMON, curve_interp_year, label="Año evaluado", color="black", linewidth=2)

ax_all.set_xlabel("Día Juliano")
ax_all.set_ylabel("EMERREL normalizada")
ax_all.legend()
st.pyplot(fig_all)

st.subheader("📏 Distancias DTW a los 3 patrones")
st.write({
    "Patrón 0 – Intermedio/Bimodal": float(d0),
    "Patrón 1 – Temprano/Compacto":  float(d1),
    "Patrón 2 – Tardío/Extendido":   float(d2),
})

# ===============================================================
# ✅ FIN
# ===============================================================
st.markdown("---")
st.markdown("""
### ✔ Diagnóstico funcional completado  
Versión **vK3**: ANN + riesgo + clasificador funcional K=3 (DTW K-Medoids)  
+ interpretación agronómica detallada y visualización de patrones.
""")



