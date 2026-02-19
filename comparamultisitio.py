# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM MULTI-SITE COMPARATOR — LOLIUM 2026
# Comparativa de salidas del modelo entre sitios (meteo_daily.csv)
# alojados en distintos repositorios de GitHub (links tipo blob o raw)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import requests
from pathlib import Path
from urllib.parse import urlparse
from io import StringIO

# ---------------------------------------------------------
# 1) CONFIG & ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM — Comparativa Multi-Sitio",
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
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .bio-alert {
        padding: 10px;
        border-radius: 6px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .ok-alert {
        padding: 10px;
        border-radius: 6px;
        background-color: #dcfce7;
        color: #14532d;
        border: 1px solid #86efac;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2) MOCK FILES (para que siempre corra en demo/offline)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100) ** 2) / 600)
        p2 = np.exp(-((jd - 160) ** 2) / 900) + 0.3 * np.exp(-((jd - 260) ** 2) / 1200)
        p3 = np.exp(-((jd - 230) ** 2) / 1500)
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2],
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

    # mock meteo local (solo si quisieras probar sin GitHub)
    if not (BASE / "meteo_daily.csv").exists():
        dates = pd.date_range(start="2026-01-01", periods=180)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(25, 35, size=len(dates)) - (np.arange(len(dates)) * 0.06),
            "TMIN": np.random.uniform(10, 18, size=len(dates)) - (np.arange(len(dates)) * 0.03),
            "Prec": np.random.choice([0, 0, 2, 6, 12, 25], size=len(dates)),
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3) DTW + TT + ANN
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb])

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base:
        return 0.0
    elif t <= t_opt:
        return float(t - t_base)
    elif t < t_crit:
        factor = (t_crit - t) / (t_crit - t_opt)
        return float((t - t_base) * factor)
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
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer).flatten() + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    ann = PracticalANNModel(
        np.load(BASE / "IW.npy"),
        np.load(BASE / "bias_IW.npy"),
        np.load(BASE / "LW.npy"),
        np.load(BASE / "bias_out.npy"),
    )
    with open(BASE / "modelo_clusters_k3.pkl", "rb") as f:
        k3 = pickle.load(f)
    return ann, k3

# ---------------------------------------------------------
# 4) GitHub: blob -> raw + carga robusta
# ---------------------------------------------------------
def github_blob_to_raw(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("https://raw.githubusercontent.com/"):
        return u

    p = urlparse(u)
    if p.netloc not in ("github.com", "www.github.com"):
        raise ValueError(f"URL no parece GitHub: {u}")

    parts = p.path.strip("/").split("/")
    # /OWNER/REPO/blob/BRANCH/path/file.csv
    if len(parts) < 5 or parts[2] != "blob":
        raise ValueError(f"URL GitHub no tiene formato /blob/: {u}")

    owner, repo, _, branch = parts[:4]
    path = "/".join(parts[4:])
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_meteo_from_github(blob_or_raw_url: str, token: str | None = None) -> pd.DataFrame:
    raw_url = github_blob_to_raw(blob_or_raw_url)

    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    r = requests.get(raw_url, headers=headers, timeout=30)
    r.raise_for_status()

    # autodetect separador (si viniera con ;)
    try:
        df = pd.read_csv(StringIO(r.text), parse_dates=["Fecha"])
    except Exception:
        df = pd.read_csv(StringIO(r.text), sep=None, engine="python", parse_dates=["Fecha"])

    # normalización columnas
    df.columns = [c.upper().strip() for c in df.columns]
    mapeo = {
        "FECHA": "Fecha", "DATE": "Fecha",
        "TMAX": "TMAX", "TMIN": "TMIN",
        "PREC": "Prec", "LLUVIA": "Prec", "RAIN": "Prec", "PPT": "Prec"
    }
    df = df.rename(columns=mapeo)

    # validación mínima
    required = {"Fecha", "TMAX", "TMIN", "Prec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Columnas encontradas: {list(df.columns)}")

    return df

# ---------------------------------------------------------
# 5) Pipeline por sitio + métricas comparables
# ---------------------------------------------------------
def run_site_pipeline(df_in: pd.DataFrame,
                      modelo_ann: PracticalANNModel,
                      t_base_val: float,
                      t_opt_max: float,
                      t_critica: float,
                      umbral_er: float,
                      dga_optimo: float,
                      dga_critico: float,
                      cluster_model: dict) -> dict:
    df = df_in.copy()
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)

    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # primer pico
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_inicio = df.loc[indices_pulso[0], "Fecha"] if indices_pulso else None

    # TT acumulado desde pico
    dga_total = 0.0
    dias_stress = 0
    fecha_cruce_opt = None
    fecha_cruce_crit = None
    if fecha_inicio is not None:
        d = df[df["Fecha"] >= fecha_inicio].copy()
        d["DGA_cum"] = d["DG"].cumsum()
        dga_total = float(d["DGA_cum"].iloc[-1]) if not d.empty else 0.0
        dias_stress = int((d["Tmedia"] > t_opt_max).sum())

        # cruces
        if (d["DGA_cum"] >= dga_optimo).any():
            fecha_cruce_opt = d.loc[d["DGA_cum"] >= dga_optimo, "Fecha"].iloc[0]
        if (d["DGA_cum"] >= dga_critico).any():
            fecha_cruce_crit = d.loc[d["DGA_cum"] >= dga_critico, "Fecha"].iloc[0]

    # DTW clasificación (hasta 1 mayo)
    pred_cluster = None
    dtw_score = None
    try:
        fecha_corte = pd.Timestamp("2026-05-01")
        df_obs = df[df["Fecha"] < fecha_corte].copy()
        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = int(df_obs["Julian_days"].max())
            max_e = float(df_obs["EMERREL"].max()) if float(df_obs["EMERREL"].max()) > 0 else 1.0
            JD_COM = np.array(cluster_model["JD_common"])
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)

            dists = []
            for m in cluster_model["curves_interp"]:
                m = np.array(m)
                m_slice = m[JD_COM <= jd_corte]
                m_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
                dists.append(dtw_distance(obs_norm, m_norm))

            pred_cluster = int(np.argmin(dists))
            dtw_score = float(min(dists))
    except Exception:
        pred_cluster, dtw_score = None, None

    resumen = {
        "fecha_inicio_pico": fecha_inicio,
        "max_emerrel": float(df["EMERREL"].max()) if len(df) else 0.0,
        "sum_emerrel": float(df["EMERREL"].sum()) if len(df) else 0.0,
        "dga_total_desde_pico": float(dga_total),
        "dias_stress_desde_pico": int(dias_stress),
        "cruce_600": fecha_cruce_opt,
        "cruce_800": fecha_cruce_crit,
        "cluster_dtw": pred_cluster,
        "dtw_score": dtw_score,
        "n_dias": int(len(df)),
        "fecha_inicio": df["Fecha"].min() if len(df) else None,
        "fecha_fin": df["Fecha"].max() if len(df) else None,
    }

    return {"df": df, "resumen": resumen}

# ---------------------------------------------------------
# 6) SITES (tus links)
# ---------------------------------------------------------
DEFAULT_SITES = {
    "Bordenave": "https://github.com/PREDWEEM/LOLIUM_BOR2026/blob/main/meteo_daily.csv",
    "Tres Arroyos": "https://github.com/PREDWEEM/loliumTA_2026/blob/main/meteo_daily.csv",
    "Balcarce": "https://github.com/PREDWEEM/LOLIUM_BAL2026/blob/main/meteo_daily.csv",
    "Lartigau": "https://github.com/PREDWEEM/LOLIUM_LARTIGAU-2026/blob/main/meteo_daily.csv",
}

# ---------------------------------------------------------
# 7) SIDEBAR (parámetros + selección sitios)
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ Configuración")
st.sidebar.caption("Compará múltiples sitios desde repositorios GitHub.")

# Parámetros pico
st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Emergencia**")
umbral_er = st.sidebar.slider("Umbral Tasa Diaria (pico)", 0.05, 0.80, 0.50)

# Bio-limit
st.sidebar.divider()
st.sidebar.markdown("🌡️ **Fisiología Térmica (Bio-Limit)**")
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2:
    t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = float(st.sidebar.number_input("Objetivo Control", value=600, step=50))
dga_critico = float(st.sidebar.number_input("Límite Ventana", value=800, step=50))

# GitHub token opcional (para privados / rate-limit)
github_token = st.secrets.get("GITHUB_TOKEN", None)
if github_token:
    st.sidebar.markdown('<div class="ok-alert">🔐 GITHUB_TOKEN detectado en secrets.</div>', unsafe_allow_html=True)
else:
    st.sidebar.caption("Tip: si algún repo es privado / rate limit, agregá GITHUB_TOKEN en .streamlit/secrets.toml")

# Selección de sitios
st.sidebar.divider()
st.sidebar.markdown("## 🌍 Sitios (GitHub)")

selected_sites = st.sidebar.multiselect(
    "Elegí sitios",
    list(DEFAULT_SITES.keys()),
    default=list(DEFAULT_SITES.keys())
)

# Permitir editar URLs
with st.sidebar.expander("✏️ Editar URLs (opcional)"):
    site_urls = {}
    for k, v in DEFAULT_SITES.items():
        site_urls[k] = st.text_input(f"{k}", value=v, key=f"url_{k}")
else:
    site_urls = DEFAULT_SITES.copy()

# Opción de sumar un sitio extra pegando URL
st.sidebar.divider()
st.sidebar.markdown("## ➕ Sitio extra (opcional)")
extra_name = st.sidebar.text_input("Nombre", value="")
extra_url = st.sidebar.text_input("URL meteo_daily.csv (GitHub blob/raw)", value="")

run_btn = st.sidebar.button("🚀 Ejecutar comparativa", use_container_width=True)

# ---------------------------------------------------------
# 8) EJECUCIÓN MULTI-SITE
# ---------------------------------------------------------
if not run_btn:
    st.title("🌾 PREDWEEM — Comparativa Multi-Sitio")
    st.info("Configura parámetros en la barra lateral y presioná **Ejecutar comparativa**.")
    st.stop()

targets = {}
for s in selected_sites:
    targets[s] = site_urls[s]
if extra_name.strip() and extra_url.strip():
    targets[extra_name.strip()] = extra_url.strip()

if not targets:
    st.warning("No seleccionaste ningún sitio.")
    st.stop()

site_results = {}
site_errors = {}

with st.spinner("Cargando meteo_daily.csv y ejecutando modelo por sitio..."):
    for site, url in targets.items():
        try:
            df_site = load_meteo_from_github(url, token=github_token)
            site_results[site] = run_site_pipeline(
                df_site, modelo_ann,
                t_base_val, t_opt_max, t_critica,
                umbral_er, dga_optimo, dga_critico,
                cluster_model
            )
        except Exception as e:
            site_errors[site] = str(e)

# ---------------------------------------------------------
# 9) OUTPUTS: Comparativa + Detalle por sitio
# ---------------------------------------------------------
st.title("🌾 PREDWEEM — Comparativa Multi-Sitio")

if site_errors:
    st.warning("Algunos sitios no pudieron cargarse:")
    for k, v in site_errors.items():
        st.write(f"- **{k}**: {v}")

if not site_results:
    st.error("No se pudo procesar ningún sitio. Revisá URLs y formato del CSV.")
    st.stop()

# ---------- Tabla resumen ----------
rows = []
cluster_names = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
for site, pack in site_results.items():
    r = pack["resumen"].copy()
    r["sitio"] = site
    # formateos
    def fmt_date(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        except Exception:
            return str(x)

    r["fecha_inicio_pico"] = fmt_date(r["fecha_inicio_pico"])
    r["cruce_600"] = fmt_date(r["cruce_600"])
    r["cruce_800"] = fmt_date(r["cruce_800"])
    r["fecha_inicio"] = fmt_date(r["fecha_inicio"])
    r["fecha_fin"] = fmt_date(r["fecha_fin"])
    r["cluster_dtw"] = cluster_names.get(r["cluster_dtw"], None)
    rows.append(r)

df_sum = pd.DataFrame(rows).set_index("sitio")
st.subheader("📌 Resumen comparativo")
st.dataframe(df_sum, use_container_width=True)

# ---------- EMERREL comparado ----------
st.subheader("📈 EMERREL comparado (tasa diaria)")
fig_em = go.Figure()
for site, pack in site_results.items():
    dfx = pack["df"]
    fig_em.add_trace(go.Scatter(x=dfx["Fecha"], y=dfx["EMERREL"], mode="lines", name=site))
fig_em.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral pico ({umbral_er})")
fig_em.update_layout(height=420, legend_title_text="Sitios")
st.plotly_chart(fig_em, use_container_width=True)

# ---------- TT acumulado desde pico comparado ----------
st.subheader("🌡️ TT acumulado desde el primer pico (homologado)")
fig_tt = go.Figure()
for site, pack in site_results.items():
    dfx = pack["df"]
    f0 = pack["resumen"]["fecha_inicio_pico"]
    if f0 is None:
        continue
    d = dfx[dfx["Fecha"] >= f0].copy()
    d["DGA_cum"] = d["DG"].cumsum()
    fig_tt.add_trace(go.Scatter(x=d["Fecha"], y=d["DGA_cum"], mode="lines", name=site))
fig_tt.add_hline(y=dga_optimo, line_dash="dot", annotation_text="Objetivo control")
fig_tt.add_hline(y=dga_critico, line_dash="dash", annotation_text="Límite ventana")
fig_tt.update_layout(height=420, legend_title_text="Sitios")
st.plotly_chart(fig_tt, use_container_width=True)

# ---------- Heatmap multi-sitio ----------
st.subheader("🗺️ Heatmap multi-sitio (EMERREL)")
all_dates = sorted(set(pd.concat([pack["df"]["Fecha"] for pack in site_results.values()]).unique()))
site_names = list(site_results.keys())

Z = []
for site in site_names:
    dfx = site_results[site]["df"].set_index("Fecha")
    z = dfx.reindex(all_dates)["EMERREL"].fillna(0.0).to_numpy()
    Z.append(z)

colorscale_hard = [
    [0.0, "green"], [0.49, "green"],
    [0.50, "yellow"], [0.79, "yellow"],
    [0.80, "red"], [1.0, "red"]
]
fig_hm = go.Figure(data=go.Heatmap(
    z=Z, x=all_dates, y=site_names,
    colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
))
fig_hm.update_layout(height=240 + 35 * len(site_names), margin=dict(t=30, b=0, l=10, r=10))
st.plotly_chart(fig_hm, use_container_width=True)

# ---------------------------------------------------------
# 10) DETALLE POR SITIO (tabs)
# ---------------------------------------------------------
st.subheader("🔎 Detalle por sitio")
tabs = st.tabs([f"📍 {s}" for s in site_results.keys()])

for i, site in enumerate(site_results.keys()):
    with tabs[i]:
        dfx = site_results[site]["df"]
        res = site_results[site]["resumen"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Max EMERREL", f"{res['max_emerrel']:.3f}")
        with c2:
            st.metric("Σ EMERREL", f"{res['sum_emerrel']:.3f}")
        with c3:
            st.metric("TT desde pico", f"{res['dga_total_desde_pico']:.1f} °Cd")
        with c4:
            st.metric("Estrés T > Topt", f"{res['dias_stress_desde_pico']} días")

        if res["fecha_inicio_pico"] is None:
            st.warning("⏳ No se detectó pico (EMERREL >= umbral) para este sitio con la configuración actual.")
        else:
            st.success(f"📅 Inicio de conteo (primer pico): {pd.to_datetime(res['fecha_inicio_pico']).strftime('%d-%m-%Y')}")

        # Heatmap sitio
        fig_risk = go.Figure(data=go.Heatmap(
            z=[dfx["EMERREL"].values],
            x=dfx["Fecha"], y=["Emergencia"],
            colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
        ))
        fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Intensidad de Emergencia")
        st.plotly_chart(fig_risk, use_container_width=True)

        # EMERREL + umbral
        fig_site = go.Figure()
        fig_site.add_trace(go.Scatter(
            x=dfx["Fecha"], y=dfx["EMERREL"],
            mode="lines", name="EMERREL", fill="tozeroy"
        ))
        fig_site.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral ({umbral_er})")
        fig_site.update_layout(height=320, title=f"Dinámica de EMERREL — {site}")
        st.plotly_chart(fig_site, use_container_width=True)

        # Gauge TT hoy +7d (si hay datos próximos)
        colA, colB = st.columns([2, 1])
        with colB:
            fecha_hoy = pd.Timestamp.now().normalize()
            if fecha_hoy not in dfx["Fecha"].values:
                fecha_hoy = dfx["Fecha"].max()
            idx_hoy = int(dfx[dfx["Fecha"] == fecha_hoy].index[0])
            df_periodo_total = dfx.iloc[: min(idx_hoy + 8, len(dfx))].copy()

            indices_pico = df_periodo_total.index[df_periodo_total["EMERREL"] >= umbral_er].tolist()
            dga_hoy, dga_7dias = 0.0, 0.0
            msg_estado = "Esperando pico..."

            if indices_pico:
                idx_primer_pico = indices_pico[0]
                fecha_inicio_pico = dfx.loc[idx_primer_pico, "Fecha"]

                if fecha_inicio_pico <= fecha_hoy:
                    df_hasta_hoy = dfx[(dfx["Fecha"] >= fecha_inicio_pico) & (dfx["Fecha"] <= fecha_hoy)]
                    dga_hoy = float(df_hasta_hoy["DG"].sum())
                    df_pronostico = dfx.iloc[idx_hoy + 1 : min(idx_hoy + 8, len(dfx))]
                    dga_7dias = float(dga_hoy + df_pronostico["DG"].sum())
                    msg_estado = f"Pico detectado el {fecha_inicio_pico.strftime('%d/%m')}"
                else:
                    df_futuro_post_pico = dfx[(dfx["Fecha"] >= fecha_inicio_pico) & (dfx.index <= idx_hoy + 7)]
                    dga_7dias = float(df_futuro_post_pico["DG"].sum())
                    msg_estado = f"⚠️ Pico previsto para el {fecha_inicio_pico.strftime('%d/%m')}"

            max_axis = dga_critico * 1.2
            fig_g = go.Figure()
            fig_g.add_trace(go.Indicator(
                mode="gauge+number",
                value=dga_hoy,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>TT ACUMULADO (°Cd)</b>", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [None, max_axis]},
                    'bar': {'color': "#1e293b", 'thickness': 0.3},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#4ade80"},
                        {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                        {'range': [dga_critico, max_axis], 'color': "#f87171"},
                    ],
                    'threshold': {
                        'line': {'color': "#2563eb", 'width': 6},
                        'thickness': 0.8,
                        'value': dga_7dias
                    }
                }
            ))
            fig_g.add_annotation(
                x=0.5, y=-0.1,
                text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>",
                showarrow=False, font=dict(size=13, color="#1e3a8a"), align="center"
            )
            fig_g.update_layout(height=320, margin=dict(t=70, b=45, l=10, r=10))
            st.plotly_chart(fig_g, use_container_width=True)

        with colA:
            # DTW plot
            st.markdown("### 🔍 Clasificación DTW (hasta 1 mayo)")
            if res["cluster_dtw"] is None:
                st.info("Datos insuficientes para DTW (o sin señal antes de mayo).")
            else:
                # reconstituir para plot
                fecha_corte = pd.Timestamp("2026-05-01")
                df_obs = dfx[dfx["Fecha"] < fecha_corte].copy()
                jd_corte = int(df_obs["Julian_days"].max())
                max_e = float(df_obs["EMERREL"].max()) if float(df_obs["EMERREL"].max()) > 0 else 1.0
                JD_COM = np.array(cluster_model["JD_common"])
                jd_grid = JD_COM[JD_COM <= jd_corte]
                obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)

                pred = { "🌾 Bimodal":0, "🌱 Temprano":1, "🍂 Tardío":2 }.get(res["cluster_dtw"], None)
                # res["cluster_dtw"] ya está mapeado como string en tabla, pero aquí guardamos int en resumen original:
                pred_int = site_results[site]["resumen"]["cluster_dtw"]
                names = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
                cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}

                fp = go.Figure()
                fp.add_trace(go.Scatter(
                    x=JD_COM, y=np.array(cluster_model["curves_interp"][pred_int]),
                    name="Patrón histórico", line=dict(dash="dash", color=cols.get(pred_int))
                ))
                fp.add_trace(go.Scatter(
                    x=jd_grid,
                    y=obs_norm * np.array(cluster_model["curves_interp"][pred_int]).max(),
                    name="Observado 2026", line=dict(color="black", width=3)
                ))
                fp.update_layout(height=320, title=f"{names.get(pred_int)} — DTW={site_results[site]['resumen']['dtw_score']:.2f}")
                st.plotly_chart(fp, use_container_width=True)

        # Curva bio
        st.markdown("### 🧪 Curva de Respuesta Fisiológica (Bio-limit)")
        x_temps = np.linspace(0, 45, 200)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=x_temps, y=y_tt, mode="lines", fill="tozeroy", name="TT"))
        fig_b.add_vrect(x0=t_base_val, x1=t_opt_max, fillcolor="green", opacity=0.1)
        fig_b.add_vrect(x0=t_opt_max, x1=t_critica, fillcolor="orange", opacity=0.1)
        fig_b.add_vrect(x0=t_critica, x1=45, fillcolor="red", opacity=0.1)
        fig_b.update_layout(height=320, xaxis_title="T media (°C)", yaxis_title="TT (°Cd)")
        st.plotly_chart(fig_b, use_container_width=True)

# ---------------------------------------------------------
# 11) EXPORT EXCEL (Resumen + 1 hoja por sitio)
# ---------------------------------------------------------
st.subheader("📥 Exportación")

output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df_sum.reset_index().to_excel(writer, index=False, sheet_name="Resumen")
    # parámetros globales
    pd.DataFrame({
        "Parametro": ["T_Base", "T_Optima", "T_Critica", "Umbral_Pico", "Objetivo_600", "Limite_800"],
        "Valor": [t_base_val, t_opt_max, t_critica, umbral_er, dga_optimo, dga_critico]
    }).to_excel(writer, index=False, sheet_name="Parametros")

    for site, pack in site_results.items():
        sheet = site[:28]
        pack["df"].to_excel(writer, index=False, sheet_name=sheet)

st.download_button(
    "📥 Descargar Comparativa (Excel)",
    data=output.getvalue(),
    file_name="PREDWEEM_Comparativa_MultiSitio.xlsx",
    use_container_width=True
)

