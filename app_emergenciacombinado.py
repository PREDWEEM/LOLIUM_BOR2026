# -*- coding: utf-8 -*-
"""
Punto de entrada de PREDWEEM Bordenave.

La aplicación científica original se conserva en
``app_emergenciacombinado_core.py``. Antes de ejecutarla:

* ``parchear_visualizacion_intensidad_relativa`` incorpora al motor el techo y el
  decaimiento de EMERREL desde el 15 de abril (y ajustes de formato de su gráfico);
* se ocultan los elementos que reemplaza la alerta validada del final de la página:
  mapa de riesgo, gráfico de dinámica principal, mensajes de inicio y momento de
  control del motor, control "Umbral Alerta Temprana", la leyenda de versión visual, la imagen de
  fondo (reemplazada por un fondo liso claro, de mejor contraste) y
  las pestañas Monitor de decisión, Precipitaciones y suelo, Análisis estratégico y
  Bio-calibración (incluido el indicador de tiempo térmico), la sección "Datos del lote"
  (se usa el clima SIGA del repositorio y la cobertura de rastrojo por defecto, 75 %) y, en la
  barra lateral, los parámetros del motor (quedan en sus valores por defecto), sus avisos y el
  calibrador 2D; la barra lateral conserva los parámetros de la ventana de control.

Los cálculos del motor no cambian.
"""
from pathlib import Path

from visualizacion_horizonte_pronostico import mostrar_horizonte_pronostico
from visualizacion_intensidad_relativa import parchear_visualizacion_intensidad_relativa

_CORE_APP = Path(__file__).with_name("app_emergenciacombinado_core.py")
_core_source = _CORE_APP.read_text(encoding="utf-8")
_core_source = parchear_visualizacion_intensidad_relativa(_core_source)

# Elementos del motor que reemplaza la alerta validada (al final de la página). El cálculo no cambia.
_GRAFICOS_OCULTOS = (
    "st.plotly_chart(go.Figure(data=go.Heatmap(",   # mapa de riesgo temporal
    "st.plotly_chart(\n                fig_emer,",  # dinámica principal de emergencia
)
for _llamada in _GRAFICOS_OCULTOS:
    if _llamada in _core_source:
        _core_source = _core_source.replace(_llamada, _llamada.replace("st.plotly_chart(", "_omitir_grafico("), 1)

# la regla de estilo "header {visibility: hidden;}" del motor ocultaba también el encabezado de los
# calendarios desplegables (mes y flechas de navegación); se restringe a la barra superior de Streamlit
_core_source = _core_source.replace(
    "    header {visibility: hidden;}",
    '    header[data-testid="stHeader"] {visibility: hidden;}',
    1,
)

# sección "Datos del lote" (carga de clima y de conteos, cobertura de rastrojo): se retira de la página;
# se usa el clima SIGA del repositorio, sin conteos de campo y con la cobertura por defecto (75 %)
_ini_lote = _core_source.find('with st.expander("📂 1. Datos del Lote", expanded=True):')
_fin_lote = _core_source.find('df_meteo_raw = load_data(archivo_meteo, "meteo_daily")', _ini_lote) if _ini_lote >= 0 else -1
if _ini_lote >= 0 and _fin_lote > _ini_lote:
    _core_source = (
        _core_source[:_ini_lote]
        + "archivo_meteo = None\narchivo_campo = None\ncobertura_pct = 75\n"
        + "ke_val, mod_termico = parametros_superficie(cobertura_pct)\n\n"
        + _core_source[_fin_lote:]
    )
    _core_source = _core_source.replace('df_campo_raw = load_data(archivo_campo, "bordenave_campo")', "df_campo_raw = None", 1)

# barra lateral: solo los parámetros que usa la alerta (tiempo térmico y ventana de control). Los del motor
# PREDWEEM, que ahora solo estima el inicio cuando no se informan las primeras plántulas, quedan en sus valores
# por defecto; se quitan también los avisos del motor y el calibrador 2D (requería conteos del formato anterior).
_ini_lateral = _core_source.find('st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")')
_fin_lateral = _core_source.find("# ---------------------------------------------------------\n# 6. MOTOR DE CÁLCULO", _ini_lateral) if _ini_lateral >= 0 else -1
if _ini_lateral >= 0 and _fin_lateral > _ini_lateral:
    _core_source = _core_source[:_ini_lateral] + """st.sidebar.markdown("## ⚙️ Ventana de control")
st.sidebar.caption("Tiempo térmico sobre la temperatura media del aire y objetivos de la ventana post-emergente.")
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)
st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("TT Control Post-emergente (°Cd)", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800, step=10)

# parámetros del motor PREDWEEM en sus valores por defecto (controles retirados de la página)
umbral_er = 0.0001
umbral_termoinhibicion = 24.0
umbral_choque_hidrico = 45.0
tau_decaimiento, beta_decaimiento, intensidad_decaimiento = 60.0, 1.0, 0.75
residualidad = 0
calentamiento_suelo = 0.0
w_max_val = 18.81
exponente_kr = 0.0

""" + _core_source[_fin_lateral:]

_REEMPLAZOS_TEXTO = (
    # botón lateral del reporte del motor: la descarga final reúne los resultados de la alerta validada
    ('st.sidebar.download_button("📥 Descargar Reporte Bordenave"', '_omitir_grafico("📥 Descargar Reporte Bordenave"'),
    # control del umbral del gráfico retirado: se fija su valor por defecto (lo usa la métrica Lead Time)
    ('umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.0001, 0.80, 0.0001)',
     "umbral_er = 0.0001  # control retirado; valor por defecto"),
    # fondo: la imagen de plantas sobre verde medio restaba contraste a textos y gráficos; se usa un fondo liso claro
    ('set_bg_hack("fondo_predweem_v3.png")', "st.markdown(_FONDO_CSS, unsafe_allow_html=True)"),
    # leyenda que describía el gráfico retirado
    ('st.caption("Bordenave · VISUAL V3 · ventana eficiente verde · exportación PNG de alta resolución")', "pass"),
)
for _viejo, _nuevo in _REEMPLAZOS_TEXTO:
    _core_source = _core_source.replace(_viejo, _nuevo, 1)

# mensajes de inicio de conteo térmico y momento crítico de control del motor
_ini = _core_source.find("            if fecha_inicio_ventana:\n                st.success(")
_fin = _core_source.find("        col_gauge = col_gauge", _ini) if _ini >= 0 else -1
if _ini >= 0 and _fin > _ini:
    _core_source = _core_source[:_ini] + "            pass\n\n" + _core_source[_fin:]


# pestañas del motor (Monitor de decisión, Precipitaciones y suelo, Análisis estratégico, Bio-calibración):
# se omite su contenido visual; el reporte Excel y la alerta validada no dependen de él
_ini_tabs = _core_source.find("    tab1, tab2, tab3, tab4 = st.tabs(")
_fin_tabs = _core_source.find("    output = io.BytesIO()", _ini_tabs) if _ini_tabs >= 0 else -1
if _ini_tabs >= 0 and _fin_tabs > _ini_tabs:
    _core_source = _core_source[:_ini_tabs] + _core_source[_fin_tabs:]


_FONDO_CSS = """<style>
.stApp { background: #f4f6f3; }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { background: transparent; }
[data-testid="stSidebar"] { background-color: #e8f3ea; border-right: 1px solid #cfe3d3; }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #14532d !important; }
[data-testid="stCaptionContainer"], .stCaption { color: #4b5563 !important; }
h1, h2, h3, h4 { color: #1f2937; }
</style>"""


def _omitir_grafico(*_args, **_kwargs):
    """No dibuja el gráfico (se conserva la figura y todos los cálculos del motor)."""
    return None

exec(
    compile(_core_source, str(_CORE_APP), "exec"),
    globals(),
)

if "df" in globals() and isinstance(df, pd.DataFrame) and not df.empty:
    st.divider()
    mostrar_horizonte_pronostico(df, "Bordenave")

# Alerta validada de flushes (independiente del motor científico; no modifica sus resultados)
if "df_meteo_raw" in globals() and isinstance(df_meteo_raw, pd.DataFrame) and not df_meteo_raw.empty:
    st.divider()
    try:
        import importlib

        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        mostrar_alerta_lolmu = _vis_alerta.mostrar_alerta_lolmu

        _idx_pico = globals().get("idx_primer_pico")
        _inicio_predicho = (
            df.loc[_idx_pico, "Fecha"]
            if _idx_pico is not None and "df" in globals() and isinstance(df, pd.DataFrame) and _idx_pico in df.index
            else None
        )
        _resultados_alerta = mostrar_alerta_lolmu(
            df_meteo_raw,
            campo=globals().get("df_campo"),
            col_fecha=globals().get("col_fecha"),
            col_plm2=globals().get("col_plm2"),
            inicio_predicho=_inicio_predicho,
            t_base=float(globals().get("t_base_val", 2.0)),
            t_opt=float(globals().get("t_opt_max", 20.0)),
            t_crit=float(globals().get("t_critica", 30.0)),
            tt_control=float(globals().get("dga_optimo", 600.0)),
            tt_limite=float(globals().get("dga_critico", 800.0)),
            tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
            curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta validada: {_error_alerta}")


def _fecha_reporte(valor):
    """Convierte una fecha a texto legible y admite valores ausentes."""
    if valor is None or pd.isna(valor):
        return ""
    return pd.Timestamp(valor).strftime("%d/%m/%Y")


def _escribir_hoja(writer, dataframe, nombre):
    """Escribe una hoja y aplica un formato básico de lectura."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return

    dataframe.to_excel(writer, sheet_name=nombre, index=False)
    hoja = writer.sheets[nombre]
    ultima_fila = len(dataframe)
    ultima_columna = max(0, len(dataframe.columns) - 1)
    hoja.freeze_panes(1, 0)
    hoja.autofilter(0, 0, ultima_fila, ultima_columna)
    hoja.set_column(0, ultima_columna, 18)


# Descarga final: los resultados que muestra la app (alerta validada, emergencia acumulada y ventana de control)
_resultados_alerta = globals().get("_resultados_alerta")
if isinstance(_resultados_alerta, dict) and _resultados_alerta:
    reporte_excel_final = io.BytesIO()
    with pd.ExcelWriter(reporte_excel_final, engine="xlsxwriter", datetime_format="dd/mm/yyyy", date_format="dd/mm/yyyy") as writer:
        for _hoja, _tabla in _resultados_alerta.items():
            _escribir_hoja(writer, _tabla, _hoja)
    reporte_excel_final.seek(0)
    _emision = _resultados_alerta["Resumen"].set_index("Indicador")["Valor"].get("Fecha de emisión", "")

    st.divider()
    st.subheader("📥 Descarga de resultados")
    st.caption(
        "Excel con los resultados que muestra la app para la fecha de emisión elegida: resumen (nivel y probabilidad de "
        "gran flujo, inicio y ventana de control 600–800 °Cd, emergencia acumulada esperada), alerta semanal, emergencia "
        "acumulada con su rango, tiempo térmico diario desde el inicio, conteos usados, clima, parámetros y validación."
    )
    st.download_button(
        label="📊 Descargar resultados en Excel",
        data=reporte_excel_final.getvalue(),
        file_name=f"PREDWEEM_Lolium_Bordenave_alerta_{str(_emision).replace('/', '-') or 'resultados'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        key="descarga_excel_resultados_final_bordenave",
    )
