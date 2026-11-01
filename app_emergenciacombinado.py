# -*- coding: utf-8 -*-
"""
Punto de entrada de PREDWEEM Bordenave.

La aplicación científica original se conserva en
``app_emergenciacombinado_core.py``. Antes de ejecutarla:

* ``parchear_visualizacion_intensidad_relativa`` incorpora al motor el techo y el
  decaimiento de EMERREL desde el 15 de abril (y ajustes de formato de su gráfico);
* se ocultan los elementos que reemplaza la alerta validada del final de la página:
  mapa de riesgo, gráfico de dinámica principal, mensajes de inicio y momento de
  control del motor, control "Umbral Alerta Temprana", la leyenda de versión visual y
  las pestañas Monitor de decisión, Precipitaciones y suelo, Análisis estratégico y
  Bio-calibración (incluido el indicador de tiempo térmico).

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

_REEMPLAZOS_TEXTO = (
    # control del umbral del gráfico retirado: se fija su valor por defecto (lo usa la métrica Lead Time)
    ('umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.0001, 0.80, 0.0001)',
     "umbral_er = 0.0001  # control retirado; valor por defecto"),
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
        from visualizacion_alerta_lolmu import mostrar_alerta_lolmu

        _idx_pico = globals().get("idx_primer_pico")
        _inicio_predicho = (
            df.loc[_idx_pico, "Fecha"]
            if _idx_pico is not None and "df" in globals() and isinstance(df, pd.DataFrame) and _idx_pico in df.index
            else None
        )
        mostrar_alerta_lolmu(
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


# El botón final se muestra únicamente si el motor completó la simulación.
if "df" in globals() and isinstance(df, pd.DataFrame) and not df.empty:
    reporte_excel_final = io.BytesIO()

    resumen_decision = pd.DataFrame(
        {
            "Indicador": [
                "Localidad",
                "Fecha de generación",
                "Inicio del conteo térmico",
                "Fecha objetivo de control",
                "Fecha límite de la ventana",
                "TT acumulado actual (°Cd)",
                "TT pronosticado +7 días (°Cd)",
                "Estado operativo",
                "Cobertura de rastrojo (%)",
                "Wmax superficial (mm)",
                "Ke aplicado",
                "Exponente Kr configurable",
                "Calentamiento diagnóstico del suelo (°C)",
            ],
            "Valor": [
                "Bordenave (Buenos Aires)",
                pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
                _fecha_reporte(globals().get("fecha_inicio_ventana")),
                _fecha_reporte(globals().get("fecha_control")),
                _fecha_reporte(globals().get("fecha_limite")),
                globals().get("dga_hoy", 0.0),
                globals().get("dga_7dias", 0.0),
                globals().get("msg_estado", ""),
                globals().get("cobertura_pct", ""),
                globals().get("w_max_val", ""),
                globals().get("ke_val", ""),
                globals().get("exponente_kr", ""),
                globals().get("calentamiento_suelo", ""),
            ],
        }
    )

    metricas_reporte = pd.DataFrame(
        {
            "Métrica": [
                "PEC (%)",
                "Lag control vs. pico de campo (días)",
                "Lead time (días)",
                "Pearson de flujos",
                "NSE de flujos",
                "KGE de flujos",
                "RMSE acumulado",
                "R2 acumulado",
                "CCC acumulado",
                "Desfase T50 (días)",
                "F1-Score de coincidencia",
                "Exactitud global",
                "Hits",
                "Misses",
                "Falsos positivos",
                "Correctos negativos",
                "Desfase del primer flujo (días)",
            ],
            "Valor": [
                globals().get("pec", 0.0),
                globals().get("peak_lag", 0),
                globals().get("lead_time", 0),
                globals().get("pearson_r", 0.0),
                globals().get("nse_flujos", 0.0),
                globals().get("kge_flujos", 0.0),
                globals().get("rmse_acum", 0.0),
                globals().get("r2_acum", 0.0),
                globals().get("ccc_acum", 0.0),
                globals().get("desfase_t50", 0),
                globals().get("f1_score_coincidencia", 0.0),
                globals().get("exactitud_global", 0.0),
                globals().get("hits_val", 0),
                globals().get("misses_val", 0),
                globals().get("falsos_pos_val", 0),
                globals().get("correctos_neg_val", 0),
                globals().get("lag_inicio_dias", "N/A"),
            ],
        }
    )

    parametros_reporte = pd.DataFrame(
        {
            "Parámetro": [
                "Latitud",
                "Latencia fija (JD)",
                "Ventana termoinhibición (días)",
                "Umbral termoinhibición (°C)",
                "Ventana de lluvia (días)",
                "Choque hídrico (mm)",
                "Fin choque hídrico (JD)",
                "Umbral del primer pico",
                "Cobertura de rastrojo (%)",
                "Wmax superficial (mm)",
                "Ke",
                "Exponente Kr",
                "Modulador térmico diagnóstico",
                "Calentamiento diagnóstico del suelo (°C)",
                "Temperatura base (°C)",
                "Temperatura óptima (°C)",
                "Temperatura crítica (°C)",
                "TT objetivo de control (°Cd)",
                "TT límite de ventana (°Cd)",
                "Residualidad del herbicida (días)",
                "Umbral de alerta temprana",
            ],
            "Valor": [
                -37.761671,
                15,
                5,
                globals().get("umbral_termoinhibicion", ""),
                3,
                globals().get("umbral_choque_hidrico", ""),
                110,
                globals().get("UMBRAL_PRIMER_PICO", ""),
                globals().get("cobertura_pct", ""),
                globals().get("w_max_val", ""),
                globals().get("ke_val", ""),
                globals().get("exponente_kr", ""),
                globals().get("mod_termico", ""),
                globals().get("calentamiento_suelo", ""),
                globals().get("t_base_val", ""),
                globals().get("t_opt_max", ""),
                globals().get("t_critica", ""),
                globals().get("dga_optimo", ""),
                globals().get("dga_critico", ""),
                globals().get("residualidad", ""),
                globals().get("umbral_er", ""),
            ],
        }
    )

    with pd.ExcelWriter(
        reporte_excel_final,
        engine="xlsxwriter",
        datetime_format="dd/mm/yyyy",
        date_format="dd/mm/yyyy",
    ) as writer:
        _escribir_hoja(writer, df, "Resultados_Diarios")
        _escribir_hoja(writer, globals().get("df_sincronizado"), "Validacion_Intervalos")
        _escribir_hoja(writer, globals().get("df_campo"), "Observaciones_Campo")
        _escribir_hoja(writer, resumen_decision, "Resumen_Decision")
        _escribir_hoja(writer, metricas_reporte, "Metricas_Validacion")
        _escribir_hoja(writer, parametros_reporte, "Parametros_Modelo")
        _escribir_hoja(writer, globals().get("df_desde_pico"), "Tiempo_Termico")
        _escribir_hoja(writer, globals().get("tabla_optima"), "Optimizador_2D")

    reporte_excel_final.seek(0)

    st.divider()
    st.subheader("📥 Descarga final de resultados")
    st.caption(
        "El archivo reúne los resultados diarios, la validación Event-to-Event, "
        "las observaciones de campo, las métricas, los parámetros, el tiempo "
        "térmico y, cuando está disponible, el calibrador biofísico 2D."
    )
    st.download_button(
        label="📊 Descargar resultados completos en Excel",
        data=reporte_excel_final.getvalue(),
        file_name="PREDWEEM_Bordenave_Resultados_Completos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        key="descarga_excel_resultados_final_bordenave",
    )
