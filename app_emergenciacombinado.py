# -*- coding: utf-8 -*-
"""
Punto de entrada de PREDWEEM Bordenave.

La aplicación científica original se conserva en
``app_emergenciacombinado_core.py``. Este archivo la ejecuta sin alterar su
lógica y agrega, al final de toda la interfaz, una descarga Excel completa de
los resultados generados.
"""
from pathlib import Path

_CORE_APP = Path(__file__).with_name("app_emergenciacombinado_core.py")
exec(
    compile(_CORE_APP.read_text(encoding="utf-8"), str(_CORE_APP), "exec"),
    globals(),
)


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
        _escribir_hoja(
            writer,
            globals().get("df_sincronizado"),
            "Validacion_Intervalos",
        )
        _escribir_hoja(
            writer,
            globals().get("df_campo"),
            "Observaciones_Campo",
        )
        _escribir_hoja(writer, resumen_decision, "Resumen_Decision")
        _escribir_hoja(writer, metricas_reporte, "Metricas_Validacion")
        _escribir_hoja(writer, parametros_reporte, "Parametros_Modelo")
        _escribir_hoja(
            writer,
            globals().get("df_desde_pico"),
            "Tiempo_Termico",
        )
        _escribir_hoja(
            writer,
            globals().get("tabla_optima"),
            "Optimizador_2D",
        )

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
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
        key="descarga_excel_resultados_final_bordenave",
    )
