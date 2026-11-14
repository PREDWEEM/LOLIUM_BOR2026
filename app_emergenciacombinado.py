# -*- coding: utf-8 -*-
"""
PREDWEEM Lolium — Bordenave.

La página muestra la alerta de grandes flujos de Lolium del paquete ``lolium-comun`` (probabilidad de gran flujo en
los próximos 7 días, qué hacer esta semana, ventana del graminicida de barbecho, residual a la siembra y flujos tardíos)
con el clima diario del repositorio (``meteo_daily.csv``).

El motor PREDWEEM anterior (red neuronal y balance hídrico, ``app_emergenciacombinado_core.py`` y sus modelos) ya no se
ejecuta: su inicio estimado de la emergencia erró 12–14 días en promedio y no mejoraba la alerta. Sus archivos quedan
en el repositorio solo como referencia.
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

# Control de acceso (ingreso con Google y usuarios aprobados). Inactivo hasta configurar los secretos [acceso] y [auth].
from acceso_predweem import mostrar_sesion, requerir_acceso
from lolium_comun import pagina

BASE = Path(__file__).resolve().parent

requerir_acceso("Bordenave")
pagina.encabezado("🌾 PREDWEEM LOLIUM — BORDENAVE (BA) lat=-37.761671 lon=-63.083717", "PREDWEEM BORDENAVE")
parametros = pagina.barra_lateral(BASE, "Bordenave")
meteo = pagina.cargar_meteo(BASE, None)


_resultados_alerta = None
if meteo is not None and not meteo.empty:
    try:
        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        _resultados_alerta = _vis_alerta.mostrar_alerta_lolmu(
            meteo,
            **parametros,
    tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
    curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
    curva_tiempo_termico=True,  # con primeras plántulas, curva acumulada en tiempo térmico desde esa fecha
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")

pagina.descarga_excel(_resultados_alerta, "Bordenave", "descarga_excel_resultados_final_bordenave")
mostrar_sesion()
