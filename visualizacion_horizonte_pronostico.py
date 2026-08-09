from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FORECAST_FILL = "rgba(125, 211, 252, 0.20)"
FORECAST_LINE = "#0284C7"
_INLINE_FORECAST_RENDERED = False


def _normalizar_serie(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or "Fecha" not in data.columns or "EMERREL" not in data.columns:
        return pd.DataFrame()
    frame = data.copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.normalize()
    frame["EMERREL"] = pd.to_numeric(frame["EMERREL"], errors="coerce")
    frame = frame.dropna(subset=["Fecha", "EMERREL"])
    return frame.sort_values("Fecha").drop_duplicates("Fecha", keep="last")


def _mascara_pronostico(frame: pd.DataFrame) -> pd.Series:
    today = pd.Timestamp.now().normalize()
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for column in ("TipoDato", "TIPODATO", "TIPO", "Fuente", "FUENTE", "CalidadDato", "CALIDADDATO", "Calidad", "CALIDAD"):
        if column in frame.columns:
            text = frame[column].astype("string").fillna("").str.lower()
            mask |= text.str.contains("pronost", regex=False)
    mask &= frame["Fecha"] >= today
    return mask if bool(mask.any()) else frame["Fecha"] >= today


def obtener_horizonte_pronostico(data: Any) -> pd.DataFrame:
    frame = _normalizar_serie(data)
    if frame.empty:
        return frame
    forecast = frame.loc[_mascara_pronostico(frame), ["Fecha", "EMERREL"]].copy()
    if forecast.empty:
        return forecast
    return forecast.groupby("Fecha", as_index=False, sort=True)["EMERREL"].max().reset_index(drop=True)


def es_figura_emergencia_principal(figure: Any) -> bool:
    try:
        title = str(figure.layout.title.text or "").lower()
        names = " ".join(str(getattr(trace, "name", "") or "").lower() for trace in figure.data)
    except Exception:
        return False
    return "emerg" in title and (
        "dinám" in title or "fisiol" in title or "tasa diaria" in names or "simulad" in names
    )


def _rango_temporal_figura(figure: Any) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    fechas: list[pd.Timestamp] = []
    try:
        for trace in figure.data:
            values = getattr(trace, "x", None)
            if values is None:
                continue
            parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce").dropna()
            fechas.extend(pd.Timestamp(value).normalize() for value in parsed)
    except Exception:
        return None
    return (min(fechas), max(fechas)) if fechas else None


def aplicar_area_pronostico(figure: Any, data: Any = None) -> Any:
    """Sombrea en celeste exclusivamente el horizonte futuro disponible."""
    if not es_figura_emergencia_principal(figure):
        return figure
    start = end = None
    forecast = obtener_horizonte_pronostico(data)
    if not forecast.empty:
        start = pd.Timestamp(forecast["Fecha"].min())
        end = pd.Timestamp(forecast["Fecha"].max())
    else:
        rango = _rango_temporal_figura(figure)
        if rango is not None:
            today = pd.Timestamp.now().normalize()
            _, maximum = rango
            if maximum >= today:
                start, end = today, maximum
    if start is not None and end is not None:
        figure.add_vrect(
            x0=start,
            x1=end + pd.Timedelta(days=1),
            fillcolor=FORECAST_FILL,
            layer="below",
            line_width=0,
        )
    return figure


def _forecast_desde_figura_principal(figure: Any) -> pd.DataFrame:
    """Recupera EMERREL del gráfico principal para ubicar el detalle justo debajo."""
    if not es_figura_emergencia_principal(figure):
        return pd.DataFrame()
    today = pd.Timestamp.now().normalize()
    for trace in figure.data:
        name = str(getattr(trace, "name", "") or "").lower()
        if "tasa diaria simulada" not in name and "emergencia diaria simulada" not in name:
            continue
        x_values = getattr(trace, "x", None)
        if x_values is None:
            continue
        dates = pd.to_datetime(pd.Series(list(x_values)), errors="coerce").dt.normalize()
        emerrel = None
        custom = getattr(trace, "customdata", None)
        if custom is not None and "emergencia diaria simulada" in name:
            try:
                array = np.asarray(custom, dtype=object)
                if array.ndim == 2 and array.shape[1] >= 1:
                    emerrel = pd.to_numeric(pd.Series(array[:, -1]), errors="coerce")
            except Exception:
                emerrel = None
        if emerrel is None:
            y_values = pd.to_numeric(pd.Series(list(getattr(trace, "y", []))), errors="coerce")
            if "log" in name:
                emerrel = (10.0 ** y_values) - 0.01
            else:
                emerrel = y_values
                if float(emerrel.max(skipna=True) or 0.0) > 1.01:
                    emerrel = emerrel / 100.0
        forecast = pd.DataFrame({"Fecha": dates, "EMERREL": emerrel})
        forecast = forecast.dropna(subset=["Fecha", "EMERREL"])
        forecast["EMERREL"] = forecast["EMERREL"].clip(lower=0.0, upper=1.0)
        forecast = forecast.loc[forecast["Fecha"] >= today]
        if not forecast.empty:
            return forecast.groupby("Fecha", as_index=False, sort=True)["EMERREL"].max()
    return pd.DataFrame()


def _figura_detalle(forecast: pd.DataFrame, site_name: str = "") -> go.Figure:
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())
    labels = [f"{value:.4f}" for value in forecast["EMERREL"]]
    suffix = f" · {site_name}" if site_name else ""
    figure = go.Figure(go.Scatter(
        x=forecast["Fecha"],
        y=forecast["EMERREL"],
        mode="lines+markers+text",
        text=labels,
        textposition="top center",
        cliponaxis=False,
        line={"color": FORECAST_LINE, "width": 3.0},
        marker={"size": 9, "color": FORECAST_LINE},
        hovertemplate="<b>%{x|%d-%m-%Y}</b><br>EMERREL: %{y:.4f}<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_white",
        title={"text": f"Detalle del pronóstico de emergencia{suffix}", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Fecha", "tickmode": "array", "tickvals": forecast["Fecha"], "ticktext": forecast["Fecha"].dt.strftime("%d-%m"), "range": [start - pd.Timedelta(hours=12), end + pd.Timedelta(hours=12)], "showgrid": False, "fixedrange": False},
        yaxis={"title": "EMERREL", "range": [0.0, 1.0], "tickmode": "array", "tickvals": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "showgrid": True, "gridcolor": "rgba(148,163,184,0.25)", "fixedrange": False},
        hovermode="x unified",
        height=390,
        margin={"l": 72, "r": 24, "t": 70, "b": 62},
        showlegend=False,
    )
    return figure


def construir_figura_horizonte(data: Any, site_name: str) -> tuple[pd.DataFrame, go.Figure] | None:
    forecast = obtener_horizonte_pronostico(data)
    if forecast.empty:
        return None
    return forecast, _figura_detalle(forecast, site_name)


def mostrar_horizonte_pronostico(data: Any, site_name: str) -> None:
    """Respaldo para apps donde el detalle no pudo insertarse junto al principal."""
    if _INLINE_FORECAST_RENDERED:
        return
    result = construir_figura_horizonte(data, site_name)
    if result is None:
        return
    forecast, figure = result
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())
    st.markdown("##### 🔭 Pronóstico de emergencia — detalle diario")
    st.caption(
        f"{site_name}: horizonte disponible del {start.strftime('%d-%m-%Y')} al {end.strftime('%d-%m-%Y')}. "
        "Eje Y en escala lineal EMERREL 0–1; no se agregan ni extrapolan días."
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


if not getattr(st.plotly_chart, "_predweem_forecast_area", False):
    _ORIGINAL_ST_PLOTLY_CHART = st.plotly_chart

    def _plotly_chart_con_area_pronostico(figure: Any, *args: Any, **kwargs: Any):
        global _INLINE_FORECAST_RENDERED
        is_main = es_figura_emergencia_principal(figure)
        aplicar_area_pronostico(figure)
        result = _ORIGINAL_ST_PLOTLY_CHART(figure, *args, **kwargs)
        if is_main:
            forecast = _forecast_desde_figura_principal(figure)
            if not forecast.empty:
                start = pd.Timestamp(forecast["Fecha"].min())
                end = pd.Timestamp(forecast["Fecha"].max())
                st.markdown("##### 🔭 Pronóstico de emergencia — detalle diario")
                st.caption(
                    f"Horizonte disponible del {start.strftime('%d-%m-%Y')} al {end.strftime('%d-%m-%Y')}. "
                    "Eje Y en escala lineal EMERREL 0–1."
                )
                _ORIGINAL_ST_PLOTLY_CHART(
                    _figura_detalle(forecast),
                    width="stretch",
                    config={"displaylogo": False, "responsive": True, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                )
                _INLINE_FORECAST_RENDERED = True
        return result

    _plotly_chart_con_area_pronostico._predweem_forecast_area = True
    st.plotly_chart = _plotly_chart_con_area_pronostico
