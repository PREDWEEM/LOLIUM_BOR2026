from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _normalizar_serie(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame()
    if "Fecha" not in data.columns or "EMERREL" not in data.columns:
        return pd.DataFrame()

    frame = data.copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.normalize()
    frame["EMERREL"] = pd.to_numeric(frame["EMERREL"], errors="coerce")
    frame = frame.dropna(subset=["Fecha", "EMERREL"])
    if frame.empty:
        return frame
    return frame.sort_values("Fecha").drop_duplicates("Fecha", keep="last")


def _mascara_pronostico(frame: pd.DataFrame, today: pd.Timestamp) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for column in (
        "TipoDato", "TIPODATO", "TIPO", "Fuente", "FUENTE",
        "CalidadDato", "CALIDADDATO", "Calidad", "CALIDAD",
    ):
        if column in frame.columns:
            text = frame[column].astype("string").fillna("").str.lower()
            mask = mask | text.str.contains("pronost", regex=False)
    mask = mask & (frame["Fecha"] >= today)
    if bool(mask.any()):
        return mask
    return frame["Fecha"] >= today


def mostrar_horizonte_pronostico(data: Any, site_name: str) -> None:
    frame = _normalizar_serie(data)
    if frame.empty:
        return
    today = pd.Timestamp.now().normalize()
    forecast = frame.loc[_mascara_pronostico(frame, today), ["Fecha", "EMERREL"]].copy()
    if forecast.empty:
        return
    forecast = forecast.groupby("Fecha", as_index=False, sort=True)["EMERREL"].max().reset_index(drop=True)
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())
    st.markdown("##### 🔭 Pronóstico de emergencia — horizonte completo")
    st.caption(
        f"{site_name}: {len(forecast)} días meteorológicos disponibles, "
        f"del {start.strftime('%d-%m-%Y')} al {end.strftime('%d-%m-%Y')}. "
        "Se muestran únicamente las fechas presentes en la serie operativa; no se agregan ni extrapolan días."
    )
    labels = [f"{v:.5f}" if abs(float(v)) < 0.001 else f"{v:.3f}" for v in forecast["EMERREL"]]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=forecast["Fecha"], y=forecast["EMERREL"], mode="lines+markers+text",
        name="EMERREL pronosticada", text=labels, textposition="top center", cliponaxis=False,
        line={"color": "#2563EB", "width": 3.0}, marker={"size": 9, "color": "#2563EB"},
        hovertemplate="<b>%{x|%d-%m-%Y}</b><br>EMERREL: %{y:.5f}<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_white",
        title={"text": f"Detalle diario del pronóstico de emergencia · {site_name}", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Fecha", "tickmode": "array", "tickvals": forecast["Fecha"], "ticktext": forecast["Fecha"].dt.strftime("%d-%m"), "range": [start - pd.Timedelta(hours=12), end + pd.Timedelta(hours=12)], "showgrid": False, "fixedrange": False},
        yaxis={"title": "EMERREL", "rangemode": "tozero", "showgrid": True, "gridcolor": "rgba(148,163,184,0.25)", "fixedrange": False},
        hovermode="x unified", height=390, margin={"l": 72, "r": 24, "t": 70, "b": 62}, showlegend=False,
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
