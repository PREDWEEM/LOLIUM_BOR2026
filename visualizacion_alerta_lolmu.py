# -*- coding: utf-8 -*-
"""Sección de la app: alerta validada de flushes, reloj térmico desde las primeras plántulas y calendario."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import alerta_lolmu as al

COLOR = {"alerta": "#c2410c", "atencion": "#d97706", "baja": "#15803d"}
NOMBRE = {"alerta": "Alerta", "atencion": "Atención", "baja": "Sin alerta"}
TEXTO = {"alerta": "Flush probable en los próximos 7 días: recorrer el lote y preparar el control.",
         "atencion": "Probabilidad de flush por encima de lo habitual: conviene mirar el lote esta semana.",
         "baja": "Flush poco probable en los próximos 7 días."}
INFO = {1: "solo clima (sin datos del lote)", 2: "clima + fecha de primeras plántulas", 3: "clima + primeras plántulas + conteos"}


def _ultima_observada(meteo: pd.DataFrame) -> pd.Timestamp:
    m = meteo.copy()
    fechas = pd.to_datetime(m["Fecha"], errors="coerce")
    if "TipoDato" in m.columns:
        obs = fechas[~m["TipoDato"].astype(str).str.contains("Pronostico", case=False, na=False)]
        if not obs.empty:
            return obs.max().normalize()
    return fechas.max().normalize()


def _fecha_tt(dg: pd.Series, desde: pd.Timestamp, objetivo: float):
    acum = dg[dg.index >= desde].cumsum()
    hit = acum[acum >= objetivo]
    return hit.index[0] if not hit.empty else None


def mostrar_alerta_lolmu(meteo_raw: pd.DataFrame, campo: pd.DataFrame | None = None, col_fecha: str | None = None,
                         col_plm2: str | None = None, inicio_predicho: pd.Timestamp | None = None,
                         t_base: float = 2.0, t_opt: float = 20.0, t_crit: float = 30.0,
                         tt_control: float = 600.0, tt_limite: float = 800.0) -> None:
    st.subheader("🔔 Alerta de flushes validada (17 campañas)")
    st.caption("Probabilidad de que en los próximos 7 días emerja ≥ 5 % de la campaña. Combina el calendario medio de "
               "LOLMU, la respuesta a las lluvias ya registradas y, si se informan, la fecha de las primeras plántulas "
               "y los conteos del lote. Independiente del motor PREDWEEM de arriba.")
    if meteo_raw is None or meteo_raw.empty:
        st.info("Sin datos meteorológicos.")
        return
    try:
        modelo = al.AlertaLolmu()
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo cargar alerta_lolmu_params.json: {e}")
        return

    meteo = meteo_raw.copy()
    fechas = pd.to_datetime(meteo["Fecha"], errors="coerce")
    ultima = _ultima_observada(meteo)
    anio = ultima.year
    conteos = None
    if campo is not None and col_fecha and col_plm2 and not campo.empty:
        conteos = (campo[[col_fecha, col_plm2]].rename(columns={col_fecha: "Fecha", col_plm2: "PLM2"})
                   .assign(Fecha=lambda d: pd.to_datetime(d["Fecha"], errors="coerce").dt.normalize(),
                           PLM2=lambda d: pd.to_numeric(d["PLM2"], errors="coerce"))
                   .dropna().sort_values("Fecha"))
        conteos = conteos[conteos["Fecha"].dt.year == anio]
        conteos = conteos if not conteos.empty else None
    vistas_conteo = conteos[conteos["PLM2"] > 0]["Fecha"].min() if conteos is not None and (conteos["PLM2"] > 0).any() else None

    # asegura que el calendario desplegable muestre el mes y las flechas de navegación
    st.markdown("<style>[data-baseweb='calendar'] header, [data-baseweb='popover'] header {visibility: visible !important;}</style>",
                unsafe_allow_html=True)
    with st.form("alerta_lolmu_fechas", border=True):
        c1, c2 = st.columns(2)
        with c1:
            emision = st.date_input(
                "Fecha de emisión", value=ultima.date(), min_value=pd.Timestamp(anio, 2, 1).date(),
                max_value=ultima.date(), format="DD/MM/YYYY", key="alerta_lolmu_emision",
                help=f"Por defecto, el último día con clima observado ({ultima:%d/%m/%Y}).")
        with c2:
            primeras = st.date_input(
                "Fecha en que se vieron las primeras plántulas (opcional)",
                value=vistas_conteo.date() if vistas_conteo is not None else None,
                min_value=pd.Timestamp(anio, 1, 1).date(), max_value=ultima.date(), format="DD/MM/YYYY",
                key="alerta_lolmu_primeras",
                help="Un solo dato por campaña: el día en que aparecieron las primeras plántulas en el lote. Ancla el reloj "
                     "térmico de la ventana de control y mejora la alerta. Si hay conteos cargados se toma el primero con "
                     "plántulas. Para borrarla, vacíe el campo.")
        st.form_submit_button("Actualizar alerta", type="primary")
    emision = pd.Timestamp(emision).normalize()
    primeras = pd.Timestamp(primeras) if primeras else None

    tt = dict(t_base=t_base, t_opt=t_opt, t_crit=t_crit)
    r = al.evaluar(meteo, emision, primeras, conteos, modelo, **tt)
    p, nivel = r["probabilidad"], r["nivel"]
    st.markdown(
        f"<div style='background:{COLOR[nivel]};color:white;padding:0.9rem 1.1rem;border-radius:0.6rem;margin:0.4rem 0 0.8rem 0;"
        f"font-size:1.05rem'><b>{emision:%d/%m/%Y}</b> · <b>{NOMBRE[nivel]}</b> · probabilidad de flush en los próximos 7 días: "
        f"<b>{p:.0%}</b>. {TEXTO[nivel]}<br><span style='font-size:0.9rem;opacity:0.9'>Información usada: {INFO[r['nivel_info']]}.</span></div>",
        unsafe_allow_html=True)

    clima = al.limpiar_clima(meteo)
    dg = al.tiempo_termico(clima, **tt)
    ancla, ancla_txt = (r["primeras_plantulas"], "primeras plántulas observadas") if r["primeras_plantulas"] is not None else (
        (pd.Timestamp(inicio_predicho).normalize(), "inicio estimado por PREDWEEM") if inicio_predicho is not None and pd.Timestamp(inicio_predicho) <= emision
        else (None, None))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad de flush (7 días)", f"{p:.0%}", help="Tasa histórica: ~25 % de las semanas de campaña.")
    m2.metric("Emergencia esperada por lluvias", f"{100 * r['senales']['lluvia']:.1f} %",
              help="Respuesta relativa a las lluvias registradas hasta la fecha (señal, no % de la campaña).")
    if ancla is not None:
        tt_hoy = float(dg[(dg.index >= ancla) & (dg.index <= emision)].sum())
        f_ctl, f_lim = _fecha_tt(dg, ancla, tt_control), _fecha_tt(dg, ancla, tt_limite)
        m3.metric("Tiempo térmico desde el inicio", f"{tt_hoy:.0f} °Cd", help=f"Desde el {ancla:%d/%m} ({ancla_txt}).")
        m4.metric(f"Control post-emergente ({tt_control:.0f} °Cd)", f"{f_ctl:%d/%m}" if f_ctl is not None else "—",
                  help=(f"Límite de la ventana ({tt_limite:.0f} °Cd): {f_lim:%d/%m}" if f_lim is not None else "Fuera del clima disponible.")
                       + (" · Estimado: informe la fecha de las primeras plántulas para anclarlo." if r["primeras_plantulas"] is None else ""))
    else:
        m3.metric("Tiempo térmico desde el inicio", "—", help="Todavía no hay inicio observado ni estimado.")
        m4.metric(f"Control post-emergente ({tt_control:.0f} °Cd)", "—")
    if r["primeras_plantulas"] is None:
        st.info("Sin fecha de primeras plántulas: la alerta usa solo el clima. Cuando aparezcan las primeras plántulas "
                "en el lote, cargue esa fecha arriba: ancla el reloj térmico de la ventana de control y mejora la alerta.")

    # gráfico único: alerta semanal (arriba) y emergencia acumulada (abajo), con la ventana de control en tiempo térmico
    s = al.serie(meteo, pd.Timestamp(anio, 2, 2), emision, primeras, conteos, modelo, paso=7, **tt)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.56, 0.44], vertical_spacing=0.06,
                        subplot_titles=("Probabilidad de flush en los próximos 7 días", "Emergencia acumulada esperada (calendario de 17 campañas)"))
    fig.add_hrect(y0=100 * al.ATENCION, y1=100 * al.ALERTA, fillcolor=COLOR["atencion"], opacity=0.08, line_width=0, exclude_empty_subplots=False, row=1, col=1)
    fig.add_hrect(y0=100 * al.ALERTA, y1=100, fillcolor=COLOR["alerta"], opacity=0.06, line_width=0, exclude_empty_subplots=False, row=1, col=1)
    fig.add_hline(y=100 * al.ATENCION, line_dash="dot", line_color=COLOR["atencion"], annotation_text=f"Atención ≥ {al.ATENCION:.0%}",
                  annotation_position="bottom left", annotation_font_color=COLOR["atencion"], exclude_empty_subplots=False, row=1, col=1)
    fig.add_hline(y=100 * al.ALERTA, line_dash="dash", line_color=COLOR["alerta"], annotation_text=f"Alerta ≥ {al.ALERTA:.0%}",
                  annotation_position="top left", annotation_font_color=COLOR["alerta"], exclude_empty_subplots=False, row=1, col=1)
    x_fin = max(emision + pd.Timedelta(days=14), pd.Timestamp(anio, 6, 30))
    if ancla is not None:
        estimado = r["primeras_plantulas"] is None
        f_ctl, f_lim = _fecha_tt(dg, ancla, tt_control), _fecha_tt(dg, ancla, tt_limite)
        if f_ctl is not None:
            fin_ventana = f_lim if f_lim is not None else dg.index.max()
            x_fin = max(x_fin, fin_ventana + pd.Timedelta(days=7))
            for fila in (1, 2):
                fig.add_vrect(x0=f_ctl, x1=fin_ventana, fillcolor="#7c3aed", opacity=0.14, line_width=0, exclude_empty_subplots=False, row=fila, col=1)
            fig.add_annotation(x=f_ctl + (fin_ventana - f_ctl) / 2, y=97, xref="x", yref="y", showarrow=False, font=dict(size=12, color="#5b21b6"),
                               text=f"Ventana de control {tt_control:.0f}–{tt_limite:.0f} °Cd" + (" (estimada)" if estimado else "")
                                    + f"<br>{f_ctl:%d/%m} – " + (f"{f_lim:%d/%m}" if f_lim is not None else "fuera del clima disponible"))
        for fila in (1, 2):
            fig.add_vline(x=ancla, line_dash="dot", line_color="#374151", exclude_empty_subplots=False, row=fila, col=1)
        fig.add_annotation(x=ancla, y=4, xref="x", yref="y", showarrow=False, xanchor="left", font=dict(size=11, color="#374151"),
                           text=f" Inicio {ancla:%d/%m} ({'estimado' if estimado else 'primeras plántulas'})")
    for fila in (1, 2):
        fig.add_vline(x=emision, line_dash="dash", line_color="#9ca3af", exclude_empty_subplots=False, row=fila, col=1)
    fig.add_annotation(x=emision, y=4, xref="x", yref="y", showarrow=False, xanchor="left", font=dict(size=11, color="#6b7280"),
                       text=f" emisión {emision:%d/%m}")
    fig.add_trace(go.Scatter(x=s["Fecha"], y=100 * s["probabilidad"], mode="lines+markers", name="Probabilidad de flush (7 días)",
                             line=dict(color="#2a78d6", width=2),
                             marker=dict(size=9, color=[COLOR[al.nivel_de(v)] for v in s["probabilidad"]], line=dict(color="white", width=1)),
                             customdata=np.c_[[INFO[v] for v in s["nivel_info"]], [NOMBRE[al.nivel_de(v)] for v in s["probabilidad"]]],
                             hovertemplate="%{x|%d/%m}: %{y:.0f} % · %{customdata[1]}<br>%{customdata[0]}<extra></extra>"), row=1, col=1)
    # emergencia acumulada: curva media y rango 10–90 % de las 17 campañas; marca del valor esperado a la fecha de emisión
    cv = modelo.curva(pd.date_range(pd.Timestamp(anio, 1, 15), x_fin))
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(21,128,61,0.16)", name="Rango 10–90 % de las campañas", hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["media"], mode="lines", name="Emergencia acumulada esperada",
                             line=dict(color="#15803d", width=2.5),
                             customdata=np.c_[100 * cv["p10"], 100 * cv["p90"]],
                             hovertemplate="%{x|%d/%m}: %{y:.0f} % (rango %{customdata[0]:.0f}–%{customdata[1]:.0f} %)<extra></extra>"), row=2, col=1)
    hoy = modelo.curva(pd.DatetimeIndex([emision])).iloc[0]
    fig.add_trace(go.Scatter(x=[emision], y=[100 * hoy["media"]], mode="markers+text", marker=dict(size=10, color="#15803d", line=dict(color="white", width=1.5)),
                             text=[f"{100 * hoy['media']:.0f} % ({100 * hoy['p10']:.0f}–{100 * hoy['p90']:.0f} %)"], textposition="middle left",
                             textfont=dict(size=11, color="#14532d"), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Prob. de flush (%)", range=[0, 106], row=1, col=1)
    fig.update_yaxes(title_text="% de la campaña", range=[0, 102], row=2, col=1)
    fig.update_xaxes(range=[pd.Timestamp(anio, 2, 1), x_fin], tickformat="%d/%m")
    fig.update_layout(height=640, margin=dict(t=40, b=10, l=10, r=10), showlegend=False, hovermode="closest",
                      title=dict(text="Alerta de flushes, emergencia acumulada y ventana de control", font=dict(size=15)))
    fig.update_annotations(selector=dict(text="Probabilidad de flush en los próximos 7 días"), font_size=12)
    fig.update_annotations(selector=dict(text="Emergencia acumulada esperada (calendario de 17 campañas)"), font_size=12)
    st.plotly_chart(fig, width="stretch")
    st.caption("Arriba, la alerta emitida cada semana con la información disponible en ese momento (color = nivel). Abajo, la "
               "emergencia acumulada esperada para Bordenave: curva media y rango 10–90 % de 17 campañas (error típico ±13 puntos). "
               "La franja violeta es la ventana de control post-emergente en tiempo térmico desde el inicio.")

    with st.expander("Cómo se validó esta alerta"):
        v = modelo.p.get("validacion", {})
        filas = [(k, d) for k, d in v.items()]
        if filas:
            st.markdown("| Información disponible | AUC | Flushes detectados con Alerta (≥ 25 %) | Semanas marcadas |\n|---|---|---|---|\n"
                        + "\n".join(f"| {k} | {d['auc']:.2f} | {d['det_at']:.0%} | {d['vis_at']:.0%} |" for k, d in filas))
        st.markdown(
            f"{modelo.p.get('fuente', '')}\n\nCada campaña se predijo con el modelo ajustado sin ella. Las probabilidades están "
            "calibradas (cuando dice ~25 %, el flush ocurrió en ~27 % de las semanas). Como referencia, con las mismas campañas "
            "el motor PREDWEEM de arriba da AUC ~0,70 para anticipar flushes, pero anticipa bien la primera cohorte, por eso se "
            "usa como inicio estimado del reloj térmico hasta que se informan las primeras plántulas.\n\n"
            "**Límites.** Un sitio y lotes sin labranza. La fecha de primeras plántulas mejora la alerta en forma moderada "
            "(AUC +0,04); los conteos semanales, en forma más clara (+0,07). No anticipa flushes sin lluvia registrada.")
