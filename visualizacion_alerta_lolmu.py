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
TEXTO_TOPE = ("Lluvias y calendario favorables a un flush, pero todavía no se informaron plántulas: recorrer el lote para "
              "detectar el inicio (sin plántulas, la alerta llega como máximo a Atención).")
INFO = {1: "solo clima (sin datos del lote)", 2: "clima + fecha de primeras plántulas", 3: "clima + primeras plántulas + conteos"}


def _ultima_observada(meteo: pd.DataFrame) -> pd.Timestamp:
    m = meteo.copy()
    fechas = pd.to_datetime(m["Fecha"], errors="coerce")
    col_tipo = next((c for c in ("TipoDato", "TIPO") if c in m.columns), None)
    if col_tipo:
        # solo el tramo de pronóstico vigente; "Historico_pronostico" (pronósticos archivados) cuenta como serie usada
        obs = fechas[m[col_tipo].astype(str).str.strip().str.lower() != "pronostico"]
        if not obs.empty:
            return obs.max().normalize()
    return fechas.max().normalize()


def _parsear_fechas(valores: pd.Series) -> pd.Series:
    """Fechas de una columna de planilla: fechas de Excel, texto ISO (AAAA-MM-DD) o texto DD/MM/AAAA.

    Las columnas numéricas (conteos) no se interpretan como fechas.
    """
    v = valores.dropna()
    if v.empty:
        return pd.Series(pd.NaT, index=valores.index)
    if pd.to_numeric(v, errors="coerce").notna().mean() > 0.5:
        return pd.Series(pd.NaT, index=valores.index)
    texto = valores.astype(str).str.strip()
    iso = pd.to_datetime(texto.where(texto.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")), errors="coerce", format="ISO8601")
    otros = pd.to_datetime(valores.where(iso.isna()), errors="coerce", dayfirst=True, format="mixed")
    return iso.fillna(otros)


def _leer_planilla_conteos(archivo, columnas: list | None = None) -> tuple[pd.DataFrame, list]:
    """Lee una planilla de conteos (Excel o CSV): una columna de fechas y una o más columnas de marcos.

    Busca la fila de encabezado con «Fecha»; promedia las columnas numéricas de cada fila (plántulas por marco).
    """
    nombre = getattr(archivo, "name", "").lower()
    crudo = pd.read_csv(archivo, header=None) if nombre.endswith(".csv") else pd.read_excel(archivo, header=None)
    fila = next((i for i, r in crudo.iterrows() if any(str(v).strip().lower().startswith("fecha") for v in r.values)), None)
    tabla = crudo.iloc[fila + 1:].copy() if fila is not None else crudo.copy()
    if fila is not None:
        tabla.columns = [str(int(c)) if isinstance(c, float) and c.is_integer() else str(c) for c in crudo.iloc[fila].values]
    fechas = None
    for c in tabla.columns:
        f = _parsear_fechas(tabla[c])
        if f.notna().mean() > 0.5:
            fechas = f
            otras = [k for k in tabla.columns if k != c]
            break
    if fechas is None:
        raise ValueError("no se encontró una columna de fechas")
    num = tabla[otras].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, [k for k in num.columns if num[k].notna().any()
                      and not any(p in str(k).lower() for p in ("prom", "media", "acum", "%", "total"))]]
    num.columns = [str(c) for c in num.columns]
    disponibles = list(num.columns)
    if columnas:
        num = num[[c for c in columnas if c in num.columns]]
    out = pd.DataFrame({"Fecha": fechas.dt.normalize(), "PLM2": num.mean(axis=1)}).dropna()
    return out.sort_values("Fecha").drop_duplicates("Fecha", keep="last").reset_index(drop=True), disponibles


def _fecha_tt(dg: pd.Series, desde: pd.Timestamp, objetivo: float):
    acum = dg[dg.index >= desde].cumsum()
    hit = acum[acum >= objetivo]
    return hit.index[0] if not hit.empty else None


def mostrar_alerta_lolmu(meteo_raw: pd.DataFrame, campo: pd.DataFrame | None = None, col_fecha: str | None = None,
                         col_plm2: str | None = None, inicio_predicho: pd.Timestamp | None = None,
                         t_base: float = 2.0, t_opt: float = 20.0, t_crit: float = 30.0,
                         tt_control: float = 600.0, tt_limite: float = 800.0, localidad: str = "Bordenave",
                         nota_sitio: str | None = None, calendario_local: dict | None = None,
                         calendario_en_alerta: bool = False, agotamiento: dict | None = None,
                         tope_sin_inicio: bool = False, cero_reciente_dias: int | None = None, tope_hasta_ultimo_cero: bool = False,
                         curva_desde_ultimo_cero: bool = False) -> dict | None:
    """``calendario_local`` (opcional): curva acumulada propia del sitio para el panel de emergencia acumulada, con claves
    ``dia`` (día del año), ``media``, ``bajo``, ``alto`` (fracciones 0–1), ``campanias`` (lista de años) y ``rango``
    (texto del rango, p. ej. "mínimo–máximo"). La probabilidad de flush sigue usando el calendario de Bordenave."""
    calibrada_aqui = localidad == "Bordenave"
    if calendario_local:
        _d = np.asarray(calendario_local["dia"], float)
        _cols = {k: np.asarray(calendario_local[c], float) for k, c in (("media", "media"), ("p10", "bajo"), ("p90", "alto"))}
        _n_cal = len(calendario_local.get("campanias", []))
        _anios = ", ".join(str(a) for a in calendario_local.get("campanias", []))

        def _curva(fechas):
            dd = pd.DatetimeIndex(fechas).dayofyear
            return pd.DataFrame({"Fecha": fechas, **{k: np.interp(dd, _d, v) for k, v in _cols.items()}})

        rango_txt = calendario_local.get("rango", "mínimo–máximo")
        titulo_cal = f"Emergencia acumulada esperada (calendario de {localidad}, {_n_cal} campañas)"
    else:
        _curva = None
        rango_txt = "10–90 %"
        titulo_cal = "Emergencia acumulada esperada (calendario de 17 campañas)"
    st.subheader("🔔 Alerta de flushes validada (17 campañas)" if calibrada_aqui
                 else f"🔔 Alerta de flushes · {localidad} (modelo calibrado en Bordenave, 17 campañas)")
    st.caption("Probabilidad de que en los próximos 7 días emerja ≥ 5 % de la campaña. Combina el calendario medio de "
               "LOLMU, la respuesta a las lluvias ya registradas y, si se informa, la fecha de las primeras plántulas "
               "del lote. Independiente del motor PREDWEEM de arriba.")
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
                "Fecha de emisión", value=ultima.date(), min_value=min(pd.Timestamp(anio, 2, 1), ultima).date(),
                max_value=ultima.date(), format="DD/MM/YYYY", key="alerta_lolmu_emision",
                help=f"Por defecto, el último día con clima observado ({ultima:%d/%m/%Y}).")
        with c2:
            primeras = st.date_input(
                "Fecha en que se vieron las primeras plántulas (opcional)",
                value=vistas_conteo.date() if vistas_conteo is not None else None,
                min_value=pd.Timestamp(anio, 1, 1).date(), max_value=ultima.date(), format="DD/MM/YYYY",
                key="alerta_lolmu_primeras",
                help="Un solo dato por campaña: el día en que aparecieron las primeras plántulas en el lote. Ancla el reloj "
                     "térmico de la ventana de control y mejora la alerta. Para borrarla, vacíe el campo.")
        with st.expander("🌱 Conteos de plántulas (opcional)", expanded=False):
            st.caption("Si se pueden hacer recorridas, cargue las plántulas nuevas en cada fecha (por marco o por m²; solo "
                       "importa la proporción entre fechas). Con conteos semanales la alerta mejora (AUC 0,87 contra 0,84 "
                       "solo con la fecha de primeras plántulas) y, si no se informó esa fecha, se toma del primer conteo con plántulas.")
            base = st.session_state.get("alerta_lolmu_tabla_base",
                                        pd.DataFrame({"Fecha": pd.Series([pd.NaT] * 4, dtype="datetime64[ns]"),
                                                      "Plántulas": pd.Series([np.nan] * 4, dtype="float")}))
            tabla = st.data_editor(base, num_rows="dynamic", width="stretch", key="alerta_lolmu_tabla",
                                   column_config={"Fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                                                  "Plántulas": st.column_config.NumberColumn("Plántulas nuevas", min_value=0.0)})
            archivo = st.file_uploader("…o subir una planilla (Excel o CSV: columna «Fecha» y una columna por marco)",
                                       type=["xlsx", "xls", "csv"], key="alerta_lolmu_archivo")
        st.form_submit_button("Actualizar alerta", type="primary")
    emision = pd.Timestamp(emision).normalize()
    cargados = None
    if archivo is not None:
        try:
            _, disponibles = _leer_planilla_conteos(archivo)
            archivo.seek(0)
            elegidas = disponibles
            if len(disponibles) > 1:
                elegidas = st.multiselect(
                    "Marcos de la planilla a usar", disponibles, default=disponibles, key="alerta_lolmu_columnas",
                    help="Si la planilla tiene varios tratamientos (por ejemplo, con y sin remoción), deje solo los marcos "
                         "del lote que quiere evaluar.") or disponibles
            cargados, _ = _leer_planilla_conteos(archivo, elegidas)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer la planilla de conteos: {e}")
    if cargados is None and tabla is not None and not tabla.empty:
        t = pd.DataFrame({"Fecha": pd.to_datetime(tabla["Fecha"], errors="coerce").dt.normalize(),
                          "PLM2": pd.to_numeric(tabla["Plántulas"], errors="coerce")}).dropna()
        cargados = t.sort_values("Fecha").drop_duplicates("Fecha", keep="last") if not t.empty else None
    if cargados is not None:
        cargados = cargados[cargados["Fecha"].dt.year == anio]
        conteos = cargados if not cargados.empty else conteos
        if not cargados.empty:
            st.caption(f"Conteos cargados: {len(cargados)} fechas ({cargados['Fecha'].min():%d/%m} – {cargados['Fecha'].max():%d/%m}); "
                       "se usan los que son anteriores o iguales a la fecha de emisión.")
    primeras = pd.Timestamp(primeras) if primeras else None

    tt = dict(t_base=t_base, t_opt=t_opt, t_crit=t_crit)
    # opciones del sitio: calendario local también dentro de la alerta y agotamiento de la cohorte
    opc = {}
    if calendario_en_alerta and calendario_local:
        opc["calendario"] = {"dia": calendario_local["dia"], "media": calendario_local["media"]}
    if agotamiento:
        opc["agotamiento"] = agotamiento
    if tope_sin_inicio:
        opc["tope_sin_inicio"] = True
    if cero_reciente_dias is not None:
        opc["cero_reciente_dias"] = int(cero_reciente_dias)
    if tope_hasta_ultimo_cero:
        opc["tope_hasta_ultimo_cero"] = True
    r = al.evaluar(meteo, emision, primeras, conteos, modelo, **tt, **opc)
    p, nivel = r["probabilidad"], r["nivel"]
    st.markdown(
        f"<div style='background:{COLOR[nivel]};color:white;padding:0.9rem 1.1rem;border-radius:0.6rem;margin:0.4rem 0 0.8rem 0;"
        f"font-size:1.05rem'><b>{emision:%d/%m/%Y}</b> · <b>{NOMBRE[nivel]}</b> · probabilidad de flush en los próximos 7 días: "
        + (f"<b>{p:.0%}</b>. Conteo en cero el {r['cero_reciente']:%d/%m}: esta semana ya pasó sin plántulas en el lote, por eso la "
           f"probabilidad queda por debajo de Atención (por clima sería {r['probabilidad_clima']:.0%}). "
           "La semana que sigue al último conteo en cero se evalúa normalmente.<br>" if r.get("cero_reciente") is not None else
           f"<b>{p:.0%}</b>. {TEXTO_TOPE if r.get('tope_aplicado') else TEXTO[nivel]}<br>")
        + f"<span style='font-size:0.9rem;opacity:0.9'>Información usada: {'clima + conteo en cero posterior (semana sin plántulas)' if r.get('cero_reciente') is not None else INFO[r['nivel_info']]}.</span></div>",
        unsafe_allow_html=True)
    pt = r.get("probabilidad_tardia")
    if pt is not None:
        tardia_si = pt >= al.UMBRAL_TARDIA
        st.markdown(
            f"<div style='background:{'#6d28d9' if tardia_si else '#ede9fe'};color:{'white' if tardia_si else '#3b0764'};"
            "padding:0.75rem 1.1rem;border-radius:0.6rem;margin:-0.3rem 0 0.8rem 0;font-size:1.0rem'>"
            f"🌾 <b>Ventana de siembra · {'Emergencia tardía probable' if tardia_si else 'Emergencia tardía poco probable'}</b> · "
            f"probabilidad de que emerja al menos 2 % de la campaña en los próximos 7 días: <b>{pt:.0%}</b> "
            f"(aviso desde {al.UMBRAL_TARDIA:.0%}). "
            + ("Antes de definir la fecha de siembra, conviene recorrer el lote: puede haber una cohorte tardía por controlar."
               if tardia_si else "Con la información disponible no se espera una cohorte tardía relevante esta semana.")
            + "<br><span style='font-size:0.88rem;opacity:0.9'>Del 1/05 al 31/08. En esta época los flushes de ≥ 5 % son raros, pero "
            "pocas plantas por semana pueden importar para la fecha de siembra; con conteos semanales el aviso es más preciso.</span></div>",
            unsafe_allow_html=True)

    clima = al.limpiar_clima(meteo)
    dg = al.tiempo_termico(clima, **tt)
    ancla, ancla_txt = (r["primeras_plantulas"], "primeras plántulas observadas") if r["primeras_plantulas"] is not None else (
        (pd.Timestamp(inicio_predicho).normalize(), "inicio estimado por PREDWEEM") if inicio_predicho is not None and pd.Timestamp(inicio_predicho) <= emision
        else (None, None))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad de flush (7 días)", f"{p:.0%}", help="Tasa histórica: ~25 % de las semanas de campaña."
              + (f" Conteo en cero el {r['cero_reciente']:%d/%m}: semana ya pasada sin plántulas, por debajo de Atención (por clima {r['probabilidad_clima']:.0%})."
                 if r.get("cero_reciente") is not None else ""))
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
    s = al.serie(meteo, pd.Timestamp(anio, 2, 2), emision, primeras, conteos, modelo, paso=7, **tt, **opc)
    if s.empty:  # comienzo de campaña (antes del 2 de febrero): todavía no hay emisiones semanales
        s = pd.DataFrame({"Fecha": pd.DatetimeIndex([]), "probabilidad": pd.Series(dtype=float),
                          "probabilidad_clima": pd.Series(dtype=float), "nivel_info": pd.Series(dtype=int),
                          "nivel": pd.Series(dtype=str), "cero_reciente": pd.Series(dtype=bool),
                          "probabilidad_tardia": pd.Series(dtype=float)})
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.56, 0.44], vertical_spacing=0.06,
                        subplot_titles=("Probabilidad de flush en los próximos 7 días", titulo_cal))
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
    # semanas ya cerradas por un conteo posterior en cero (sin plántulas): probabilidad por debajo de Atención
    sin_inicio = s["cero_reciente"].astype(bool) if "cero_reciente" in s else pd.Series(False, index=s.index)
    y_plot = 100 * s["probabilidad"]
    etiqueta = [f"semana sin plántulas según el conteo en cero (por clima {100 * pc:.0f} %)" if z else INFO[n]
                for z, pc, n in zip(sin_inicio, s.get("probabilidad_clima", s["probabilidad"]), s["nivel_info"])]
    # cada probabilidad se dibuja sobre la semana que anticipa (de la fecha de emisión a 7 días después): tramos
    # escalonados para las semanas anteriores a la emisión y un tramo punteado para los próximos 7 días
    ini_sem = s["Fecha"]
    fin_sem = (ini_sem + pd.Timedelta(days=7)).clip(upper=emision)
    pasadas = fin_sem > ini_sem
    if pasadas.any():
        sp = s[pasadas]
        fig.add_trace(go.Scatter(x=list(sp["Fecha"]) + [fin_sem[pasadas].iloc[-1]],
                                 y=list(100 * sp["probabilidad"]) + [100 * sp["probabilidad"].iloc[-1]],
                                 mode="lines", line=dict(color="#2a78d6", width=2.5, shape="hv"), hoverinfo="skip",
                                 name="Probabilidad de flush de cada semana"), row=1, col=1)
        medio = ini_sem[pasadas] + (fin_sem[pasadas] - ini_sem[pasadas]) / 2
        et = [e for e, k in zip(etiqueta, pasadas) if k]
        fig.add_trace(go.Scatter(x=medio, y=100 * sp["probabilidad"], mode="markers", showlegend=False,
                                 marker=dict(size=9, color=[COLOR[v] for v in sp["nivel"]], line=dict(color="white", width=1)),
                                 customdata=np.c_[sp["Fecha"].dt.strftime("%d/%m"), fin_sem[pasadas].dt.strftime("%d/%m"), et,
                                                  [NOMBRE[v] for v in sp["nivel"]]],
                                 hovertemplate="Semana %{customdata[0]}–%{customdata[1]}: %{y:.0f} % · %{customdata[3]}"
                                               "<br>emitida el %{customdata[0]} · %{customdata[2]}<extra></extra>"), row=1, col=1)
    fin_prox = emision + pd.Timedelta(days=7)
    fig.add_trace(go.Scatter(x=[emision, fin_prox], y=[100 * p, 100 * p], mode="lines", line=dict(color=COLOR[nivel], width=3.5, dash="dot"),
                             hoverinfo="skip", name="Próximos 7 días"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[emision + pd.Timedelta(days=3.5)], y=[100 * p], mode="markers+text", showlegend=False,
                             marker=dict(size=11, color=COLOR[nivel], symbol="circle-open", line=dict(width=2.5)),
                             text=[f"{100 * p:.0f} %"], textposition="top center", textfont=dict(size=11, color=COLOR[nivel]),
                             hovertemplate=f"Próximos 7 días ({emision:%d/%m}–{fin_prox:%d/%m}): %{{y:.0f}} % · {NOMBRE[nivel]}<extra></extra>"),
                  row=1, col=1)
    if "probabilidad_tardia" in s and (s["probabilidad_tardia"].notna().any() or r.get("probabilidad_tardia") is not None):
        st_ = s[s["probabilidad_tardia"].notna() & pasadas]
        if not st_.empty:
            fin_t = (st_["Fecha"] + pd.Timedelta(days=7)).clip(upper=emision)
            fig.add_trace(go.Scatter(x=list(st_["Fecha"]) + [fin_t.iloc[-1]], y=list(100 * st_["probabilidad_tardia"]) + [100 * st_["probabilidad_tardia"].iloc[-1]],
                                     mode="lines", line=dict(color="#6d28d9", width=2, dash="dash", shape="hv"), hoverinfo="skip",
                                     name="Emergencia tardía (≥ 2 % en 7 días)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=st_["Fecha"] + (fin_t - st_["Fecha"]) / 2, y=100 * st_["probabilidad_tardia"], mode="markers", showlegend=False,
                                     marker=dict(size=7, color="#6d28d9", symbol="diamond"),
                                     customdata=np.c_[st_["Fecha"].dt.strftime("%d/%m"), fin_t.dt.strftime("%d/%m")],
                                     hovertemplate="Semana %{customdata[0]}–%{customdata[1]}: emergencia tardía %{y:.0f} %<extra></extra>"), row=1, col=1)
        if r.get("probabilidad_tardia") is not None:
            pt_ = 100 * r["probabilidad_tardia"]
            fig.add_trace(go.Scatter(x=[emision, fin_prox], y=[pt_, pt_], mode="lines", line=dict(color="#6d28d9", width=2.5, dash="dot"),
                                     hovertemplate=f"Próximos 7 días: emergencia tardía {pt_:.0f} %<extra></extra>", showlegend=False), row=1, col=1)
        fig.add_shape(type="line", x0=pd.Timestamp(anio, 5, 1), x1=pd.Timestamp(anio, 8, 31), y0=100 * al.UMBRAL_TARDIA,
                      y1=100 * al.UMBRAL_TARDIA, line=dict(color="#6d28d9", width=1, dash="dashdot"), row=1, col=1)
        fig.add_annotation(x=pd.Timestamp(anio, 5, 1), y=100 * al.UMBRAL_TARDIA, xref="x", yref="y", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=10, color="#6d28d9"),
                           text=f"emergencia tardía ≥ {al.UMBRAL_TARDIA:.0%}")
    if conteos is not None and not conteos.empty:
        cc = conteos[conteos["Fecha"] <= emision]
        fig.add_trace(go.Scatter(x=cc["Fecha"], y=[2] * len(cc), mode="markers", name="Conteo",
                                 marker=dict(symbol="triangle-up", size=10, color=np.where(cc["PLM2"] > 0, "#15803d", "#9ca3af"),
                                             line=dict(color="white", width=1)),
                                 customdata=cc["PLM2"], hovertemplate="Conteo %{x|%d/%m}: %{customdata:.0f} plántulas<extra></extra>"),
                      row=1, col=1)
    # emergencia acumulada: curva media y rango 10–90 % de las 17 campañas; marca del valor esperado a la fecha de emisión
    curva_de = _curva or modelo.curva
    cv = curva_de(pd.date_range(pd.Timestamp(anio, 1, 15), x_fin))
    # conteos en cero antes de las primeras plántulas: la curva queda en 0 hasta la mitad del intervalo entre el último cero
    # y el primer conteo con plántulas (las plántulas de ese conteo salieron en algún momento del intervalo; sin conteo
    # positivo todavía, hasta el último cero) y reparte el resto
    # de la campaña desde esa fecha (curva condicionada a que hasta allí no hubo emergencia)
    t0 = None
    if curva_desde_ultimo_cero and conteos is not None and not conteos.empty:
        cc0 = conteos[conteos["Fecha"] <= emision].sort_values("Fecha")
        positivos = cc0[cc0["PLM2"] > 0]["Fecha"]
        if r["primeras_plantulas"] is not None:
            positivos = pd.concat([positivos, pd.Series([r["primeras_plantulas"]])])
        hasta = positivos.min() if not positivos.empty else emision + pd.Timedelta(days=1)
        ceros0 = cc0[(cc0["PLM2"] == 0) & (cc0["Fecha"] < hasta)]["Fecha"]
        t0 = ceros0.max() if not ceros0.empty else None
        if t0 is not None and not positivos.empty and positivos.min() > t0:
            t0 = (t0 + (positivos.min() - t0) / 2).normalize()  # mitad de camino entre el último cero y el primer positivo
    if t0 is not None:
        _curva_base = curva_de

        def curva_de(fechas, _b=_curva_base, _t0=t0):
            out = _b(pd.DatetimeIndex(fechas)).copy()
            ref = _b(pd.DatetimeIndex([_t0])).iloc[0]
            for k in ("media", "p10", "p90"):
                out[k] = np.where(pd.DatetimeIndex(out["Fecha"]) <= _t0, 0.0,
                                  np.clip((out[k] - ref[k]) / max(1 - ref[k], 1e-6), 0, 1))
            return out

        cv = curva_de(pd.date_range(pd.Timestamp(anio, 1, 15), x_fin))
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(21,128,61,0.16)", name=f"Rango {rango_txt} de las campañas", hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["media"], mode="lines", name="Emergencia acumulada esperada",
                             line=dict(color="#15803d", width=2.5),
                             customdata=np.c_[100 * cv["p10"], 100 * cv["p90"]],
                             hovertemplate="%{x|%d/%m}: %{y:.0f} % (rango %{customdata[0]:.0f}–%{customdata[1]:.0f} %)<extra></extra>"), row=2, col=1)
    hoy = curva_de(pd.DatetimeIndex([emision])).iloc[0]
    fig.add_trace(go.Scatter(x=[emision], y=[100 * hoy["media"]], mode="markers+text", marker=dict(size=10, color="#15803d", line=dict(color="white", width=1.5)),
                             text=[f"{100 * hoy['media']:.0f} % ({100 * hoy['p10']:.0f}–{100 * hoy['p90']:.0f} %)"], textposition="middle left",
                             textfont=dict(size=11, color="#14532d"), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Prob. de flush (%)", range=[0, 106], row=1, col=1)
    fig.update_yaxes(title_text="% de la campaña", range=[0, 102], row=2, col=1)
    fig.update_xaxes(range=[pd.Timestamp(anio, 2, 1), x_fin], tickformat="%d/%m")
    fig.update_layout(height=640, margin=dict(t=40, b=10, l=10, r=10), showlegend=False, hovermode="closest",
                      title=dict(text="Alerta de flushes, emergencia acumulada y ventana de control", font=dict(size=15)))
    fig.update_annotations(selector=dict(text="Probabilidad de flush en los próximos 7 días"), font_size=12)
    fig.update_annotations(selector=dict(text=titulo_cal), font_size=12)
    st.plotly_chart(fig, width="stretch")
    st.caption("Arriba, cada tramo horizontal es la probabilidad de flush de esa semana, emitida al comienzo de la semana con la "
               "información disponible ese día (color del punto = nivel); el tramo punteado, a la derecha de la fecha de emisión, es "
               "el pronóstico para los próximos 7 días. Un conteo en cero confirma las semanas que terminaron antes de esa fecha: "
               "esos tramos quedan por debajo de Atención. Abajo, la "
               + ("emergencia acumulada esperada para Bordenave: curva media y rango 10–90 % de 17 campañas (error típico ±13 puntos). "
                  if calibrada_aqui else (f"emergencia acumulada esperada según el calendario propio de {localidad} (curva media y "
                  f"rango {rango_txt} de las campañas {_anios}"
                  + ("). " if calendario_en_alerta else "; la probabilidad de flush usa el calendario de Bordenave). ")
                  + ("Desde las primeras plántulas, la probabilidad incluye el agotamiento de la cohorte. " if agotamiento else "")
                  if calendario_local else "emergencia acumulada esperada según el calendario de Bordenave (curva media y rango 10–90 % "
                  "de 17 campañas), usado como referencia regional. "))
               + "La franja violeta es la ventana de control post-emergente en tiempo térmico desde el inicio. "
               "Del 1/05 al 31/08, la línea violeta es la probabilidad de emergencia tardía (≥ 2 % de la campaña en 7 días).")

    with st.expander("Cómo se validó esta alerta"):
        if nota_sitio:
            st.markdown(nota_sitio)
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
        tv = modelo.p.get("tardia")
        if tv:
            st.markdown(
                "**Emergencia tardía (ventana de siembra, 1/05–31/08).** Probabilidad de que en los próximos 7 días emerja al menos "
                f"2 % de la campaña; aviso desde {al.UMBRAL_TARDIA:.0%}. Es la misma alerta recalibrada para esa época, en la que los "
                "flushes de ≥ 5 % son raros pero pocas plantas por semana importan para decidir la fecha de siembra.\n\n"
                "| Información | AUC Bordenave | Detectadas | Semanas marcadas | AUC otras localidades | Detectadas | Marcadas |\n|---|---|---|---|---|---|---|\n"
                + "\n".join(f"| {k} | {d['auc']:.2f} | {d['det']:.0%} | {d['marcadas']:.0%} | "
                             + (f"{o['auc']:.2f} | {o['det']:.0%} | {o['marcadas']:.0%} |" if (o := tv['validacion_otras'].get(k if k != '+ conteos semanales' else '+ conteos')) else "— | — | — |")
                             for k, d in tv["validacion_bordenave"].items())
                + f"\n\n{tv['fuente']} En las otras localidades la probabilidad tiende a ser alta (promedio 19 % contra 10 % "
                "observado): el aviso detecta casi todas las semanas con emergencia tardía, con más falsas alarmas.")

    # ---------------------------------------------------------------
    # resultados de la sección, para la descarga final
    # ---------------------------------------------------------------
    f_ctl_r, f_lim_r = (_fecha_tt(dg, ancla, tt_control), _fecha_tt(dg, ancla, tt_limite)) if ancla is not None else (None, None)
    tt_r = float(dg[(dg.index >= ancla) & (dg.index <= emision)].sum()) if ancla is not None else None
    fmt_f = lambda d: pd.Timestamp(d).strftime("%d/%m/%Y") if d is not None and not pd.isna(d) else ""  # noqa: E731
    resumen = pd.DataFrame({"Indicador": [
        "Localidad", "Fecha de generación", "Fecha de emisión", "Nivel de alerta", "Probabilidad de flush en 7 días (%)",
        "Información usada", "Inicio del reloj térmico", "Origen del inicio", "Tiempo térmico desde el inicio (°Cd)",
        f"Fecha de control post-emergente ({tt_control:.0f} °Cd)", f"Fecha límite de la ventana ({tt_limite:.0f} °Cd)",
        "Emergencia acumulada esperada a la fecha (%)", f"Rango {rango_txt} de la emergencia acumulada (%)",
        "Señal de calendario (emergencia esperada en 7 días, %)", "Señal de lluvia (respuesta relativa, %)",
        "Conteos usados (fechas)", "Primeras plántulas informadas", "Inicio estimado por PREDWEEM"],
        "Valor": [
        f"{localidad} (Buenos Aires)", pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"), fmt_f(emision), NOMBRE[nivel], round(100 * p, 1),
        INFO[r["nivel_info"]], fmt_f(ancla), ancla_txt or "sin inicio", round(tt_r, 1) if tt_r is not None else "",
        fmt_f(f_ctl_r), fmt_f(f_lim_r) if f_lim_r is not None else ("fuera del clima disponible" if f_ctl_r is not None else ""),
        round(100 * hoy["media"], 1), f"{100 * hoy['p10']:.0f}–{100 * hoy['p90']:.0f}",
        round(100 * r["senales"]["cal"], 2), round(100 * r["senales"]["lluvia"], 2),
        int((conteos["Fecha"] <= emision).sum()) if conteos is not None else 0, fmt_f(r["primeras_plantulas"]),
        fmt_f(inicio_predicho)]})
    resumen = pd.concat([resumen, pd.DataFrame({"Indicador": ["Probabilidad de emergencia tardía (≥ 2 % de la campaña en 7 días, %)",
                                                              "Aviso de emergencia tardía (ventana de siembra)"],
                                                "Valor": [round(100 * pt, 1) if pt is not None else "fuera de la ventana (1/05–31/08)",
                                                          ("probable" if pt >= al.UMBRAL_TARDIA else "poco probable") if pt is not None else ""]})],
                        ignore_index=True)
    if agotamiento:
        _ag = r["senales"].get("agotamiento")
        resumen = pd.concat([resumen, pd.DataFrame({"Indicador": ["Agotamiento de la cohorte (factor sobre la probabilidad)"],
                                                    "Valor": [round(_ag, 3) if _ag is not None else "sin primeras plántulas: no se aplica"]})],
                            ignore_index=True)
    serie_out = s.assign(**{"Probabilidad de flush (%)": (100 * s["probabilidad"]).round(1),
                            "Semana sin plántulas (conteo en cero posterior)": s["cero_reciente"] if "cero_reciente" in s else False,
                            "Probabilidad por clima (%)": (100 * s.get("probabilidad_clima", s["probabilidad"])).round(1),
                            "Nivel": [NOMBRE[v] for v in s["nivel"]],
                            "Información usada": [INFO[v] for v in s["nivel_info"]],
                            "Emergencia tardía ≥ 2 % en 7 días (%)": (100 * s["probabilidad_tardia"]).round(1) if "probabilidad_tardia" in s else np.nan})[["Fecha", "Probabilidad de flush (%)", "Nivel", "Información usada", "Semana sin plántulas (conteo en cero posterior)", "Probabilidad por clima (%)", "Emergencia tardía ≥ 2 % en 7 días (%)"]]
    _lo, _hi = ("P10 (%)", "P90 (%)") if not calendario_local else ("Mínimo (%)", "Máximo (%)")
    curva_out = cv.assign(**{"Emergencia acumulada media (%)": (100 * cv["media"]).round(2), _lo: (100 * cv["p10"]).round(2),
                             _hi: (100 * cv["p90"]).round(2)})[["Fecha", "Emergencia acumulada media (%)", _lo, _hi]]
    if ancla is not None:
        dd = dg[dg.index >= ancla]
        tt_out = pd.DataFrame({"Fecha": dd.index, "TT diario (°Cd)": dd.values.round(2), "TT acumulado desde el inicio (°Cd)": dd.cumsum().values.round(1)})
        tt_out["Etapa"] = np.where(tt_out["TT acumulado desde el inicio (°Cd)"] < tt_control, "antes de la ventana",
                                   np.where(tt_out["TT acumulado desde el inicio (°Cd)"] <= tt_limite, "ventana de control", "después de la ventana"))
    else:
        tt_out = pd.DataFrame()
    clima_out = clima[clima["Fecha"] <= emision + pd.Timedelta(days=7)].rename(columns={"Prec": "Lluvia (mm)"})
    params = pd.DataFrame({"Parámetro": [
        "Umbral de Atención (%)", "Umbral de Alerta (%)", "Definición de flush", "Horizonte de la alerta (días)",
        "Aviso de emergencia tardía (%)", "Definición de emergencia tardía",
        "Temperatura base (°C)", "Temperatura óptima (°C)", "Temperatura crítica (°C)", "TT de control (°Cd)", "TT límite (°Cd)",
        "Calibración"], "Valor": [
        100 * al.ATENCION, 100 * al.ALERTA, "≥ 5 % de la emergencia de la campaña en 7 días", 7,
        100 * al.UMBRAL_TARDIA, "≥ 2 % de la emergencia de la campaña en 7 días, del 1/05 al 31/08",
        t_base, t_opt, t_crit, tt_control, tt_limite, modelo.p.get("fuente", "")]})
    val = pd.DataFrame([{"Información disponible": k, "AUC": d["auc"], "Flushes detectados con Alerta (%)": round(100 * d["det_at"], 1),
                         "Semanas marcadas con Alerta (%)": round(100 * d["vis_at"], 1)} for k, d in modelo.p.get("validacion", {}).items()])
    return {"Resumen": resumen, "Alerta_semanal": serie_out, "Emergencia_acumulada": curva_out, "Tiempo_termico": tt_out,
            "Conteos": conteos.rename(columns={"PLM2": "Plántulas"}) if conteos is not None else pd.DataFrame(),
            "Clima": clima_out, "Parametros": params, "Validacion": val}
