# -*- coding: utf-8 -*-
"""Sección de la app: alerta validada de grandes flujos (flushes), reloj térmico desde las primeras plántulas y calendario."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import alerta_lolmu as al

COLOR = {"alerta": "#c2410c", "atencion": "#d97706", "baja": "#15803d"}
NOMBRE = {"alerta": "Alerta", "atencion": "Atención", "baja": "Sin alerta"}
TEXTO = {"alerta": "Gran flujo probable en los próximos 7 días: recorrer el lote y preparar el control.",
         "atencion": "Probabilidad de un gran flujo por encima de lo habitual: conviene mirar el lote esta semana.",
         "baja": "Gran flujo poco probable en los próximos 7 días."}
TEXTO_TOPE = ("Lluvias y calendario favorables a un gran flujo, pero todavía no se informaron plántulas: recorrer el lote para "
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


def _tt_ventana(meteo: pd.DataFrame, dg: pd.Series, emision: pd.Timestamp, modelo,
                localidad: str | None = None) -> tuple[pd.Series, pd.Timestamp]:
    """Tiempo térmico diario conocido a la fecha de emisión, extendido con el clima medio.

    Se usa el clima hasta la emisión más los días de pronóstico cargados a continuación (``TipoDato`` = ``Pronostico``);
    los días posteriores observados no se usan, porque ese día no se conocían. Desde el día siguiente al último conocido
    se suma el tiempo térmico diario medio histórico de ese día del año en la localidad (``modelo.clima_tt``).
    Devuelve la serie y el último día conocido.
    """
    emision = pd.Timestamp(emision).normalize()
    ult = min(emision, dg.index.max())
    cols = {str(c).strip().lower(): c for c in meteo.columns}
    if "tipodato" in cols and ult == emision:
        f = pd.to_datetime(meteo[cols["fecha"]], errors="coerce").dt.tz_localize(None).dt.normalize()
        pron = set(f[meteo[cols["tipodato"]].astype(str).str.strip().eq("Pronostico").values])
        while ult + pd.Timedelta(days=1) in pron and ult + pd.Timedelta(days=1) <= dg.index.max():
            ult += pd.Timedelta(days=1)
    conocido = dg[dg.index <= ult]
    if not modelo.p.get("curva_tt"):
        return conocido, ult
    clima = modelo.clima_tt(localidad)
    dias = pd.date_range(ult + pd.Timedelta(days=1), periods=400)
    medio = pd.Series(np.asarray(clima, float)[np.minimum(dias.dayofyear.values, len(clima)) - 1], index=dias)
    return pd.concat([conocido, medio]), ult


def _pct(u: float) -> str:
    """20 % / 10 %."""
    return f"{100 * u:g} %".replace(".", ",")


def _texto_muestreo_inicio(meteo: pd.DataFrame, emision: pd.Timestamp, conteos: pd.DataFrame | None, modelo) -> str:
    """Recuadro de muestreo mientras no se informan plántulas: el inicio no se puede anticipar bien con el clima."""
    k = modelo.p.get("inicio", {})
    f = lambda md: pd.Timestamp(emision.year, md[0], md[1])  # noqa: E731
    p10, med, p90 = f(k.get("p10", [2, 7])), f(k.get("mediana", [2, 28])), f(k.get("p90", [3, 26]))
    reg = k.get("regla_lluvia", {"desde": [2, 1], "dias": 5, "mm": 20, "demora_dias": 5})
    desde = f(reg["desde"])
    partes = []
    if emision < desde:
        partes.append(f"Todavía es temprano: empezar a recorrer el lote cada semana desde el <b>{desde:%d/%m}</b>.")
    else:
        partes.append("<b>Recorrer el lote cada semana</b> hasta ver las primeras plántulas y cargar esa fecha arriba.")
    clima = al.limpiar_clima(meteo)
    clima = clima[(clima["Fecha"] >= desde - pd.Timedelta(days=reg["dias"] - 1)) & (clima["Fecha"] <= emision)]
    if not clima.empty:
        acum = clima.set_index("Fecha")["Prec"].rolling(f"{reg['dias']}D").sum()
        acum = acum[(acum.index >= desde) & (acum.index >= emision - pd.Timedelta(days=14))]
        if not acum.empty and acum.max() >= reg["mm"]:
            dia = acum.idxmax()
            partes.append(f"Llovieron <b>{acum.max():.0f} mm</b> en los {reg['dias']} días al {dia:%d/%m}: conviene recorrer el lote "
                          f"entre el {dia + pd.Timedelta(days=reg['demora_dias']):%d/%m} y el {dia + pd.Timedelta(days=reg['demora_dias'] + 7):%d/%m}.")
    if conteos is not None and not conteos.empty:
        cc = conteos[conteos["Fecha"] <= emision]
        if not cc.empty and (cc["PLM2"] <= 0).all():
            ult = cc["Fecha"].max()
            partes.append(f"Último conteo en cero: {ult:%d/%m}; el próximo, el {ult + pd.Timedelta(days=7):%d/%m}.")
    partes.append(f"El inicio de la emergencia suele darse entre el <b>{p10:%d/%m} y el {p90:%d/%m}</b> (mediana {med:%d/%m}), pero "
                  "varía mucho entre años y lotes y el clima no lo anticipa bien: lo que más mejora la alerta es la fecha de las "
                  "primeras plántulas, que ancla la curva acumulada, el reloj térmico y la ventana de control.")
    return ("<div style='background:#eff6ff;color:#1e3a8a;border:1px solid #bfdbfe;padding:0.75rem 1.1rem;border-radius:0.6rem;"
            "margin:0.2rem 0 0.8rem 0;font-size:0.98rem'>🗓️ <b>Muestreo hasta el inicio.</b> " + " ".join(partes) + "</div>")


def mostrar_alerta_lolmu(meteo_raw: pd.DataFrame, campo: pd.DataFrame | None = None, col_fecha: str | None = None,
                         col_plm2: str | None = None, inicio_predicho: pd.Timestamp | None = None,
                         t_base: float = 2.0, t_opt: float = 20.0, t_crit: float = 30.0,
                         tt_control: float = 600.0, tt_limite: float = 800.0, localidad: str = "Bordenave",
                         nota_sitio: str | None = None, calendario_local: dict | None = None,
                         calendario_en_alerta: bool = False, agotamiento: dict | None = None,
                         tope_sin_inicio: bool = False, cero_reciente_dias: int | None = None, tope_hasta_ultimo_cero: bool = False,
                         curva_desde_ultimo_cero: bool = False, curva_tiempo_termico: bool = False,
                         curva_desde_primeras: bool = False) -> dict | None:
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
    st.subheader("🔔 Alerta de grandes flujos validada (17 campañas)" if calibrada_aqui
                 else f"🔔 Alerta de grandes flujos · {localidad} (modelo calibrado en Bordenave, 17 campañas)")
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
        c1, c2, c3 = st.columns(3)
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
        with c3:
            siembra = st.date_input(
                "Fecha de siembra prevista (opcional)", value=None,
                min_value=pd.Timestamp(anio, 5, 1).date(), max_value=pd.Timestamp(anio, 7, 31).date(), format="DD/MM/YYYY",
                key="alerta_lolmu_siembra",
                help="Entre el 1/05 y el 31/07. Muestra el riesgo histórico de emergencia después de la siembra, como guía para "
                     "decidir un herbicida residual a la siembra.")
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
        f"font-size:1.05rem'><b>{emision:%d/%m/%Y}</b> · <b>{NOMBRE[nivel]}</b> · probabilidad de un gran flujo (≥ 5 % de la campaña) en los próximos 7 días: "
        + (f"<b>{p:.0%}</b>. Conteo en cero el {r['cero_reciente']:%d/%m}: esta semana ya pasó sin plántulas en el lote, por eso la "
           f"probabilidad es 0 % (por clima sería {r['probabilidad_clima']:.0%}). "
           "La semana que sigue al último conteo en cero se evalúa normalmente.<br>" if r.get("cero_reciente") is not None else
           f"<b>{p:.0%}</b>. {TEXTO_TOPE if r.get('tope_aplicado') else TEXTO[nivel]}<br>")
        + f"<span style='font-size:0.9rem;opacity:0.9'>Información usada: {'clima + conteo en cero posterior (semana sin plántulas)' if r.get('cero_reciente') is not None else INFO[r['nivel_info']]}.</span></div>",
        unsafe_allow_html=True)
    pt = r.get("probabilidad_tardia")
    if pt is not None:
        u_t = r.get("umbral_tardia", al.UMBRAL_TARDIA)
        tardia_si = pt >= u_t
        hl = al.historia_local(emision, modelo)
        # si en las campañas del sitio, en mediana, emergió menos de 3 % después de esta fecha (poco margen para un flujo
        # tardío de ≥ 2 % en una semana), el aviso se presenta como «posible»: la recalibración de tardíos se ajustó con
        # Bordenave, donde la emergencia se extiende hasta el invierno, y en sitios donde se corta antes avisa de más
        corte_local = hl is not None and hl["mediana"] < 0.03
        if hl is None:
            _txt_hl = ""
        elif len(hl["valores"]) <= 4:
            _txt_hl = (f"<br><span style='font-size:0.88rem;opacity:0.95'>En {hl['sitio']}, después del {emision:%d/%m} emergió: "
                       + " · ".join(f"{a} {100 * v:.0f} %" for a, v in sorted(hl["valores"].items())) + " de la campaña.</span>")
        else:
            _vv = np.array(list(hl["valores"].values()))
            _txt_hl = (f"<br><span style='font-size:0.88rem;opacity:0.95'>En {hl['sitio']}, después del {emision:%d/%m} emergió en mediana "
                       f"{100 * hl['mediana']:.0f} % de la campaña (entre {100 * _vv.min():.0f} y {100 * _vv.max():.0f} % según el año).</span>")
        _tit = ("Flujo tardío posible según el modelo; en " + hl["sitio"] + " la emergencia suele cortarse antes") if (tardia_si and corte_local) else (
            "Flujo tardío probable" if tardia_si else "Flujo tardío poco probable")
        _fondo, _letra = ("#6d28d9", "white") if (tardia_si and not corte_local) else (("#c4b5fd", "#2e1065") if tardia_si else ("#ede9fe", "#3b0764"))
        st.markdown(
            f"<div style='background:{_fondo};color:{_letra};"
            "padding:0.75rem 1.1rem;border-radius:0.6rem;margin:-0.3rem 0 0.8rem 0;font-size:1.0rem'>"
            f"🌾 <b>Flujos tardíos · {_tit}</b> · "
            f"probabilidad de un flujo tardío (≥ 2 % de la campaña) en los próximos 7 días: <b>{pt:.0%}</b> "
            f"(aviso desde {_pct(u_t)}{', más sensible de julio a septiembre' if u_t < al.UMBRAL_TARDIA else ''}). "
            + ("Antes de definir la fecha de siembra, conviene recorrer el lote: puede haber una cohorte tardía por controlar."
               if tardia_si else "Con la información disponible no se espera una cohorte tardía relevante esta semana.")
            + f"<br><span style='font-size:0.88rem;opacity:0.9'>Del {al.limites_ventana_siembra(anio, modelo)[0]:%d/%m} al "
            f"{al.limites_ventana_siembra(anio, modelo)[1]:%d/%m}. En esta época los grandes flujos (≥ 5 %) son raros, pero "
            "pocas plantas por semana pueden importar para la fecha de siembra; con conteos semanales el aviso es más preciso.</span>"
            + _txt_hl + "</div>",
            unsafe_allow_html=True)
    elif r.get("recorrida_tardia"):
        rec = r["recorrida_tardia"]
        st.markdown(
            "<div style='background:#ede9fe;color:#3b0764;padding:0.75rem 1.1rem;border-radius:0.6rem;margin:-0.3rem 0 0.8rem 0;font-size:1.0rem'>"
            f"🌾 <b>Flujos tardíos · Recorrer cada {rec['cada_dias'] // 7} semanas</b> · {rec['texto']}"
            f"<br><span style='font-size:0.88rem;opacity:0.9'>Del {rec['desde']:%d/%m} al {rec['hasta']:%d/%m} no se muestra una probabilidad de flujo "
            "tardío: en 17 campañas de Bordenave no distinguió las semanas con emergencia de las demás.</span></div>",
            unsafe_allow_html=True)
    rr = al.riesgo_residual(pd.Timestamp(siembra), modelo) if siembra is not None else None
    if modelo.p.get("residual"):
        _ref = rr is None
        _r = rr if rr is not None else al.riesgo_residual(pd.Timestamp(anio, 6, 1), modelo)
        _ico = {"Muy alto": "🔴", "Alto": "🟠", "Medio": "🟡", "Bajo": "🟢"}
        _pc = lambda v: f"{100 * v:.1f} %".replace(".", ",")  # noqa: E731
        _rs, _po = _r["residual"], _r["post"]
        _v0, _v1 = _r["ventana"]
        _rec_res = {"Muy alto": "Usar residual", "Alto": "Residual recomendado", "Medio": "Según infestación",
                    "Bajo": "Solo si muy infestado"}[_rs["nivel"]]
        _rec_post = {"Muy alto": "Prever graminicida", "Alto": "Probable graminicida", "Medio": "Recorrer y decidir",
                     "Bajo": "Poco probable"}[_po["nivel"]]

        _dens = _r.get("por_densidad")
        _pl = lambda v: f"{v:.0f} pl/m²"  # noqa: E731

        def _dist(x):
            g = ((x["plm2_ge50"], x["plm2_ge20"], x["plm2_ge5"]) if _dens else (x["ge5"], x["ge2"], x["ge05"]) if "ge05" in x else None)
            if g is None:
                return ""
            return " · ".join(f"{n} {100 * v:.0f} %" for n, v in (("muy alto", g[0]), ("alto", g[1] - g[0]),
                                                                  ("medio", g[2] - g[1]), ("bajo", 1 - g[2])))

        def _loc(x):
            lo = x.get("locales_plm2") if _dens else x["locales"]
            fm = _pl if _dens else _pc
            if not lo:
                return ""
            if len(lo) <= 4:
                return " · ".join(f"{a} {fm(v)}" for a, v in sorted(lo.items()))
            vv = np.array(list(lo.values()))
            return f"mediana {fm(np.median(vv))}, entre {fm(vv.min())} y {fm(vv.max())} según la campaña"

        with st.container(border=True):
            _base = (f"datos de {_r['sitio']}, {_r['n']} campañas" if _r["fuente"] == "local"
                     else f"referencia regional, {_r['n']} campañas: {_r['sitio']} todavía tiene pocos datos")
            st.markdown(f"**🧪 Residual a la siembra · siembra {'de referencia ' if _ref else ''}{_r['siembra']:%d/%m}** ({_base}) · "
                        f"ventana de acción del residual: del {_r['siembra'] + pd.Timedelta(days=_v0):%d/%m} al "
                        f"{_r['siembra'] + pd.Timedelta(days=_v1):%d/%m} ({-_v0} días antes a {_v1} días después de sembrar)"
                        + (" — cargue la fecha de siembra prevista (arriba) para ver su caso" if _ref else ""))
            _hn = (("Según las plantas que nacieron en ese período en una campaña típica (mediana): muy alto ≥ 50 pl/m², alto 20–50, "
                    f"medio 5–20 y bajo < 5. Calculado con la densidad registrada en el sitio (referencia {_r['densidad_referencia']:.0f} pl/m² "
                    "por campaña); en un lote con menos raigrás el riesgo es menor. ") if _dens else
                   "Según la emergencia típica (mediana) de la campaña en ese período: muy alto ≥ 5 % de la campaña, alto 2–5 %, "
                   "medio 0,5–2 % y bajo < 0,5 %. ") + "Riesgo histórico: el modelo no anticipa cuánto nacerá en esta campaña."
            r1, r2, r3 = st.columns(3)
            r1.metric("Riesgo para el residual", f"{_ico[_rs['nivel']]} {_rs['nivel']}",
                      help=f"Nacimientos entre {-_v0} días antes y {_v1} días después de la siembra, período de acción promedio del residual. " + _hn)
            r2.metric(f"Nacimientos de −{-_v0} a +{_v1} días", f"{_pl(_rs['plm2'])} ({_pc(_rs['mediana'])})" if _dens else _pc(_rs["mediana"]),
                      help=(f"Mediana de las plantas/m² que nacieron en la ventana y, entre paréntesis, la fracción de la campaña; en 1 de cada 10 "
                            f"campañas, más de {_pl(_rs['plm2_p90'])}." if _dens else
                            f"Mediana de la fracción de la campaña que emergió en la ventana; en 1 de cada 10 campañas, más de {_pc(_rs['p90'])}."))
            r3.metric("Recomendación residual", _rec_res,
                      help="Muy alto: usar residual. Alto: residual recomendado. Medio: residual si el lote tiene alta infestación o "
                           "antecedentes de raigrás. Bajo: residual solo en lotes muy infestados. Producto y dosis: consultar al asesor y la etiqueta.")
            p1, p2, p3 = st.columns(3)
            p1.metric("Riesgo para el post-emergente", f"{_ico[_po['nivel']]} {_po['nivel']}",
                      help=f"Nacimientos más de {_v1} días después de la siembra, fuera del período de acción del residual. " + _hn)
            p2.metric(f"Nacimientos después de +{_v1} días", f"{_pl(_po['plm2'])} ({_pc(_po['mediana'])})" if _dens else _pc(_po["mediana"]),
                      help=(f"Mediana de las plantas/m² que nacieron después y, entre paréntesis, la fracción de la campaña; en 1 de cada 10 "
                            f"campañas, más de {_pl(_po['plm2_p90'])}." if _dens else
                            f"Mediana de la fracción de la campaña que emergió después; en 1 de cada 10 campañas, más de {_pc(_po['p90'])}."))
            p3.metric("Graminicida post-emergente", _rec_post,
                      help="Muy alto: prever un graminicida post-emergente. Alto: probable. Medio: recorrer el cultivo y decidir según "
                           "lo que se encuentre. Bajo: poco probable. En todos los casos, recorrer el cultivo 30–45 días después de la siembra.")
            _lr, _lp = _loc(_rs), _loc(_po)
            st.caption((f"{_r['sitio']}, residual: {_lr}. " if _lr else "") + (f"Post-emergente: {_lp}. " if _lp else "")
                       + (f"Campañas por nivel, residual: {_dist(_rs)}; post-emergente: {_dist(_po)}. " if _dist(_rs) else "")
                       + (f"Plantas/m² con la densidad registrada en {_r['sitio']} (referencia {_r['densidad_referencia']:.0f} pl/m² por campaña): "
                          "en un lote con menos raigrás el riesgo es menor." if _dens else
                          "En lotes con alta infestación, aun una fracción chica puede ser más de 100 plantas/m²."))

    clima = al.limpiar_clima(meteo)
    dg = al.tiempo_termico(clima, **tt)
    dgv, ult_tt = _tt_ventana(meteo, dg, emision, modelo, localidad)  # ventana de control: clima conocido + clima medio
    # la ventana de control se ancla solo en las primeras plántulas observadas: ninguna regla climática estimó el inicio con
    # precisión suficiente (error medio de 10–20 días, del orden del ancho de la ventana), así que no se usa un inicio estimado
    # (``inicio_predicho`` se acepta por compatibilidad y se ignora)
    ancla, ancla_txt = (r["primeras_plantulas"], "primeras plántulas observadas") if r["primeras_plantulas"] is not None else (None, None)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad de gran flujo (7 días)", f"{p:.0%}", help="Tasa histórica: ~25 % de las semanas de campaña."
              + (f" Conteo en cero el {r['cero_reciente']:%d/%m}: semana ya pasada sin plántulas: 0 % (por clima {r['probabilidad_clima']:.0%})."
                 if r.get("cero_reciente") is not None else ""))
    m2.metric("Emergencia esperada por lluvias", f"{100 * r['senales']['lluvia']:.1f} %",
              help="Respuesta relativa a las lluvias registradas hasta la fecha (señal, no % de la campaña).")
    if ancla is not None:
        tt_hoy = float(dg[(dg.index >= ancla) & (dg.index <= emision)].sum())
        f_ctl, f_lim = _fecha_tt(dgv, ancla, tt_control), _fecha_tt(dgv, ancla, tt_limite)
        _est = lambda d: d is not None and d > ult_tt  # noqa: E731
        m3.metric("Tiempo térmico desde el inicio", f"{tt_hoy:.0f} °Cd", help=f"Desde el {ancla:%d/%m} ({ancla_txt}).")
        m4.metric(f"Control post-emergente ({tt_control:.0f} °Cd)",
                  (f"{f_ctl:%d/%m}" + (" (estimada)" if _est(f_ctl) else "")) if f_ctl is not None else "—",
                  help=(f"Límite de la ventana ({tt_limite:.0f} °Cd): {f_lim:%d/%m}" + (" (estimada)" if _est(f_lim) else "") + "."
                        if f_lim is not None else "")
                       + (f" Las fechas estimadas suman el clima medio desde el {ult_tt + pd.Timedelta(days=1):%d/%m}; se corrigen "
                          "a medida que se carga el clima." if _est(f_lim) else " Alcanzada con el clima registrado.")
                       )
    else:
        m3.metric("Tiempo térmico desde el inicio", "—", help="Informe la fecha de las primeras plántulas para iniciar el reloj térmico.")
        m4.metric(f"Control post-emergente ({tt_control:.0f} °Cd)", "—",
                  help="Informe la fecha de las primeras plántulas para calcular la ventana de control. El inicio de la emergencia no se "
                       "estima con el clima: ninguna regla lo predijo con precisión suficiente (error medio de 10–20 días).")
    if r["primeras_plantulas"] is None and emision.month <= 6:
        st.markdown(_texto_muestreo_inicio(meteo, emision, conteos, modelo), unsafe_allow_html=True)
    elif r["primeras_plantulas"] is None:
        st.info("Sin fecha de primeras plántulas: la alerta usa solo el clima. Cuando aparezcan las primeras plántulas "
                "en el lote, cargue esa fecha arriba: ancla el reloj térmico de la ventana de control y mejora la alerta.")

    # gráfico único: alerta semanal (arriba) y emergencia acumulada (abajo), con la ventana de control en tiempo térmico
    s = al.serie(meteo, pd.Timestamp(anio, 2, 2), emision, primeras, conteos, modelo, paso=7, **tt, **opc)
    if s.empty:  # comienzo de campaña (antes del 2 de febrero): todavía no hay emisiones semanales
        s = pd.DataFrame({"Fecha": pd.DatetimeIndex([]), "probabilidad": pd.Series(dtype=float),
                          "probabilidad_clima": pd.Series(dtype=float), "nivel_info": pd.Series(dtype=int),
                          "nivel": pd.Series(dtype=str), "cero_reciente": pd.Series(dtype=bool),
                          "probabilidad_tardia": pd.Series(dtype=float)})
    # con fecha de primeras plántulas, las semanas que terminaron más de 7 días antes de esa fecha pasan a 0 (antes del
    # inicio); la semana anterior a las primeras plántulas conserva su probabilidad (ahí salieron las primeras plantas)
    s["antes_inicio"] = False
    if r["primeras_plantulas"] is not None and not s.empty:
        _p_ini = pd.Timestamp(r["primeras_plantulas"])
        _m = (s["Fecha"] + pd.Timedelta(days=13)) < _p_ini
        if _m.any():
            s.loc[_m, "probabilidad"] = 0.0
            s.loc[_m, "nivel"] = "baja"
            s.loc[_m & s["probabilidad_tardia"].notna(), "probabilidad_tardia"] = 0.0
            s["antes_inicio"] = _m
    # con fecha de primeras plántulas, la curva acumulada sigue el tiempo térmico desde esa fecha (opción por sitio)
    usa_tt = bool(curva_tiempo_termico and r["primeras_plantulas"] is not None and modelo.p.get("curva_tt"))
    if usa_tt:
        _prim_tt = pd.Timestamp(r["primeras_plantulas"])
        _dg_tt = al.tiempo_termico(al.limpiar_clima(meteo))
        _dg_tt = _dg_tt[_dg_tt.index <= ult_tt]  # sin días observados posteriores a la emisión (los completa el clima medio)

        def _curva(fechas, _d=_dg_tt, _p=_prim_tt):
            return modelo.curva_tiempo_termico(_d, _p, fechas, localidad)

        rango_txt = "10–90 %"
        titulo_cal = "Emergencia acumulada esperada (tiempo térmico desde las primeras plántulas, 17 campañas)"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.44, 0.13, 0.43], vertical_spacing=0.05,
                        subplot_titles=("Probabilidad de un gran flujo en los próximos 7 días", "Semáforo semanal", titulo_cal))
    fig.add_hrect(y0=100 * al.ATENCION, y1=100 * al.ALERTA, fillcolor=COLOR["atencion"], opacity=0.08, line_width=0, exclude_empty_subplots=False, row=1, col=1)
    fig.add_hrect(y0=100 * al.ALERTA, y1=100, fillcolor=COLOR["alerta"], opacity=0.06, line_width=0, exclude_empty_subplots=False, row=1, col=1)
    fig.add_hline(y=100 * al.ATENCION, line_dash="dot", line_color=COLOR["atencion"], annotation_text=f"Atención ≥ {al.ATENCION:.0%}",
                  annotation_position="bottom left", annotation_font_color=COLOR["atencion"], exclude_empty_subplots=False, row=1, col=1)
    fig.add_hline(y=100 * al.ALERTA, line_dash="dash", line_color=COLOR["alerta"], annotation_text=f"Alerta ≥ {al.ALERTA:.0%}",
                  annotation_position="top left", annotation_font_color=COLOR["alerta"], exclude_empty_subplots=False, row=1, col=1)
    x_fin = max(emision + pd.Timedelta(days=14), pd.Timestamp(anio, 6, 30))
    if ancla is not None:
        estimado = r["primeras_plantulas"] is None
        f_ctl, f_lim = _fecha_tt(dgv, ancla, tt_control), _fecha_tt(dgv, ancla, tt_limite)
        if f_ctl is not None:
            fin_ventana = f_lim if f_lim is not None else dgv.index.max()
            x_fin = max(x_fin, fin_ventana + pd.Timedelta(days=7))
            con_medio = fin_ventana > ult_tt
            # tramo alcanzado con el clima registrado (fuerte) y tramo estimado con el clima medio (claro, borde punteado)
            tramos = [(f_ctl, min(fin_ventana, ult_tt), 0.14, None)] if f_ctl <= ult_tt else []
            if con_medio:
                tramos.append((max(f_ctl, ult_tt), fin_ventana, 0.06, dict(color="#7c3aed", width=1, dash="dot")))
            for x0, x1, op, borde in tramos:
                for fila in (1, 2, 3):
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="#7c3aed", opacity=op, line=borde or dict(width=0),
                                  exclude_empty_subplots=False, row=fila, col=1)
            fig.add_annotation(x=f_ctl + (fin_ventana - f_ctl) / 2, y=97, xref="x", yref="y", showarrow=False, font=dict(size=12, color="#5b21b6"),
                               text=f"Ventana de control {tt_control:.0f}–{tt_limite:.0f} °Cd" + (" (inicio estimado)" if estimado else "")
                                    + f"<br>{f_ctl:%d/%m} – {fin_ventana:%d/%m}"
                                    + (f"<br><i>estimada con clima medio desde el {ult_tt + pd.Timedelta(days=1):%d/%m}</i>" if con_medio else ""))
        for fila in (1, 2, 3):
            fig.add_vline(x=ancla, line_dash="dot", line_color="#374151", exclude_empty_subplots=False, row=fila, col=1)
        fig.add_annotation(x=ancla, y=4, xref="x", yref="y", showarrow=False, xanchor="left", font=dict(size=11, color="#374151"),
                           text=f" Inicio {ancla:%d/%m} ({'estimado' if estimado else 'primeras plántulas'})")
    for fila in (1, 2, 3):
        fig.add_vline(x=emision, line_dash="dash", line_color="#9ca3af", exclude_empty_subplots=False, row=fila, col=1)
    fig.add_annotation(x=emision, y=4, xref="x", yref="y", showarrow=False, xanchor="left", font=dict(size=11, color="#6b7280"),
                       text=f" emisión {emision:%d/%m}")
    # semanas ya cerradas por un conteo posterior en cero (sin plántulas): probabilidad 0
    sin_inicio = s["cero_reciente"].astype(bool) if "cero_reciente" in s else pd.Series(False, index=s.index)
    y_plot = 100 * s["probabilidad"]
    etiqueta = [f"antes del inicio (primeras plántulas el {pd.Timestamp(r['primeras_plantulas']):%d/%m}; por clima {100 * pc:.0f} %)" if a
                else f"semana sin plántulas según el conteo en cero (por clima {100 * pc:.0f} %)" if z else INFO[n]
                for a, z, pc, n in zip(s["antes_inicio"], sin_inicio, s.get("probabilidad_clima", s["probabilidad"]), s["nivel_info"])]
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
                                 name="Probabilidad de gran flujo de cada semana"), row=1, col=1)
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
    # semáforo semanal: una fila para el flush de la campaña y otra para la emergencia tardía (1/05–1/10); una
    # celda por semana (de la emisión a 7 días después) y, con borde punteado, la de los próximos 7 días
    v0, v1 = al.limites_ventana_siembra(anio, modelo)
    _rec = (modelo.p.get("tardia") or {}).get("recorrida")
    v1b = pd.Timestamp(anio, *_rec["hasta"]) if _rec else v1
    fig.add_shape(type="rect", x0=v0, x1=v1b + pd.Timedelta(days=1), y0=0.02, y1=0.98, fillcolor="#f5f3ff", line_width=0, layer="below", row=2, col=1)
    fig.add_annotation(x=v0, y=0.5, xref="x2", yref="y2", showarrow=False, xanchor="right", font=dict(size=10, color="#6d28d9"),
                       text="flujos tardíos →" if emision < v0 else "")
    hx, hy, ht = [], [], []

    def _celda(x0, x1, fila, color, texto, futura=False):
        y0, y1 = (1.08, 1.92) if fila == "flush" else (0.08, 0.92)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, opacity=0.55 if futura else 1.0,
                      line=dict(color="#111827" if futura else "white", width=2 if futura else 1, dash="dot" if futura else "solid"),
                      row=2, col=1)
        hx.append(x0 + (x1 - x0) / 2); hy.append((y0 + y1) / 2); ht.append(texto)

    tardia_col = lambda v, u: "#6d28d9" if v >= u else "#ddd6fe"  # noqa: E731
    tardia_nom = lambda v, u: "flujo tardío probable" if v >= u else "flujo tardío poco probable"  # noqa: E731
    _ni = s["nivel_info"] if "nivel_info" in s else pd.Series(1, index=s.index)
    for f0, f1, v, n, vt, ok, ni in zip(ini_sem, fin_sem, s["probabilidad"], s["nivel"], s["probabilidad_tardia"], pasadas, _ni):
        if not ok:
            continue
        sem = f"{f0:%d/%m}–{f1 - pd.Timedelta(days=1):%d/%m}"
        _celda(f0, f1, "flush", COLOR[n], f"Semana {sem} · gran flujo: {NOMBRE[n]} ({100 * v:.0f} %)")
        if not pd.isna(vt):
            ut = al.umbral_tardia(f0, ni, modelo)
            _celda(f0, f1, "tardia", tardia_col(vt, ut), f"Semana {sem} · {tardia_nom(vt, ut)} ({100 * vt:.0f} %; aviso desde {_pct(ut)})")
        elif (_rc := al.recorrida_tardia(f0, modelo)):
            _celda(f0, f1, "tardia", "#e5e7eb", f"Semana {sem} · flujos tardíos: recorrer cada {_rc['cada_dias'] // 7} semanas (sin probabilidad en invierno)")
    sem = f"{emision:%d/%m}–{fin_prox - pd.Timedelta(days=1):%d/%m}"
    _celda(emision, fin_prox, "flush", COLOR[nivel], f"Próximos 7 días ({sem}) · gran flujo: {NOMBRE[nivel]} ({100 * p:.0f} %)", futura=True)
    if r.get("probabilidad_tardia") is not None:
        vt = r["probabilidad_tardia"]
        ut = r.get("umbral_tardia", al.UMBRAL_TARDIA)
        _celda(emision, fin_prox, "tardia", tardia_col(vt, ut), f"Próximos 7 días ({sem}) · {tardia_nom(vt, ut)} ({100 * vt:.0f} %; aviso desde {_pct(ut)})", futura=True)
    elif r.get("recorrida_tardia"):
        _celda(emision, fin_prox, "tardia", "#e5e7eb", f"Próximos 7 días ({sem}) · flujos tardíos: recorrer cada {r['recorrida_tardia']['cada_dias'] // 7} semanas", futura=True)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=16, opacity=0), text=ht, showlegend=False,
                             hovertemplate="%{text}<extra></extra>"), row=2, col=1)
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
    # con fecha de primeras plántulas y calendario (sin curva en tiempo térmico): la curva vale 0 antes de esa fecha, arranca
    # en la fracción típica ya emergida al ver las primeras plantas (~18 %) y desde ahí sigue el calendario reescalado
    usa_prim = bool(curva_desde_primeras and not usa_tt and r["primeras_plantulas"] is not None)
    if usa_prim:
        _base_p = curva_de
        _p0 = pd.Timestamp(r["primeras_plantulas"])
        _f0 = float(modelo.p.get("curva_tt", {}).get("media", [0.18])[0])

        def curva_de(fechas, _b=_base_p, _t=_p0, _f=_f0):
            out = _b(pd.DatetimeIndex(fechas)).copy()
            ref = _b(pd.DatetimeIndex([_t])).iloc[0]
            for k in ("media", "p10", "p90"):
                rel = np.clip((out[k] - ref[k]) / max(1 - ref[k], 1e-6), 0, 1)
                out[k] = np.where(pd.DatetimeIndex(out["Fecha"]) < _t, 0.0, _f + (1 - _f) * rel)
            return out

        cv = curva_de(pd.date_range(pd.Timestamp(anio, 1, 15), x_fin))
    if curva_desde_ultimo_cero and not usa_tt and not usa_prim and conteos is not None and not conteos.empty:
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
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(21,128,61,0.16)", name=f"Rango {rango_txt} de las campañas", hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=cv["Fecha"], y=100 * cv["media"], mode="lines", name="Emergencia acumulada esperada",
                             line=dict(color="#15803d", width=2.5),
                             customdata=np.c_[100 * cv["p10"], 100 * cv["p90"]],
                             hovertemplate="%{x|%d/%m}: %{y:.0f} % (rango %{customdata[0]:.0f}–%{customdata[1]:.0f} %)<extra></extra>"), row=3, col=1)
    hoy = curva_de(pd.DatetimeIndex([emision])).iloc[0]
    fig.add_trace(go.Scatter(x=[emision], y=[100 * hoy["media"]], mode="markers+text", marker=dict(size=10, color="#15803d", line=dict(color="white", width=1.5)),
                             text=[f"{100 * hoy['media']:.0f} % ({100 * hoy['p10']:.0f}–{100 * hoy['p90']:.0f} %)"], textposition="middle left",
                             textfont=dict(size=11, color="#14532d"), hoverinfo="skip", showlegend=False), row=3, col=1)
    fig.update_yaxes(title_text="Prob. de gran flujo (%)", range=[0, 106], row=1, col=1)
    fig.update_yaxes(range=[0, 2], tickvals=[0.5, 1.5], ticktext=["Flujos tardíos<br>(≥ 2 %)", "Grandes flujos<br>(≥ 5 %)"], showgrid=False,
                     zeroline=False, fixedrange=True, row=2, col=1)
    fig.update_yaxes(title_text="% de la campaña", range=[0, 102], row=3, col=1)
    fig.update_xaxes(range=[pd.Timestamp(anio, 2, 1), x_fin], tickformat="%d/%m")
    fig.update_layout(height=780, margin=dict(t=40, b=10, l=10, r=10), showlegend=False, hovermode="closest",
                      title=dict(text="Alerta de grandes flujos, emergencia acumulada y ventana de control", font=dict(size=15)))
    fig.update_annotations(selector=dict(text="Probabilidad de un gran flujo en los próximos 7 días"), font_size=12)
    fig.update_annotations(selector=dict(text=titulo_cal), font_size=12)
    fig.update_annotations(selector=dict(text="Semáforo semanal"), font_size=12)
    st.plotly_chart(fig, width="stretch")
    st.caption("Arriba, cada tramo horizontal es la probabilidad de un gran flujo (≥ 5 % de la campaña) en esa semana, emitida al comienzo de la semana con la "
               "información disponible ese día (color del punto = nivel); el tramo punteado, a la derecha de la fecha de emisión, es "
               "el pronóstico para los próximos 7 días. Un conteo en cero confirma las semanas que terminaron antes de esa fecha: "
               "esos tramos quedan en 0 %, igual que las semanas que terminaron más de 7 días antes de la fecha de primeras plántulas. En el medio, el semáforo semanal: la fila «Grandes flujos» repite el nivel de cada semana "
               f"(verde, naranja, rojo) y la fila «Flujos tardíos», del {v0:%d/%m} al {v1:%d/%m}, marca en violeta fuerte las semanas con "
               "flujo tardío probable (≥ 2 % de la campaña)"
               + (f"; del {v1 + pd.Timedelta(days=1):%d/%m} al {v1b:%d/%m}, en gris, rige la recomendación de recorrer cada 2 semanas" if _rec else "")
               + "; las celdas con borde punteado son los próximos 7 días. Abajo, la "
               + (f"emergencia acumulada esperada según el tiempo térmico acumulado desde las primeras plántulas ({_prim_tt:%d/%m}): "
                  "curva media y rango 10–90 % de 17 campañas de Bordenave; más allá del clima disponible se usan temperaturas "
                  "medias históricas. Sin fecha de primeras plántulas se usa el calendario. "
                  if usa_tt else "emergencia acumulada esperada para Bordenave: curva media y rango 10–90 % de 17 campañas (error típico ±13 puntos). "
                  if calibrada_aqui else (f"emergencia acumulada esperada según el calendario propio de {localidad} (curva media y "
                  f"rango {rango_txt} de las campañas {_anios}"
                  + ("). " if calendario_en_alerta else "; la probabilidad de gran flujo usa el calendario de Bordenave). ")
                  + ("Desde las primeras plántulas, la probabilidad incluye el agotamiento de la cohorte. " if agotamiento else "")
                  if calendario_local else "emergencia acumulada esperada según el calendario de Bordenave (curva media y rango 10–90 % "
                  "de 17 campañas), usado como referencia regional. "))
               + (f"La curva arranca el día de las primeras plántulas ({_p0:%d/%m}) en la fracción típica ya emergida al verlas. "
                  if usa_prim else "")
               + "La franja violeta es la ventana de control post-emergente en tiempo térmico desde el inicio; el tramo claro con borde "
               "punteado todavía no se alcanzó con el clima registrado y se estima con el clima medio.")

    with st.expander("Cómo se validó esta alerta"):
        if nota_sitio:
            st.markdown(nota_sitio)
        v = modelo.p.get("validacion", {})
        filas = [(k, d) for k, d in v.items()]
        if filas:
            st.markdown("| Información disponible | AUC | Grandes flujos detectados con Alerta (≥ 25 %) | Semanas marcadas |\n|---|---|---|---|\n"
                        + "\n".join(f"| {k} | {d['auc']:.2f} | {d['det_at']:.0%} | {d['vis_at']:.0%} |" for k, d in filas))
        st.markdown(
            f"{modelo.p.get('fuente', '')}\n\nCada campaña se predijo con el modelo ajustado sin ella. Las probabilidades están "
            "calibradas (cuando dice ~25 %, el gran flujo ocurrió en ~27 % de las semanas). Como referencia, con las mismas campañas "
            "el motor PREDWEEM de arriba da AUC ~0,70 para anticipar grandes flujos, y su primer pico erró el inicio de la emergencia en "
            "17–20 días en promedio (31 campañas): por eso el reloj térmico y la ventana de control se anclan solo en las primeras "
            "plántulas observadas.\n\n"
            "**Límites.** Un sitio y lotes sin labranza. La fecha de primeras plántulas mejora la alerta (AUC +0,07 sobre el clima "
            "solo) y los conteos semanales algo más (+0,10). La señal de lluvia usa el modelo de pulsos ajustado a estas campañas "
            "de Lolium; frente al calendario aporta poco por sí sola, más cuando se combina con la fecha de primeras plántulas.")
        tv = modelo.p.get("tardia")
        if tv:
            st.markdown(
                "**Flujos tardíos (1/05–30/06).** Probabilidad de que en los próximos 7 días emerja al menos "
                f"2 % de la campaña; aviso desde {al.UMBRAL_TARDIA:.0%} (25 % sin datos del lote). Es la misma alerta recalibrada para esa época, en la que los "
                "grandes flujos (≥ 5 %) son raros pero pocas plantas por semana importan para decidir la fecha de siembra. "
                "Del 1/07 al 1/10 no se da probabilidad: en Bordenave no distinguió las semanas con flujos tardíos (AUC ≤ 0,5), "
                "así que se recomienda recorrer los lotes cada 2 semanas.\n\n"
                "| Información | AUC Bordenave | Detectadas | Semanas marcadas | AUC otras localidades | Detectadas | Marcadas |\n|---|---|---|---|---|---|---|\n"
                + "\n".join(f"| {k} | {d['auc']:.2f} | {d['det']:.0%} | {d['marcadas']:.0%} | "
                             + (f"{o['auc']:.2f} | {o['det']:.0%} | {o['marcadas']:.0%} |" if (o := tv['validacion_otras'].get(k if k != '+ conteos semanales' else '+ conteos')) else "— | — | — |")
                             for k, d in tv["validacion_bordenave"].items())
                + f"\n\n{tv['fuente']} En las otras localidades la probabilidad tiende a ser alta (promedio 30 % contra 17 % "
                "observado): el aviso detecta casi todas las semanas con flujos tardíos, con más falsas alarmas.")

    # ---------------------------------------------------------------
    # resultados de la sección, para la descarga final
    # ---------------------------------------------------------------
    f_ctl_r, f_lim_r = (_fecha_tt(dgv, ancla, tt_control), _fecha_tt(dgv, ancla, tt_limite)) if ancla is not None else (None, None)
    tt_r = float(dg[(dg.index >= ancla) & (dg.index <= emision)].sum()) if ancla is not None else None
    fmt_f = lambda d: pd.Timestamp(d).strftime("%d/%m/%Y") if d is not None and not pd.isna(d) else ""  # noqa: E731
    resumen = pd.DataFrame({"Indicador": [
        "Localidad", "Fecha de generación", "Fecha de emisión", "Nivel de alerta", "Probabilidad de gran flujo (≥ 5 %) en 7 días (%)",
        "Información usada", "Inicio del reloj térmico", "Origen del inicio", "Tiempo térmico desde el inicio (°Cd)",
        f"Fecha de control post-emergente ({tt_control:.0f} °Cd)", f"Fecha límite de la ventana ({tt_limite:.0f} °Cd)",
        "Emergencia acumulada esperada a la fecha (%)", f"Rango {rango_txt} de la emergencia acumulada (%)",
        "Señal de calendario (emergencia esperada en 7 días, %)", "Señal de lluvia (respuesta relativa, %)",
        "Conteos usados (fechas)", "Primeras plántulas informadas"],
        "Valor": [
        f"{localidad} (Buenos Aires)", pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"), fmt_f(emision), NOMBRE[nivel], round(100 * p, 1),
        INFO[r["nivel_info"]], fmt_f(ancla), ancla_txt or "sin primeras plántulas: la ventana de control no se calcula", round(tt_r, 1) if tt_r is not None else "",
        fmt_f(f_ctl_r) + (" (estimada con clima medio)" if f_ctl_r is not None and f_ctl_r > ult_tt else ""),
        fmt_f(f_lim_r) + (" (estimada con clima medio)" if f_lim_r is not None and f_lim_r > ult_tt else ""),
        round(100 * hoy["media"], 1), f"{100 * hoy['p10']:.0f}–{100 * hoy['p90']:.0f}",
        round(100 * r["senales"]["cal"], 2), round(100 * r["senales"]["lluvia"], 2),
        int((conteos["Fecha"] <= emision).sum()) if conteos is not None else 0, fmt_f(r["primeras_plantulas"])]})
    resumen = pd.concat([resumen, pd.DataFrame({"Indicador": ["Probabilidad de flujo tardío (≥ 2 % de la campaña en 7 días, %)",
                                                              "Aviso de flujo tardío (1/05–30/06; del 1/07 al 1/10, recorrida cada 2 semanas)"],
                                                "Valor": [round(100 * pt, 1) if pt is not None else ("sin probabilidad: recorrer los lotes cada 2 semanas (1/07–1/10)"
                                                          if r.get("recorrida_tardia") else "fuera del período de flujos tardíos (1/05–1/10)"),
                                                          ("probable" if pt >= r.get("umbral_tardia", al.UMBRAL_TARDIA) else "poco probable") if pt is not None else ""]})],
                        ignore_index=True)
    if rr is not None:
        resumen = pd.concat([resumen, pd.DataFrame({"Indicador": [
            "Fecha de siembra prevista",
            "Riesgo para el residual: nacimientos de 15 días antes a 45 días después de la siembra (muy alto ≥ 50 pl/m², alto ≥ 20, medio ≥ 5, bajo < 5)",
            "Nacimientos en la ventana del residual: mediana y percentil 90 (pl/m²; % de la campaña)", "Residual a la siembra",
            "Riesgo para el post-emergente: nacimientos más de 45 días después de la siembra",
            "Nacimientos después de 45 días de la siembra: mediana y percentil 90 (pl/m²; % de la campaña)", "Graminicida post-emergente"],
            "Valor": [fmt_f(rr["siembra"]), rr["residual"]["nivel"],
                      (f"{rr['residual'].get('plm2', float('nan')):.0f} y {rr['residual'].get('plm2_p90', float('nan')):.0f} pl/m²; "
                       f"{100 * rr['residual']['mediana']:.1f} y {100 * rr['residual']['p90']:.1f} %"),
                      {"Muy alto": "usar residual", "Alto": "residual recomendado", "Medio": "según infestación (lotes con alta infestación o antecedentes de raigrás)",
                       "Bajo": "solo en lotes muy infestados"}[rr["residual"]["nivel"]] + " · riesgo histórico, no predicción de la campaña",
                      rr["post"]["nivel"], (f"{rr['post'].get('plm2', float('nan')):.0f} y {rr['post'].get('plm2_p90', float('nan')):.0f} pl/m²; "
                                            f"{100 * rr['post']['mediana']:.1f} y {100 * rr['post']['p90']:.1f} %"),
                      {"Muy alto": "prever graminicida post-emergente", "Alto": "graminicida post-emergente probable",
                       "Medio": "recorrer el cultivo y decidir", "Bajo": "poco probable"}[rr["post"]["nivel"]]
                      + " · recorrer el cultivo 30–45 días después de la siembra"]})],
            ignore_index=True)
    if agotamiento:
        _ag = r["senales"].get("agotamiento")
        resumen = pd.concat([resumen, pd.DataFrame({"Indicador": ["Agotamiento de la cohorte (factor sobre la probabilidad)"],
                                                    "Valor": [round(_ag, 3) if _ag is not None else "sin primeras plántulas: no se aplica"]})],
                            ignore_index=True)
    serie_out = s.assign(**{"Probabilidad de gran flujo (%)": (100 * s["probabilidad"]).round(1),
                            "Semana sin plántulas (conteo en cero posterior)": s["cero_reciente"] if "cero_reciente" in s else False,
                            "Antes del inicio informado": s["antes_inicio"],
                            "Probabilidad por clima (%)": (100 * s.get("probabilidad_clima", s["probabilidad"])).round(1),
                            "Nivel": [NOMBRE[v] for v in s["nivel"]],
                            "Información usada": [INFO[v] for v in s["nivel_info"]],
                            "Flujo tardío ≥ 2 % en 7 días (%)": (100 * s["probabilidad_tardia"]).round(1) if "probabilidad_tardia" in s else np.nan})[["Fecha", "Probabilidad de gran flujo (%)", "Nivel", "Información usada", "Semana sin plántulas (conteo en cero posterior)", "Antes del inicio informado", "Probabilidad por clima (%)", "Flujo tardío ≥ 2 % en 7 días (%)"]]
    _lo, _hi = ("P10 (%)", "P90 (%)") if not calendario_local else ("Mínimo (%)", "Máximo (%)")
    curva_out = cv.assign(**{"Emergencia acumulada media (%)": (100 * cv["media"]).round(2), _lo: (100 * cv["p10"]).round(2),
                             _hi: (100 * cv["p90"]).round(2)})[["Fecha", "Emergencia acumulada media (%)", _lo, _hi]]
    if ancla is not None:
        dd = dgv[(dgv.index >= ancla) & (dgv.index <= max(f_lim_r if f_lim_r is not None else ult_tt, ult_tt))]
        tt_out = pd.DataFrame({"Fecha": dd.index, "TT diario (°Cd)": dd.values.round(2), "TT acumulado desde el inicio (°Cd)": dd.cumsum().values.round(1),
                               "Origen": np.where(dd.index <= min(emision, ult_tt), "clima registrado",
                                                  np.where(dd.index <= ult_tt, "pronóstico", "clima medio"))})
        tt_out["Etapa"] = np.where(tt_out["TT acumulado desde el inicio (°Cd)"] < tt_control, "antes de la ventana",
                                   np.where(tt_out["TT acumulado desde el inicio (°Cd)"] <= tt_limite, "ventana de control", "después de la ventana"))
    else:
        tt_out = pd.DataFrame()
    clima_out = clima[clima["Fecha"] <= emision + pd.Timedelta(days=7)].rename(columns={"Prec": "Lluvia (mm)"})
    params = pd.DataFrame({"Parámetro": [
        "Umbral de Atención (%)", "Umbral de Alerta (%)", "Definición de gran flujo", "Horizonte de la alerta (días)",
        "Aviso de flujo tardío (%)", "Definición de flujo tardío",
        "Temperatura base (°C)", "Temperatura óptima (°C)", "Temperatura crítica (°C)", "TT de control (°Cd)", "TT límite (°Cd)",
        "Calibración"], "Valor": [
        100 * al.ATENCION, 100 * al.ALERTA, "≥ 5 % de la emergencia de la campaña en 7 días", 7,
        f"{100 * al.UMBRAL_TARDIA:g} (10 del 1/07 al 1/10 con primeras plántulas o conteos)", "≥ 2 % de la emergencia de la campaña en 7 días, del 1/05 al 1/10",
        t_base, t_opt, t_crit, tt_control, tt_limite, modelo.p.get("fuente", "")]})
    val = pd.DataFrame([{"Información disponible": k, "AUC": d["auc"], "Grandes flujos detectados con Alerta (%)": round(100 * d["det_at"], 1),
                         "Semanas marcadas con Alerta (%)": round(100 * d["vis_at"], 1)} for k, d in modelo.p.get("validacion", {}).items()])
    return {"Resumen": resumen, "Alerta_semanal": serie_out, "Emergencia_acumulada": curva_out, "Tiempo_termico": tt_out,
            "Conteos": conteos.rename(columns={"PLM2": "Plántulas"}) if conteos is not None else pd.DataFrame(),
            "Clima": clima_out, "Parametros": params, "Validacion": val}
