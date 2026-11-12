# -*- coding: utf-8 -*-
"""Alerta validada de flushes de Lolium multiflorum (Bordenave) por niveles de información.

Señales, todas disponibles a la fecha de emisión:
  * calendario: emergencia esperada en los próximos 7 días según la curva media de 17 campañas de LOLMU;
  * lluvia: respuesta a las lluvias registradas hasta la fecha (rezago gamma en tiempo térmico de suelo, mayor
    respuesta sobre suelo seco y tamaño del pulso según la época del año), con el núcleo del modelo de pulsos ajustado
    con las 17 campañas de Lolium multiflorum de Bordenave;
  * inicio: tiempo térmico desde la fecha en que se vieron las primeras plántulas (si se informó);
  * conteos: fracción de lo contado que emergió en el último intervalo (si hay conteos).

Niveles: 1 = sin datos del lote; 2 = con la fecha de primeras plántulas; 3 = con conteos.
Cada señal se expresa como percentil de su distribución en las campañas de calibración y una regresión
logística da la probabilidad de flush (≥ 5 % de la emergencia de la campaña en los próximos 7 días).
Los coeficientes, percentiles y la curva de calendario están en ``alerta_lolmu_params.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammainc

PARAMS_FILE = Path(__file__).with_name("alerta_lolmu_params.json")
ATENCION, ALERTA = 0.15, 0.25
UMBRAL_TARDIA = 0.20  # aviso de emergencia tardía (≥ 2 % de la campaña en 7 días), del 1/05 al 30/06
HORIZONTE_DIAS = 7  # semana de la alerta
TOPE_CERO = 0.14  # con un conteo reciente en cero y sin plántulas, la probabilidad queda debajo de Atención

# balance de agua y temperatura de suelo (5 cm) del modelo de pulsos de Bordenave (sin reajustar)
SUELO = dict(ema_alpha=0.426, ts_a0=3.39, ts_a1=-0.66, W=75.6, k_loss=0.0327, Tb=5.0)
# núcleo de respuesta a la lluvia: modelo de pulsos ajustado por mínimos cuadrados no lineales a los incrementos semanales
# de emergencia de las 17 campañas de Lolium multiflorum de Bordenave. Tamaño del pulso:
# exp(a0 + a1·sen(2π·día/365) + a2·cos(2π·día/365) + delta·(1 − S⁻/W)), con el día del año de la lluvia (máximo a fines
# de mayo); rezago gamma en tiempo térmico de suelo con forma k y escala ``escala`` (mediana 114 °Cd, p90 252 °Cd).
NUCLEO = dict(a0=-5.1047, a1=1.8626, a2=-2.1532, delta=0.5124, k=float(np.exp(0.8052)), escala=float(np.exp(4.0843)))


def limpiar_clima(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}
    out = df.rename(columns={cols["fecha"]: "Fecha", cols["tmax"]: "TMAX", cols["tmin"]: "TMIN",
                             cols["prec"]: "Prec"})[["Fecha", "TMAX", "TMIN", "Prec"]].copy()
    out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates("Fecha", keep="last")
    for c in ("TMAX", "TMIN", "Prec"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.set_index("Fecha").asfreq("D")
    out[["TMAX", "TMIN"]] = out[["TMAX", "TMIN"]].interpolate(limit_direction="both")
    out["Prec"] = out["Prec"].fillna(0.0).clip(lower=0.0)
    swap = out["TMIN"] > out["TMAX"]
    out.loc[swap, ["TMAX", "TMIN"]] = out.loc[swap, ["TMIN", "TMAX"]].values
    return out.reset_index()


def modelo_suelo(clima: pd.DataFrame) -> pd.DataFrame:
    """Temperatura de suelo, agua del balde antes de cada día y tiempo térmico acumulado de suelo."""
    p = SUELO
    tmax, tmin, rain = (clima[c].to_numpy(float) for c in ("TMAX", "TMIN", "Prec"))
    doy = clima["Fecha"].dt.dayofyear.to_numpy()
    tmean = (tmax + tmin) / 2
    dtr = np.clip(tmax - tmin, 0, None)
    n = len(clima)
    ema, s_prev = np.empty(n), np.empty(n)
    e, s = tmean[0], p["W"] / 2
    for i in range(n):
        e = p["ema_alpha"] * tmean[i] + (1 - p["ema_alpha"]) * e if i else tmean[0]
        ema[i] = e
        s_prev[i] = s
        s = min(s + rain[i], p["W"])
        s = max(s - p["k_loss"] * max(tmean[i] + 5, 0) * np.sqrt(dtr[i]), 0.0)
    ts = ema + p["ts_a0"] + p["ts_a1"] * np.cos(2 * np.pi * (doy - 196) / 365.0)
    out = clima.copy()
    out["S_prev"] = s_prev
    out["C_TT"] = np.cumsum(np.clip(ts - p["Tb"], 0, None))
    return out


def senal_lluvia(suelo: pd.DataFrame, fecha: pd.Timestamp, horizonte: int = 7) -> float:
    """Emergencia relativa esperada en los próximos días por las lluvias registradas hasta ``fecha``."""
    fechas = suelo["Fecha"]
    pos = np.flatnonzero(fechas.to_numpy() <= np.datetime64(pd.Timestamp(fecha)))
    if len(pos) == 0:
        return 0.0
    s = int(pos[-1])
    e = min(s + horizonte, len(suelo) - 1)
    if e <= s:
        return 0.0
    C, P, Sp = (suelo[c].to_numpy(float) for c in ("C_TT", "Prec", "S_prev"))
    wet = np.flatnonzero(P > 0.25)
    us = wet[(wet >= s - 60) & (wet <= s)]
    if len(us) == 0:
        return 0.0
    cap = us + 60
    t_s = np.where(s >= us, np.clip(C[np.minimum(s, cap)] - C[us], 0, None), 0.0)
    t_e = np.clip(C[np.minimum(e, cap)] - C[us], 0, None)
    ok = t_e > t_s
    us, t_s, t_e = us[ok], t_s[ok], t_e[ok]
    W = SUELO["W"]
    x = np.log1p(np.clip(np.minimum(P[us], W - Sp[us]), 0, None))
    ang = 2 * np.pi * fechas.dt.dayofyear.to_numpy()[us] / 365
    A = np.exp(NUCLEO["a0"] + NUCLEO.get("a1", 0.0) * np.sin(ang) + NUCLEO.get("a2", 0.0) * np.cos(ang)
               + NUCLEO["delta"] * (1 - Sp[us] / W))
    G = gammainc(NUCLEO["k"], t_e / NUCLEO["escala"]) - gammainc(NUCLEO["k"], t_s / NUCLEO["escala"])
    return float(1 - np.exp(-np.sum(A * x * G)))


def tiempo_termico(clima: pd.DataFrame, t_base=2.0, t_opt=20.0, t_crit=30.0) -> pd.Series:
    """Tiempo térmico diario triangular sobre la temperatura media del aire (mismo esquema que la app)."""
    t = (clima["TMAX"] + clima["TMIN"]) / 2
    dg = np.where(t <= t_base, 0.0, np.where(t <= t_opt, t - t_base,
                  np.where(t < t_crit, (t - t_base) * (t_crit - t) / (t_crit - t_opt), 0.0)))
    return pd.Series(dg, index=pd.DatetimeIndex(clima["Fecha"]))


def senal_conteos(conteos: pd.DataFrame | None, fecha: pd.Timestamp) -> tuple[float, int, int]:
    """(fracción del último intervalo sobre lo acumulado, primer conteo con plántulas, nº de conteos)."""
    if conteos is None or conteos.empty:
        return 0.0, 0, 0
    c = conteos[conteos["Fecha"] <= fecha]
    if c.empty:
        return 0.0, 0, 0
    acum = float(c["PLM2"].sum())
    g1 = float(c["PLM2"].iloc[-1]) / acum if acum > 0 else 0.0
    n_pos = int((c["PLM2"] > 0).sum())
    return g1, int(n_pos == 1 and g1 >= 0.999), len(c)


def ventana_siembra(fecha: pd.Timestamp, modelo: "AlertaLolmu | None" = None) -> bool:
    """True si ``fecha`` cae en el período del aviso de emergencia tardía (por defecto, 1/05–30/06 inclusive)."""
    t = (modelo or AlertaLolmu()).p.get("tardia") or {}
    (m0, d0), (m1, d1) = t.get("desde", [5, 1]), t.get("hasta", [6, 30])
    f = pd.Timestamp(fecha)
    return (m0, d0) <= (f.month, f.day) <= (m1, d1)


def limites_ventana_siembra(anio: int, modelo: "AlertaLolmu | None" = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    t = (modelo or AlertaLolmu()).p.get("tardia") or {}
    (m0, d0), (m1, d1) = t.get("desde", [5, 1]), t.get("hasta", [6, 30])
    return pd.Timestamp(anio, m0, d0), pd.Timestamp(anio, m1, d1)


def riesgo_residual(siembra: pd.Timestamp, modelo: "AlertaLolmu | None" = None) -> dict | None:
    """Riesgo histórico de emergencia posterior a la siembra (``residual`` en los parámetros), para siembras del 1/05 al 31/07.

    Devuelve la proporción de campañas en las que emergió ≥ 10 % del total después de la siembra, la mediana y el percentil 90
    de esa fracción, y la fracción que emergió más de 45 días después de sembrar. Es un riesgo histórico de 32 campañas, no una
    predicción de la campaña en curso: ni la fecha de primeras plántulas ni el clima anticiparon cuánta emergencia queda
    después de la siembra (AUC 0,4–0,7).
    """
    t = (modelo or AlertaLolmu()).p.get("residual")
    if not t or siembra is None:
        return None
    d = pd.Timestamp(siembra).dayofyear
    if not t["dia"][0] <= d <= t["dia"][-1]:
        return None
    f = lambda k: float(np.interp(d, t["dia"], t[k]))  # noqa: E731
    return {"siembra": pd.Timestamp(siembra), "n": int(t["n"]), "frac_ge10": f("frac_ge10"), "mediana": f("mediana"), "p90": f("p90"),
            "post45_mediana": f("post45_mediana"), "post45_p90": f("post45_p90")}


def recorrida_tardia(fecha: pd.Timestamp, modelo: "AlertaLolmu | None" = None) -> dict | None:
    """Recomendación fija de recorrida para el invierno (``tardia.recorrida``; por defecto, 1/07–1/10), o None.

    En esos meses la probabilidad de flujo tardío no distinguió las semanas con emergencia (AUC ≤ 0,5 en Bordenave),
    así que en lugar de un aviso por probabilidad se recomienda recorrer los lotes cada ``cada_dias`` días.
    """
    t = ((modelo or AlertaLolmu()).p.get("tardia") or {}).get("recorrida")
    if not t:
        return None
    f = pd.Timestamp(fecha)
    if tuple(t["desde"]) <= (f.month, f.day) <= tuple(t["hasta"]):
        return {"cada_dias": int(t.get("cada_dias", 14)), "texto": t.get("texto", ""),
                "desde": pd.Timestamp(f.year, *t["desde"]), "hasta": pd.Timestamp(f.year, *t["hasta"])}
    return None


def umbral_tardia(fecha: pd.Timestamp, nivel_info: int = 1, modelo: "AlertaLolmu | None" = None) -> float:
    """Umbral del aviso de flujo tardío para una fecha y nivel de información.

    Por defecto ``tardia.umbral`` (20 %); los tramos de ``tardia.umbral_tramos`` ([mes, día] inclusive y niveles de
    información) lo reemplazan: sin datos del lote (nivel 1), 25 %.
    """
    t = (modelo or AlertaLolmu()).p.get("tardia") or {}
    f = pd.Timestamp(fecha)
    for tr in t.get("umbral_tramos", []):
        if tuple(tr["desde"]) <= (f.month, f.day) <= tuple(tr["hasta"]) and int(nivel_info) in tr.get("niveles", [1, 2, 3]):
            return float(tr["umbral"])
    return float(t.get("umbral", UMBRAL_TARDIA))


class AlertaLolmu:
    def __init__(self, path: str | Path = PARAMS_FILE):
        self.p = json.loads(Path(path).read_text(encoding="utf-8"))
        cal = self.p["calendario"]
        self.doy = np.asarray(cal["dia"], float)
        self.cal_media = np.asarray(cal["media"], float)
        self.cal_p10 = np.asarray(cal["p10"], float)
        self.cal_p90 = np.asarray(cal["p90"], float)

    def probabilidad_tardia(self, p_flush: float, nivel: int, fecha: pd.Timestamp) -> float | None:
        """Probabilidad de que en los próximos 7 días emerja ≥ 2 % de la campaña (emergencia tardía).

        Solo dentro del período de flujos tardíos (``desde``–``hasta``, [mes, día]; 1/05–30/06); fuera de él devuelve None
        (del 1/07 al 1/10 rige la recomendación fija de ``recorrida_tardia``).
        Recalibra la probabilidad de flush con logit(p) = a + b·logit(p_flush), con a y b por nivel de información.
        """
        t = self.p.get("tardia")
        if not t or not ventana_siembra(fecha, self):
            return None
        a, b = t["coef"][str(int(nivel))]
        q = float(np.clip(p_flush, 1e-4, 1 - 1e-4))
        return float(1.0 / (1.0 + np.exp(-(a + b * np.log(q / (1 - q))))))

    def calendario(self, fecha: pd.Timestamp, horizonte: int = 7) -> float:
        d0, d1 = pd.Timestamp(fecha).dayofyear, (pd.Timestamp(fecha) + pd.Timedelta(days=horizonte)).dayofyear
        return float(np.interp(d1, self.doy, self.cal_media) - np.interp(d0, self.doy, self.cal_media))

    def curva(self, fechas: pd.DatetimeIndex) -> pd.DataFrame:
        d = pd.DatetimeIndex(fechas).dayofyear
        return pd.DataFrame({"Fecha": fechas, "media": np.interp(d, self.doy, self.cal_media),
                             "p10": np.interp(d, self.doy, self.cal_p10), "p90": np.interp(d, self.doy, self.cal_p90)})

    def clima_tt(self, localidad: str | None = None) -> np.ndarray:
        """Tiempo térmico diario medio por día del año (366 valores): el de la localidad si está, si no el de Bordenave."""
        k = self.p["curva_tt"]
        return np.asarray((k.get("tt_diario_clima_sitio") or {}).get(localidad or "", k["tt_diario_clima"]), float)

    def curva_tiempo_termico(self, dg: pd.Series, primeras: pd.Timestamp, fechas, localidad: str | None = None) -> pd.DataFrame:
        """Emergencia acumulada esperada según el tiempo térmico acumulado desde las primeras plántulas.

        ``dg``: tiempo térmico diario del clima disponible (``tiempo_termico``, base 2 / óptima 20 / crítica 30 °C).
        Más allá del último día con clima se suma el tiempo térmico diario medio histórico de ese día del año
        (de la ``localidad`` si hay climatología propia; si no, el de Bordenave).
        Antes de las primeras plántulas la curva vale 0. Devuelve ``Fecha``, ``media``, ``p10`` y ``p90``.
        """
        k = self.p["curva_tt"]
        fechas = pd.DatetimeIndex(fechas)
        primeras = pd.Timestamp(primeras).normalize()
        clima = self.clima_tt(localidad)
        dias = pd.date_range(primeras + pd.Timedelta(days=1), max(fechas.max(), primeras + pd.Timedelta(days=1)))
        obs = dg.reindex(dias)
        diario = np.where(obs.notna(), obs.values, clima[dias.dayofyear.values - 1])
        acum = pd.Series(np.cumsum(diario), index=dias)
        tt = np.where(fechas > primeras, acum.reindex(fechas).fillna(0.0).values, np.nan)
        out = {"Fecha": fechas}
        for col in ("media", "p10", "p90"):
            out[col] = np.where(np.isnan(tt), 0.0, np.interp(np.nan_to_num(tt), k["tt"], k[col]))
        out = pd.DataFrame(out)
        out.loc[fechas == primeras, ["media", "p10", "p90"]] = [k["media"][0], k["p10"][0], k["p90"][0]]
        return out

    def _ecdf(self, nombre: str, valor: float) -> float:
        ref = np.asarray(self.p["referencias"][nombre], float)
        return float(np.searchsorted(ref, valor, side="right") / len(ref))

    def probabilidad(self, nivel: int, senales: dict) -> float:
        m = self.p["modelos"][str(nivel)]
        z = m["intercepto"]
        for f, b in zip(m["variables"], m["coeficientes"]):
            if f.endswith("^2"):
                z += b * self._ecdf(f[:-2], senales[f[:-2]]) ** 2
            elif f in ("primer",):
                z += b * float(senales[f])
            else:
                z += b * self._ecdf(f, senales[f])
        return float(1 / (1 + np.exp(-z)))


def factor_agotamiento(dias: float, tau: float, beta: float, intensidad: float) -> float:
    """Agotamiento Weibull de la cohorte germinable: 1 − I·(1 − exp[−(t/τ)^β]), con t en días desde el primer flush."""
    t = max(float(dias), 0.0)
    return float(1.0 - intensidad * (1.0 - np.exp(-np.power(t / max(tau, 1e-6), beta))))


def nivel_de(p: float) -> str:
    return "alerta" if p >= ALERTA else ("atencion" if p >= ATENCION else "baja")


def evaluar(clima_raw: pd.DataFrame, fecha: pd.Timestamp, primeras_plantulas: pd.Timestamp | None = None,
            conteos: pd.DataFrame | None = None, modelo: AlertaLolmu | None = None,
            t_base=2.0, t_opt=20.0, t_crit=30.0, calendario: dict | None = None, agotamiento: dict | None = None,
            tope_sin_inicio: bool = False, cero_reciente_dias: int | None = None,
            tope_hasta_ultimo_cero: bool = False) -> dict:
    """Probabilidad de flush para los 7 días posteriores a ``fecha`` con la información disponible.

    Opciones por sitio (ambas desactivadas por defecto):
      * ``calendario``: curva acumulada local (``dia``, ``media``) que reemplaza al calendario de Bordenave en la señal
        de calendario;
      * ``agotamiento``: ``tau``, ``beta``, ``intensidad`` del agotamiento Weibull de la cohorte; multiplica la
        probabilidad según los días transcurridos desde las primeras plántulas (solo si se conocen);
      * ``tope_sin_inicio``: mientras no se vieron plántulas, el nivel llega como máximo a *Atención* (la probabilidad
        no cambia); para sitios donde el inicio es tardío y variable;
      * ``cero_reciente_dias``: si todavía no se vieron plántulas y el último conteo (hasta ``fecha``) fue en cero hace
        como mucho esos días, la probabilidad queda por debajo del umbral de *Atención* (``TOPE_CERO``; *Sin alerta*);
        la probabilidad por clima se conserva en ``probabilidad_clima``;
      * ``tope_hasta_ultimo_cero``: la probabilidad es 0 en las emisiones cuya semana (los 7 días de ``fecha`` a
        ``fecha`` + 6) terminó a más tardar el día del último conteo en cero anterior a cualquier plántula: el conteo
        confirma que en esa semana no hubo emergencia (por ejemplo, un cero el 15/03 confirma la semana del 9/03 al 15/03). La semana que sigue a un conteo en cero se evalúa normalmente,
        porque un cero no anticipa que no vaya a emerger (en la validación de las 7 localidades, 37 % de las semanas
        que siguieron a un conteo en cero tuvieron flush).
    """
    modelo = modelo or AlertaLolmu()
    clima = limpiar_clima(clima_raw)
    clima = clima[clima["Fecha"] >= pd.Timestamp(pd.Timestamp(fecha).year, 1, 1)].reset_index(drop=True)
    suelo = modelo_suelo(clima)
    fecha = pd.Timestamp(fecha).normalize()
    dg = tiempo_termico(clima, t_base, t_opt, t_crit)
    if calendario:
        d0, d1 = fecha.dayofyear, (fecha + pd.Timedelta(days=7)).dayofyear
        dd, mm = np.asarray(calendario["dia"], float), np.asarray(calendario["media"], float)
        cal = float(np.interp(d1, dd, mm) - np.interp(d0, dd, mm))
    else:
        cal = modelo.calendario(fecha)
    senales = {"cal": cal, "lluvia": senal_lluvia(suelo, fecha)}
    g1, primer, n_conteos = senal_conteos(conteos, fecha)
    if primeras_plantulas is None and conteos is not None and not conteos.empty:
        vistas = conteos[(conteos["PLM2"] > 0) & (conteos["Fecha"] <= fecha)]
        primeras_plantulas = vistas["Fecha"].min() if not vistas.empty else None
    iniciada = primeras_plantulas is not None and pd.Timestamp(primeras_plantulas) <= fecha
    senales["tt_inicio"] = float(dg[(dg.index > pd.Timestamp(primeras_plantulas)) & (dg.index <= fecha)].sum()) if iniciada else 0.0
    senales["g1"], senales["primer"] = g1, primer
    nivel = 3 if (iniciada and n_conteos >= 2) else (2 if iniciada else 1)
    p = modelo.probabilidad(nivel, senales)
    if agotamiento and iniciada:
        senales["agotamiento"] = factor_agotamiento((fecha - pd.Timestamp(primeras_plantulas)).days, **agotamiento)
        p *= senales["agotamiento"]
    p_clima = p
    p_tardia = modelo.probabilidad_tardia(p, nivel, fecha)
    nivel_alerta = nivel_de(p)
    tope = bool(tope_sin_inicio and not iniciada and nivel_alerta == "alerta")
    if tope:
        nivel_alerta = "atencion"
    cero_reciente = None
    if tope_hasta_ultimo_cero and conteos is not None and not conteos.empty:
        cc = conteos.dropna(subset=["Fecha", "PLM2"]).sort_values("Fecha")
        positivos = cc.loc[cc["PLM2"] > 0, "Fecha"]
        limite = positivos.min() if not positivos.empty else None
        if primeras_plantulas is not None:
            limite = min(limite, pd.Timestamp(primeras_plantulas)) if limite is not None else pd.Timestamp(primeras_plantulas)
        ceros = cc.loc[(cc["PLM2"] == 0) & ((cc["Fecha"] < limite) if limite is not None else True), "Fecha"]
        ultimo_dia = fecha + pd.Timedelta(days=HORIZONTE_DIAS - 1)  # la semana de la alerta: de fecha a fecha + 6
        if not ceros.empty and ultimo_dia <= ceros.max():
            # semana ya confirmada sin plántulas por un conteo en cero: probabilidad 0 (la de clima queda en p_clima)
            cero_reciente = pd.Timestamp(ceros[ceros >= ultimo_dia].min())
            p = 0.0
            p_tardia = 0.0 if p_tardia is not None else None
            nivel_alerta, tope = nivel_de(p), False
    if cero_reciente is None and cero_reciente_dias is not None and not iniciada and conteos is not None and not conteos.empty:
        previos = conteos[conteos["Fecha"] <= fecha].sort_values("Fecha")
        if not previos.empty and float(previos["PLM2"].iloc[-1]) == 0.0 \
                and (fecha - pd.Timestamp(previos["Fecha"].iloc[-1])).days <= cero_reciente_dias:
            cero_reciente = pd.Timestamp(previos["Fecha"].iloc[-1])
            p = min(p, TOPE_CERO)
            p_tardia = min(p_tardia, TOPE_CERO) if p_tardia is not None else None
            nivel_alerta, tope = nivel_de(p), False
    return {"fecha": fecha, "nivel_info": nivel, "probabilidad": p, "probabilidad_clima": p_clima, "nivel": nivel_alerta, "tope_aplicado": tope,
            "probabilidad_tardia": p_tardia, "umbral_tardia": umbral_tardia(fecha, nivel, modelo),
            "recorrida_tardia": recorrida_tardia(fecha, modelo),
            "cero_reciente": cero_reciente, "senales": senales,
            "primeras_plantulas": pd.Timestamp(primeras_plantulas) if iniciada else None}


def serie(clima_raw: pd.DataFrame, desde: pd.Timestamp, hasta: pd.Timestamp, primeras_plantulas=None,
          conteos=None, modelo: AlertaLolmu | None = None, paso: int = 7, **tt) -> pd.DataFrame:
    """Probabilidad emitida cada ``paso`` días entre dos fechas (cada emisión usa solo lo disponible)."""
    modelo = modelo or AlertaLolmu()
    filas = []
    for f in pd.date_range(desde, hasta, freq=f"{paso}D"):
        r = evaluar(clima_raw, f, primeras_plantulas if (primeras_plantulas is not None and pd.Timestamp(primeras_plantulas) <= f) else None,
                    conteos, modelo, **tt)
        filas.append({"Fecha": f, "probabilidad": r["probabilidad"], "probabilidad_clima": r["probabilidad_clima"],
                      "nivel_info": r["nivel_info"], "nivel": r["nivel"],
                      "cero_reciente": r["cero_reciente"] is not None,
                      "probabilidad_tardia": r["probabilidad_tardia"] if r["probabilidad_tardia"] is not None else np.nan})
    return pd.DataFrame(filas)
