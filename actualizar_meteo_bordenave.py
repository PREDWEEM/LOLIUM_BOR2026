# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 NODO CLIMÁTICO PREDWEEM — BORDENAVE
#
# Serie operativa:
#   • Fechas observadas: SIGA–INTA Bordenave (fuente prioritaria).
#   • Fechas faltantes de SIGA: ECMWF IFS histórico, provisional.
#   • Hoy y próximos 6 días: ECMWF IFS ENS 0.25°, P50 operativo.
#
# Archivo final compatible con PREDWEEM:
#   meteo_daily.csv
# ===============================================================

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


LATITUD = float(os.getenv("LATITUD", "-37.85"))
LONGITUD = float(os.getenv("LONGITUD", "-63.02"))
ZONA_HORARIA = "America/Argentina/Buenos_Aires"

CAMPANIA_START = date(2026, 1, 1)
HORIZONTE_DIAS = 7
TBASE = 2.0

ARCHIVO_MAESTRO_DEFAULT = Path("meteo_daily.csv")
ARCHIVO_SIGA_CACHE = Path("data/siga_bordenave_observado.csv")
DIRECTORIO_PRONOSTICOS = Path("data/historico_pronosticos")
ARCHIVO_ESTADO = Path("data/estado_actualizacion_meteo.json")

SIGA_ARCHIVO_LOCAL = Path(os.getenv("SIGA_LOCAL_FILE", "A872817.xls"))
SIGA_URL_TEMPLATE = os.getenv("SIGA_DOWNLOAD_URL", "").strip()
SIGA_METHOD = os.getenv("SIGA_METHOD", "GET").strip().upper()
SIGA_PARAMS_JSON = os.getenv("SIGA_PARAMS_JSON", "").strip()
SIGA_POST_DATA_JSON = os.getenv("SIGA_POST_DATA_JSON", "").strip()
SIGA_HEADERS_JSON = os.getenv("SIGA_HEADERS_JSON", "").strip()

URL_ECMWF_ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODELO_ECMWF_ENS = "ecmwf_ifs025"
URL_ECMWF_HISTORICO = "https://archive-api.open-meteo.com/v1/archive"
MODELO_ECMWF_HISTORICO = "ecmwf_ifs"

TIMEOUT_SEGUNDOS = 90
REINTENTOS = 4
MIN_MIEMBROS_VALIDOS_ABSOLUTO = 30
FRACCION_MINIMA_MIEMBROS = 0.80
HORAS_VALIDAS_POR_DIA = 24

COLUMNAS_COMPLETAS = [
    "Fecha",
    "TMAX",
    "TMIN",
    "Prec",
    "TMEDIA",
    "TMAX_Media_Ens",
    "TMIN_Media_Ens",
    "TMEDIA_Media_Ens",
    "Prec_Media_Ens",
    "TMAX_P10",
    "TMAX_P50",
    "TMAX_P90",
    "TMIN_P10",
    "TMIN_P50",
    "TMIN_P90",
    "TMEDIA_P10",
    "TMEDIA_P50",
    "TMEDIA_P90",
    "Prec_P10",
    "Prec_P50",
    "Prec_P90",
    "Prob_Prec_ge_1mm",
    "Prob_Prec_ge_5mm",
    "Prob_Prec_ge_10mm",
    "Prob_Prec_ge_30mm",
    "GD_Tb2",
    "Fuente",
    "TipoDato",
    "CalidadDato",
    "N_miembros",
    "Latitud_grilla",
    "Longitud_grilla",
    "Elevacion_grilla_m",
    "Emision_UTC",
]


def hoy_argentina() -> date:
    return datetime.now(ZoneInfo(ZONA_HORARIA)).date()


def fecha_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalizar_nombre_columna(nombre: Any) -> str:
    texto = str(nombre).strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    return texto.strip("_")


def to_float(valor: Any) -> float | None:
    if valor is None or pd.isna(valor):
        return None
    texto = str(valor).strip()
    if not texto:
        return None
    texto = texto.replace(" ", "").replace(",", ".")
    try:
        return float(texto)
    except (TypeError, ValueError):
        return None


def parsear_json_entorno(texto: str, nombre: str) -> dict[str, Any]:
    if not texto:
        return {}
    try:
        valor = json.loads(texto)
    except json.JSONDecodeError as error:
        raise ValueError(f"{nombre} no contiene JSON válido: {error}") from error
    if not isinstance(valor, dict):
        raise ValueError(f"{nombre} debe contener un objeto JSON.")
    return valor


def reemplazar_marcadores(valor: Any, contexto: dict[str, str]) -> Any:
    if isinstance(valor, str):
        return valor.format_map(contexto)
    if isinstance(valor, dict):
        return {k: reemplazar_marcadores(v, contexto) for k, v in valor.items()}
    if isinstance(valor, list):
        return [reemplazar_marcadores(v, contexto) for v in valor]
    return valor


def solicitar_con_reintentos(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = TIMEOUT_SEGUNDOS,
    intentos: int = REINTENTOS,
) -> requests.Response:
    ultimo_error: Exception | None = None
    for intento in range(1, intentos + 1):
        try:
            respuesta = requests.request(
                method=method,
                url=url,
                params=params,
                data=data,
                headers=headers,
                timeout=timeout,
            )
            print(f"URL consultada: {respuesta.url}")
            respuesta.raise_for_status()
            return respuesta
        except requests.RequestException as error:
            ultimo_error = error
            print(f"⚠️ Intento HTTP {intento}/{intentos} fallido: {error}")
            if intento < intentos:
                time.sleep(5 * intento)
    raise RuntimeError(f"No fue posible consultar {url}") from ultimo_error


def asegurar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.copy()
    for columna in COLUMNAS_COMPLETAS:
        if columna not in salida.columns:
            salida[columna] = np.nan
    return salida[COLUMNAS_COMPLETAS]


def escribir_csv_atomico(df: pd.DataFrame, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    df.to_csv(temporal, index=False, float_format="%.3f")
    if destino.exists():
        respaldo = destino.with_suffix(destino.suffix + ".bak")
        shutil.copy2(destino, respaldo)
    temporal.replace(destino)


def resumen_fechas(fechas: list[str] | pd.Index, limite: int = 15) -> str:
    valores = [str(v) for v in list(fechas)]
    muestra = ", ".join(valores[:limite])
    if len(valores) > limite:
        muestra += f", ... ({len(valores)} fechas)"
    return muestra


# -----------------------------------------------------------------
# SIGA–INTA: descarga, lectura y normalización
# -----------------------------------------------------------------

def buscar_archivo_siga_local(archivo_preferido: Path | None = None) -> Path | None:
    candidatos: list[Path] = []
    if archivo_preferido is not None:
        candidatos.append(archivo_preferido)
    candidatos.append(SIGA_ARCHIVO_LOCAL)
    for patron in (
        "A*.xls",
        "A*.xlsx",
        "*siga*.xls",
        "*siga*.xlsx",
        "*siga*.csv",
    ):
        candidatos.extend(Path(".").glob(patron))
    existentes = {c.resolve() for c in candidatos if c.exists()}
    if not existentes:
        return None
    return max(existentes, key=lambda ruta: ruta.stat().st_mtime)


def descargar_siga(fecha_inicio: date, fecha_fin: date) -> tuple[bytes, str, str]:
    if not SIGA_URL_TEMPLATE:
        raise RuntimeError("SIGA_DOWNLOAD_URL no está configurada.")

    contexto = {
        "start": fecha_inicio.isoformat(),
        "end": fecha_fin.isoformat(),
        "start_date": fecha_inicio.isoformat(),
        "end_date": fecha_fin.isoformat(),
    }
    url = reemplazar_marcadores(SIGA_URL_TEMPLATE, contexto)
    params = reemplazar_marcadores(
        parsear_json_entorno(SIGA_PARAMS_JSON, "SIGA_PARAMS_JSON"),
        contexto,
    )
    data = reemplazar_marcadores(
        parsear_json_entorno(SIGA_POST_DATA_JSON, "SIGA_POST_DATA_JSON"),
        contexto,
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0 Safari/537.36"
        ),
        "Accept": (
            "application/vnd.ms-excel,"
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
            "text/csv,*/*"
        ),
        "Referer": "https://siga.inta.gob.ar/",
    }
    headers.update(
        reemplazar_marcadores(
            parsear_json_entorno(SIGA_HEADERS_JSON, "SIGA_HEADERS_JSON"),
            contexto,
        )
    )

    respuesta = solicitar_con_reintentos(
        SIGA_METHOD,
        url,
        params=params or None,
        data=data or None,
        headers=headers,
    )
    contenido = respuesta.content
    if len(contenido) < 100:
        raise ValueError("La descarga SIGA es demasiado pequeña para contener una tabla.")

    tipo = respuesta.headers.get("content-type", "").lower()
    disposicion = respuesta.headers.get("content-disposition", "")
    coincidencia = re.search(r'filename="?([^";]+)', disposicion, flags=re.IGNORECASE)
    nombre = coincidencia.group(1) if coincidencia else Path(url).name or "siga_bordenave.xls"

    inicio_texto = contenido[:300].lower()
    if b"<html" in inicio_texto or b"<!doctype html" in inicio_texto:
        raise ValueError("SIGA devolvió una página HTML y no un archivo de datos.")
    return contenido, nombre, tipo


def leer_tabla_siga_desde_bytes(
    contenido: bytes,
    nombre: str,
    tipo_contenido: str = "",
) -> pd.DataFrame:
    buffer = io.BytesIO(contenido)
    nombre_lower = nombre.lower()
    es_xls = (
        contenido.startswith(b"\xd0\xcf\x11\xe0")
        or nombre_lower.endswith(".xls")
        or "application/vnd.ms-excel" in tipo_contenido
    )
    es_xlsx = (
        contenido.startswith(b"PK")
        or nombre_lower.endswith(".xlsx")
        or "spreadsheetml" in tipo_contenido
    )
    if es_xls:
        return pd.read_excel(buffer, sheet_name="Datos diarios", engine="xlrd")
    if es_xlsx:
        return pd.read_excel(buffer, sheet_name="Datos diarios", engine="openpyxl")

    texto = contenido.decode("utf-8-sig", errors="replace")
    for separador in (";", ",", "\t"):
        candidato = pd.read_csv(io.StringIO(texto), sep=separador)
        if candidato.shape[1] >= 4:
            return candidato
    raise ValueError("No se pudo reconocer el formato de la descarga SIGA.")


def leer_tabla_siga_local(archivo: Path) -> pd.DataFrame:
    sufijo = archivo.suffix.lower()
    if sufijo == ".xls":
        return pd.read_excel(archivo, sheet_name="Datos diarios", engine="xlrd")
    if sufijo == ".xlsx":
        return pd.read_excel(archivo, sheet_name="Datos diarios", engine="openpyxl")
    if sufijo == ".csv":
        for separador in (";", ",", "\t"):
            candidato = pd.read_csv(archivo, sep=separador)
            if candidato.shape[1] >= 4:
                return candidato
    raise ValueError(f"Formato SIGA local no soportado: {archivo}")


def seleccionar_columna(tabla: pd.DataFrame, candidatos: list[str]) -> str | None:
    for candidato in candidatos:
        if candidato in tabla.columns:
            return candidato
    return None


def normalizar_dataframe_siga(
    tabla: pd.DataFrame,
    fecha_limite_exclusiva: date,
) -> pd.DataFrame:
    if tabla.empty:
        raise ValueError("La tabla SIGA está vacía.")

    tabla = tabla.copy()
    tabla.columns = [normalizar_nombre_columna(c) for c in tabla.columns]
    alias = {
        "fecha": ["fecha", "date"],
        "tmedia": [
            "temperatura_abrigo_150cm",
            "temperatura_media",
            "temperatura_promedio",
            "tmedia",
            "temp_media",
        ],
        "tmax": [
            "temperatura_abrigo_150cm_maxima",
            "temperatura_maxima",
            "temperatura_max",
            "tmax",
            "temp_max",
        ],
        "tmin": [
            "temperatura_abrigo_150cm_minima",
            "temperatura_minima",
            "temperatura_min",
            "tmin",
            "temp_min",
        ],
        "prec": [
            "precipitacion_pluviometrica",
            "precipitacion",
            "precipitacion_diaria",
            "lluvia",
            "prec",
        ],
    }
    seleccion = {
        destino: seleccionar_columna(tabla, candidatos)
        for destino, candidatos in alias.items()
    }
    obligatorias = {"fecha", "tmax", "tmin", "prec"}
    faltantes = sorted(k for k in obligatorias if seleccion.get(k) is None)
    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias en SIGA: "
            + ", ".join(faltantes)
            + ". Columnas encontradas: "
            + ", ".join(tabla.columns)
        )

    fechas_crudas = tabla[seleccion["fecha"]]
    fechas = pd.Series(
        pd.to_datetime(fechas_crudas, errors="coerce", yearfirst=True),
        index=tabla.index,
    )
    faltantes_fecha = fechas.isna()
    if faltantes_fecha.any():
        fechas.loc[faltantes_fecha] = pd.to_datetime(
            fechas_crudas.loc[faltantes_fecha],
            errors="coerce",
            dayfirst=True,
        )

    salida = pd.DataFrame(
        {
            "Fecha": fechas,
            "TMAX": tabla[seleccion["tmax"]].map(to_float),
            "TMIN": tabla[seleccion["tmin"]].map(to_float),
            "Prec": tabla[seleccion["prec"]].map(to_float),
        }
    )
    tmedia_calculada = (salida["TMAX"] + salida["TMIN"]) / 2.0
    if seleccion.get("tmedia") is not None:
        tmedia_siga = tabla[seleccion["tmedia"]].map(to_float)
        salida["TMEDIA"] = tmedia_siga.where(tmedia_siga.notna(), tmedia_calculada)
    else:
        salida["TMEDIA"] = tmedia_calculada

    salida = salida.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec", "TMEDIA"])
    salida["Fecha"] = pd.to_datetime(salida["Fecha"], errors="coerce").dt.normalize()
    salida = salida.dropna(subset=["Fecha"])

    salida.loc[~salida["TMAX"].between(-25, 55), "TMAX"] = np.nan
    salida.loc[~salida["TMIN"].between(-35, 45), "TMIN"] = np.nan
    salida.loc[~salida["Prec"].between(0, 500), "Prec"] = np.nan
    salida = salida.dropna(subset=["TMAX", "TMIN", "Prec", "TMEDIA"])
    salida = salida.loc[salida["TMAX"] >= salida["TMIN"]].copy()
    salida = salida.loc[
        (salida["Fecha"].dt.date >= CAMPANIA_START)
        & (salida["Fecha"].dt.date < fecha_limite_exclusiva)
    ].copy()

    salida["Fecha"] = salida["Fecha"].dt.strftime("%Y-%m-%d")
    salida = (
        salida.drop_duplicates(subset=["Fecha"], keep="last")
        .sort_values("Fecha")
        .reset_index(drop=True)
    )
    salida["GD_Tb2"] = np.maximum(0.0, salida["TMEDIA"] - TBASE)
    salida["Fuente"] = "SIGA_INTA_BORDENAVE"
    salida["TipoDato"] = "Observado"
    salida["CalidadDato"] = "Observado_estacion"
    salida["Emision_UTC"] = fecha_utc_iso()
    return asegurar_columnas(salida)


def obtener_siga_dataframe(
    fecha_inicio: date,
    fecha_fin: date,
    archivo_forzado: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    errores: list[str] = []

    if SIGA_URL_TEMPLATE and archivo_forzado is None:
        try:
            print("📡 Descargando observaciones diarias SIGA Bordenave...")
            contenido, nombre, tipo = descargar_siga(fecha_inicio, fecha_fin)
            tabla = leer_tabla_siga_desde_bytes(
                contenido,
                nombre=nombre,
                tipo_contenido=tipo,
            )
            df = normalizar_dataframe_siga(
                tabla,
                fecha_limite_exclusiva=fecha_fin + timedelta(days=1),
            )
            escribir_csv_atomico(df, ARCHIVO_SIGA_CACHE)
            return df, "SIGA_remoto"
        except Exception as error:
            errores.append(f"SIGA remoto: {error}")
            print(f"⚠️ Falló la consulta remota SIGA: {error}")

    archivo_local = buscar_archivo_siga_local(archivo_forzado)
    if archivo_local is not None:
        try:
            print(f"📄 Leyendo respaldo SIGA local: {archivo_local}")
            tabla = leer_tabla_siga_local(archivo_local)
            df = normalizar_dataframe_siga(
                tabla,
                fecha_limite_exclusiva=fecha_fin + timedelta(days=1),
            )
            escribir_csv_atomico(df, ARCHIVO_SIGA_CACHE)
            return df, f"SIGA_local:{archivo_local.name}"
        except Exception as error:
            errores.append(f"SIGA local: {error}")
            print(f"⚠️ Falló el archivo SIGA local: {error}")

    if ARCHIVO_SIGA_CACHE.exists():
        try:
            print("📦 Utilizando caché observado de SIGA Bordenave.")
            cache = pd.read_csv(ARCHIVO_SIGA_CACHE)
            cache = asegurar_columnas(cache)
            cache["Fecha_dt"] = pd.to_datetime(cache["Fecha"], errors="coerce")
            cache = cache.dropna(subset=["Fecha_dt"])
            cache = cache.loc[
                cache["Fecha_dt"].dt.date < fecha_fin + timedelta(days=1)
            ].copy()
            cache["Fecha"] = cache["Fecha_dt"].dt.strftime("%Y-%m-%d")
            cache = cache.drop(columns=["Fecha_dt"])
            return asegurar_columnas(cache), "SIGA_cache"
        except Exception as error:
            errores.append(f"Caché SIGA: {error}")

    raise RuntimeError("No fue posible obtener datos SIGA. " + " | ".join(errores))


# -----------------------------------------------------------------
# Puente provisional ECMWF IFS histórico para cualquier hueco SIGA
# -----------------------------------------------------------------

def fechas_faltantes_siga(
    observaciones: pd.DataFrame,
    fecha_inicio: date,
    fecha_fin: date,
) -> list[date]:
    if fecha_inicio > fecha_fin:
        return []
    esperadas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    presentes = pd.DatetimeIndex(
        pd.to_datetime(observaciones["Fecha"], errors="coerce").dropna()
    ).normalize()
    return [ts.date() for ts in esperadas.difference(presentes)]


def rangos_contiguos(fechas: list[date]) -> list[tuple[date, date]]:
    if not fechas:
        return []
    ordenadas = sorted(set(fechas))
    rangos: list[tuple[date, date]] = []
    inicio = anterior = ordenadas[0]
    for actual in ordenadas[1:]:
        if actual == anterior + timedelta(days=1):
            anterior = actual
            continue
        rangos.append((inicio, anterior))
        inicio = anterior = actual
    rangos.append((inicio, anterior))
    return rangos


def _parametros_ecmwf_historico(
    fecha_inicio: date,
    fecha_fin: date,
) -> dict[str, Any]:
    return {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": fecha_inicio.isoformat(),
        "end_date": fecha_fin.isoformat(),
        "models": MODELO_ECMWF_HISTORICO,
        "timezone": ZONA_HORARIA,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }


def _normalizar_provisional(
    salida: pd.DataFrame,
    datos: dict[str, Any],
    fecha_inicio: date,
    fecha_fin: date,
) -> pd.DataFrame:
    salida = salida.copy()
    salida["Fecha"] = pd.to_datetime(salida["Fecha"], errors="coerce").dt.normalize()
    for columna in ("TMAX", "TMIN", "Prec", "TMEDIA"):
        salida[columna] = pd.to_numeric(salida[columna], errors="coerce")
    salida = salida.dropna(subset=["Fecha"])
    salida = salida.loc[
        (salida["Fecha"].dt.date >= fecha_inicio)
        & (salida["Fecha"].dt.date <= fecha_fin)
    ].copy()

    esperadas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    faltan_dias = esperadas.difference(pd.DatetimeIndex(salida["Fecha"]))
    if len(faltan_dias):
        raise ValueError(
            "ECMWF histórico no devolvió todas las fechas provisionales: "
            + resumen_fechas(faltan_dias.strftime("%Y-%m-%d"))
        )
    malas = salida[["TMAX", "TMIN", "Prec", "TMEDIA"]].isna().any(axis=1)
    if malas.any():
        raise ValueError(
            "ECMWF histórico contiene datos nulos en: "
            + resumen_fechas(
                salida.loc[malas, "Fecha"].dt.strftime("%Y-%m-%d").tolist()
            )
        )
    if (salida["TMAX"] < salida["TMIN"]).any():
        raise ValueError("ECMWF histórico contiene TMAX menor que TMIN.")
    if (salida["Prec"] < 0).any():
        raise ValueError("ECMWF histórico contiene precipitación negativa.")

    for variable in ("TMAX", "TMIN", "TMEDIA", "Prec"):
        salida[f"{variable}_P50"] = salida[variable]
    salida["GD_Tb2"] = np.maximum(0.0, salida["TMEDIA"] - TBASE)
    salida["Fuente"] = "ECMWF_IFS_HISTORICO"
    salida["TipoDato"] = "Provisional"
    salida["CalidadDato"] = "Provisional_hasta_reemplazo_SIGA"
    salida["N_miembros"] = 1
    salida["Latitud_grilla"] = datos.get("latitude", np.nan)
    salida["Longitud_grilla"] = datos.get("longitude", np.nan)
    salida["Elevacion_grilla_m"] = datos.get("elevation", np.nan)
    salida["Emision_UTC"] = fecha_utc_iso()
    salida["Fecha"] = salida["Fecha"].dt.strftime("%Y-%m-%d")
    return asegurar_columnas(salida)


def _cargar_provisional_diario(
    fecha_inicio: date,
    fecha_fin: date,
) -> pd.DataFrame:
    params = {
        **_parametros_ecmwf_historico(fecha_inicio, fecha_fin),
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "temperature_2m_mean,precipitation_sum"
        ),
    }
    respuesta = solicitar_con_reintentos("GET", URL_ECMWF_HISTORICO, params=params)
    datos = respuesta.json()
    if datos.get("error"):
        raise RuntimeError(f"ECMWF histórico devolvió un error: {datos.get('reason')}")

    daily = datos.get("daily", {})
    requeridas = {
        "time",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    }
    faltantes = requeridas.difference(daily)
    if faltantes:
        raise ValueError(
            "Faltan variables en ECMWF histórico: " + ", ".join(sorted(faltantes))
        )

    tmax = pd.to_numeric(pd.Series(daily["temperature_2m_max"]), errors="coerce")
    tmin = pd.to_numeric(pd.Series(daily["temperature_2m_min"]), errors="coerce")
    prec = pd.to_numeric(pd.Series(daily["precipitation_sum"]), errors="coerce")
    if "temperature_2m_mean" in daily:
        tmedia = pd.to_numeric(pd.Series(daily["temperature_2m_mean"]), errors="coerce")
    else:
        tmedia = (tmax + tmin) / 2.0

    salida = pd.DataFrame(
        {
            "Fecha": daily["time"],
            "TMAX": tmax,
            "TMIN": tmin,
            "Prec": prec,
            "TMEDIA": tmedia,
        }
    )
    return _normalizar_provisional(salida, datos, fecha_inicio, fecha_fin)


def _cargar_provisional_horario(
    fecha_inicio: date,
    fecha_fin: date,
) -> pd.DataFrame:
    params = {
        **_parametros_ecmwf_historico(fecha_inicio, fecha_fin),
        "hourly": "temperature_2m,precipitation",
    }
    respuesta = solicitar_con_reintentos("GET", URL_ECMWF_HISTORICO, params=params)
    datos = respuesta.json()
    if datos.get("error"):
        raise RuntimeError(f"ECMWF histórico devolvió un error: {datos.get('reason')}")

    hourly = datos.get("hourly", {})
    requeridas = {"time", "temperature_2m", "precipitation"}
    faltantes = requeridas.difference(hourly)
    if faltantes:
        raise ValueError(
            "Faltan variables horarias en ECMWF histórico: "
            + ", ".join(sorted(faltantes))
        )

    bruto = pd.DataFrame(
        {
            "Hora": pd.to_datetime(hourly["time"], errors="coerce"),
            "Temp": pd.to_numeric(
                pd.Series(hourly["temperature_2m"]), errors="coerce"
            ),
            "Prec_h": pd.to_numeric(
                pd.Series(hourly["precipitation"]), errors="coerce"
            ),
        }
    ).dropna(subset=["Hora"])
    bruto["Fecha"] = bruto["Hora"].dt.normalize()
    diario = (
        bruto.groupby("Fecha", as_index=False)
        .agg(
            TMAX=("Temp", "max"),
            TMIN=("Temp", "min"),
            TMEDIA=("Temp", "mean"),
            Prec=("Prec_h", "sum"),
            Horas_T=("Temp", "count"),
            Horas_P=("Prec_h", "count"),
        )
    )
    incompletos = diario[
        (diario["Horas_T"] != HORAS_VALIDAS_POR_DIA)
        | (diario["Horas_P"] != HORAS_VALIDAS_POR_DIA)
    ]
    if not incompletos.empty:
        raise ValueError(
            "ECMWF histórico horario contiene días incompletos: "
            + resumen_fechas(
                incompletos["Fecha"].dt.strftime("%Y-%m-%d").tolist()
            )
        )
    return _normalizar_provisional(
        diario[["Fecha", "TMAX", "TMIN", "Prec", "TMEDIA"]],
        datos,
        fecha_inicio,
        fecha_fin,
    )


def cargar_provisionales_ecmwf(fechas: list[date]) -> pd.DataFrame:
    bloques: list[pd.DataFrame] = []
    for fecha_inicio, fecha_fin in rangos_contiguos(fechas):
        print(
            "🧩 Completando fechas faltantes de SIGA con ECMWF IFS: "
            f"{fecha_inicio} a {fecha_fin}..."
        )
        try:
            bloque = _cargar_provisional_diario(fecha_inicio, fecha_fin)
        except Exception as error_diario:
            print(
                "⚠️ Falló el bloque diario provisional; "
                f"se intentará agregación horaria: {error_diario}"
            )
            bloque = _cargar_provisional_horario(fecha_inicio, fecha_fin)
        bloques.append(bloque)

    if not bloques:
        return pd.DataFrame(columns=COLUMNAS_COMPLETAS)
    salida = pd.concat(bloques, ignore_index=True)
    salida = (
        salida.sort_values("Fecha")
        .drop_duplicates(subset=["Fecha"], keep="last")
        .reset_index(drop=True)
    )
    return asegurar_columnas(salida)


# -----------------------------------------------------------------
# ECMWF IFS ENS: control por miembro y P50 operativo
# -----------------------------------------------------------------

def consultar_ecmwf_ens() -> dict[str, Any]:
    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "timezone": ZONA_HORARIA,
        "models": MODELO_ECMWF_ENS,
        "hourly": "temperature_2m,precipitation",
        "forecast_days": HORIZONTE_DIAS,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
        "cell_selection": "land",
    }
    respuesta = solicitar_con_reintentos("GET", URL_ECMWF_ENS, params=params)
    datos = respuesta.json()
    if datos.get("error"):
        raise RuntimeError(f"ECMWF ENS devolvió un error: {datos.get('reason')}")
    return datos


def mapear_miembros(
    hourly: dict[str, Any],
    variable_base: str,
) -> dict[str, str]:
    patron = re.compile(rf"^{re.escape(variable_base)}(?:_member(\d+))?$")
    resultado: dict[str, str] = {}
    for clave, valor in hourly.items():
        if not isinstance(valor, list):
            continue
        coincidencia = patron.match(clave)
        if not coincidencia:
            continue
        miembro = coincidencia.group(1)
        identificador = "control" if miembro is None else f"member{int(miembro):03d}"
        resultado[identificador] = clave
    return resultado


def procesar_ecmwf_ens(datos: dict[str, Any]) -> pd.DataFrame:
    hourly = datos.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("La respuesta de Open-Meteo no contiene datos horarios.")

    tiempos = pd.Series(pd.to_datetime(hourly["time"], errors="coerce"))
    if tiempos.isna().any():
        raise ValueError("El pronóstico contiene fechas horarias inválidas.")

    temp_por_miembro = mapear_miembros(hourly, "temperature_2m")
    prec_por_miembro = mapear_miembros(hourly, "precipitation")
    miembros_comunes = sorted(set(temp_por_miembro).intersection(prec_por_miembro))
    if not miembros_comunes:
        raise ValueError("No existen miembros emparejados de temperatura y precipitación.")

    requeridos = max(
        MIN_MIEMBROS_VALIDOS_ABSOLUTO,
        math.ceil(len(miembros_comunes) * FRACCION_MINIMA_MIEMBROS),
    )
    if len(miembros_comunes) < requeridos:
        raise ValueError(
            f"Solo hay {len(miembros_comunes)} miembros emparejados; "
            f"se requieren al menos {requeridos}."
        )

    matriz_diaria: list[pd.DataFrame] = []
    for identificador in miembros_comunes:
        temp = pd.to_numeric(
            pd.Series(hourly[temp_por_miembro[identificador]]),
            errors="coerce",
        )
        prec = pd.to_numeric(
            pd.Series(hourly[prec_por_miembro[identificador]]),
            errors="coerce",
        )
        if len(temp) != len(tiempos) or len(prec) != len(tiempos):
            print(f"⚠️ Miembro {identificador} descartado por longitud inconsistente.")
            continue

        miembro = pd.DataFrame(
            {
                "Hora": tiempos,
                "Temp": temp,
                "Prec_h": prec,
            }
        )
        miembro["Fecha"] = miembro["Hora"].dt.normalize()
        diario = (
            miembro.groupby("Fecha", as_index=False)
            .agg(
                TMAX=("Temp", "max"),
                TMIN=("Temp", "min"),
                TMEDIA=("Temp", "mean"),
                Prec=("Prec_h", "sum"),
                Horas_T=("Temp", "count"),
                Horas_P=("Prec_h", "count"),
            )
        )
        valido = (
            (diario["Horas_T"] == HORAS_VALIDAS_POR_DIA)
            & (diario["Horas_P"] == HORAS_VALIDAS_POR_DIA)
            & diario[["TMAX", "TMIN", "TMEDIA", "Prec"]].notna().all(axis=1)
            & (diario["TMAX"] >= diario["TMIN"])
            & (diario["Prec"] >= 0)
        )
        diario = diario.loc[
            valido,
            ["Fecha", "TMAX", "TMIN", "TMEDIA", "Prec"],
        ]
        diario["miembro"] = identificador
        matriz_diaria.append(diario)

    if not matriz_diaria:
        raise ValueError("Ningún miembro ECMWF ENS produjo días completos válidos.")

    todos = pd.concat(matriz_diaria, ignore_index=True)
    registros: list[dict[str, Any]] = []
    emision = fecha_utc_iso()
    lat_grid = datos.get("latitude", np.nan)
    lon_grid = datos.get("longitude", np.nan)
    elev_grid = datos.get("elevation", np.nan)

    for fecha, grupo in todos.groupby("Fecha"):
        n_validos = int(grupo["miembro"].nunique())
        if n_validos < requeridos:
            raise ValueError(
                f"El día {pd.Timestamp(fecha).date()} tiene {n_validos} miembros válidos; "
                f"se requieren {requeridos}."
            )

        tmax = grupo["TMAX"]
        tmin = grupo["TMIN"]
        tmedia = grupo["TMEDIA"]
        prec = grupo["Prec"]

        tmax_p50 = float(tmax.quantile(0.50))
        tmin_p50 = float(tmin.quantile(0.50))
        tmedia_p50 = float(tmedia.quantile(0.50))
        prec_p50 = float(prec.quantile(0.50))

        registros.append(
            {
                "Fecha": pd.Timestamp(fecha).strftime("%Y-%m-%d"),
                "TMAX": tmax_p50,
                "TMIN": tmin_p50,
                "Prec": prec_p50,
                "TMEDIA": tmedia_p50,
                "TMAX_Media_Ens": float(tmax.mean()),
                "TMIN_Media_Ens": float(tmin.mean()),
                "TMEDIA_Media_Ens": float(tmedia.mean()),
                "Prec_Media_Ens": float(prec.mean()),
                "TMAX_P10": float(tmax.quantile(0.10)),
                "TMAX_P50": tmax_p50,
                "TMAX_P90": float(tmax.quantile(0.90)),
                "TMIN_P10": float(tmin.quantile(0.10)),
                "TMIN_P50": tmin_p50,
                "TMIN_P90": float(tmin.quantile(0.90)),
                "TMEDIA_P10": float(tmedia.quantile(0.10)),
                "TMEDIA_P50": tmedia_p50,
                "TMEDIA_P90": float(tmedia.quantile(0.90)),
                "Prec_P10": float(prec.quantile(0.10)),
                "Prec_P50": prec_p50,
                "Prec_P90": float(prec.quantile(0.90)),
                "Prob_Prec_ge_1mm": float((prec >= 1.0).mean() * 100.0),
                "Prob_Prec_ge_5mm": float((prec >= 5.0).mean() * 100.0),
                "Prob_Prec_ge_10mm": float((prec >= 10.0).mean() * 100.0),
                "Prob_Prec_ge_30mm": float((prec >= 30.0).mean() * 100.0),
                "GD_Tb2": max(0.0, tmedia_p50 - TBASE),
                "Fuente": "ECMWF_IFS_ENS_025",
                "TipoDato": "Pronostico",
                "CalidadDato": "Mediana_ensamble_P50",
                "N_miembros": n_validos,
                "Latitud_grilla": lat_grid,
                "Longitud_grilla": lon_grid,
                "Elevacion_grilla_m": elev_grid,
                "Emision_UTC": emision,
            }
        )

    salida = asegurar_columnas(pd.DataFrame(registros))
    salida = salida.sort_values("Fecha").reset_index(drop=True)
    if salida[["TMAX", "TMIN", "Prec", "TMEDIA"]].isna().any().any():
        raise ValueError("La serie ECMWF ENS contiene valores operativos nulos.")
    return salida


def cargar_pronostico_ecmwf() -> pd.DataFrame:
    datos = consultar_ecmwf_ens()
    pronostico = procesar_ecmwf_ens(datos)
    DIRECTORIO_PRONOSTICOS.mkdir(parents=True, exist_ok=True)
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archivo = (
        DIRECTORIO_PRONOSTICOS
        / f"ecmwf_ifs_ens_025_bordenave_{marca}.csv"
    )
    escribir_csv_atomico(pronostico, archivo)
    return pronostico


# -----------------------------------------------------------------
# Ensamble final, prioridades y validación
# -----------------------------------------------------------------

def calcular_huecos(
    df: pd.DataFrame,
    fecha_inicio: date,
    fecha_fin: date,
) -> list[str]:
    if fecha_inicio > fecha_fin:
        return []
    esperadas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    presentes = pd.DatetimeIndex(
        pd.to_datetime(df["Fecha"], errors="coerce").dropna()
    ).normalize()
    return list(esperadas.difference(presentes).strftime("%Y-%m-%d"))


def validar_serie_final(df: pd.DataFrame, fecha_final: date) -> None:
    if df.empty:
        raise ValueError("La serie meteorológica final está vacía.")

    fechas = pd.to_datetime(df["Fecha"], errors="coerce")
    if fechas.isna().any():
        raise ValueError("La serie final contiene fechas inválidas.")
    if fechas.duplicated().any():
        raise ValueError("La serie final contiene fechas duplicadas.")

    criticas = df[["TMAX", "TMIN", "Prec", "TMEDIA"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if criticas.isna().any().any():
        malas = fechas[criticas.isna().any(axis=1)].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "La serie final contiene datos meteorológicos nulos en: "
            + resumen_fechas(malas.tolist())
        )
    if (criticas["TMAX"] < criticas["TMIN"]).any():
        raise ValueError("La serie final contiene TMAX menor que TMIN.")
    if (criticas["Prec"] < 0).any():
        raise ValueError("La serie final contiene precipitación negativa.")

    faltantes = calcular_huecos(df, CAMPANIA_START, fecha_final)
    if faltantes:
        raise ValueError(
            "La serie final no es diaria y continua. Faltan: "
            + resumen_fechas(faltantes)
        )

    tipos_validos = {"Observado", "Provisional", "Pronostico"}
    desconocidos = set(df["TipoDato"].dropna().astype(str)) - tipos_validos
    if desconocidos:
        raise ValueError(f"Tipos de dato desconocidos: {sorted(desconocidos)}")

    hoy = pd.Timestamp(hoy_argentina())
    observados = df["TipoDato"].astype(str).eq("Observado")
    provisionales = df["TipoDato"].astype(str).eq("Provisional")
    pron = df["TipoDato"].astype(str).eq("Pronostico")
    if (fechas[observados] >= hoy).any():
        raise ValueError("Una observación SIGA fue ubicada en hoy o el futuro.")
    if (fechas[provisionales] >= hoy).any():
        raise ValueError("Un dato provisional fue ubicado en hoy o el futuro.")
    if (fechas[pron] < hoy).any():
        raise ValueError("Un pronóstico fue ubicado en una fecha vencida.")
    if not pron.any():
        raise ValueError("La serie final no contiene pronóstico.")

    pares = (
        ("TMAX", "TMAX_P50"),
        ("TMIN", "TMIN_P50"),
        ("TMEDIA", "TMEDIA_P50"),
        ("Prec", "Prec_P50"),
    )
    for operativo, percentil in pares:
        a = pd.to_numeric(df.loc[pron, operativo], errors="coerce")
        b = pd.to_numeric(df.loc[pron, percentil], errors="coerce")
        if not np.allclose(a, b, equal_nan=False, atol=1e-9):
            raise ValueError(
                f"El pronóstico no usa coherentemente {percentil} como {operativo}."
            )

    miembros = pd.to_numeric(df.loc[pron, "N_miembros"], errors="coerce")
    if miembros.isna().any() or (miembros < MIN_MIEMBROS_VALIDOS_ABSOLUTO).any():
        raise ValueError("El pronóstico tiene menos de 30 miembros válidos.")


def construir_meteo_daily(
    output: Path = ARCHIVO_MAESTRO_DEFAULT,
    siga_file: Path | None = None,
) -> pd.DataFrame:
    hoy = hoy_argentina()
    ayer = hoy - timedelta(days=1)

    observaciones, estado_siga = obtener_siga_dataframe(
        CAMPANIA_START,
        ayer,
        archivo_forzado=siga_file,
    )
    if observaciones.empty:
        raise ValueError("SIGA no aportó ninguna observación válida.")

    huecos_siga = fechas_faltantes_siga(
        observaciones,
        CAMPANIA_START,
        ayer,
    )
    provisionales = cargar_provisionales_ecmwf(huecos_siga)

    pronostico = cargar_pronostico_ecmwf()
    pronostico = pronostico.loc[
        pd.to_datetime(pronostico["Fecha"]).dt.date >= hoy
    ].copy()
    if pronostico.empty:
        raise ValueError("ECMWF ENS no entregó el pronóstico desde hoy.")

    combinado = pd.concat(
        [observaciones, provisionales, pronostico],
        ignore_index=True,
    )
    combinado = asegurar_columnas(combinado)
    combinado["Fecha_dt"] = pd.to_datetime(combinado["Fecha"], errors="coerce")
    combinado = combinado.dropna(subset=["Fecha_dt"])

    prioridad = combinado["TipoDato"].map(
        {"Observado": 0, "Provisional": 1, "Pronostico": 2}
    ).fillna(9)
    combinado["_prioridad"] = prioridad
    combinado = combinado.sort_values(["Fecha_dt", "_prioridad"])
    combinado = combinado.drop_duplicates(subset=["Fecha_dt"], keep="first")
    combinado = combinado.sort_values("Fecha_dt")

    fecha_final = pd.to_datetime(pronostico["Fecha"]).max().date()
    combinado = combinado.loc[
        (combinado["Fecha_dt"].dt.date >= CAMPANIA_START)
        & (combinado["Fecha_dt"].dt.date <= fecha_final)
    ].copy()
    combinado["Fecha"] = combinado["Fecha_dt"].dt.strftime("%Y-%m-%d")
    combinado = combinado.drop(columns=["Fecha_dt", "_prioridad"])
    combinado = asegurar_columnas(combinado).reset_index(drop=True)

    validar_serie_final(combinado, fecha_final)
    escribir_csv_atomico(combinado, output)

    fechas_obs = pd.to_datetime(observaciones["Fecha"], errors="coerce").dropna()
    ultima_observacion = fechas_obs.max().date()
    estado = {
        "ejecucion_utc": fecha_utc_iso(),
        "sitio": "Bordenave / INTA",
        "latitud": LATITUD,
        "longitud": LONGITUD,
        "estado_siga": estado_siga,
        "ultima_observacion_siga": ultima_observacion.isoformat(),
        "huecos_siga": [fecha.isoformat() for fecha in huecos_siga],
        "fuente_provisional": (
            "ECMWF_IFS_HISTORICO" if not provisionales.empty else None
        ),
        "filas_provisionales": int(len(provisionales)),
        "inicio_provisional": (
            str(provisionales["Fecha"].min()) if not provisionales.empty else None
        ),
        "fin_provisional": (
            str(provisionales["Fecha"].max()) if not provisionales.empty else None
        ),
        "fuente_pronostico": "ECMWF_IFS_ENS_025",
        "estadistico_operativo": "P50",
        "inicio_pronostico": str(pronostico["Fecha"].min()),
        "fin_pronostico": str(pronostico["Fecha"].max()),
        "miembros_validos_min": int(
            pd.to_numeric(pronostico["N_miembros"], errors="coerce").min()
        ),
        "huecos_finales": calcular_huecos(combinado, CAMPANIA_START, fecha_final),
    }
    ARCHIVO_ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_ESTADO.write_text(
        json.dumps(estado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ Archivo actualizado: {output}")
    print(f"✅ Observaciones SIGA: {len(observaciones)} filas ({estado_siga})")
    print(f"✅ Huecos SIGA cubiertos provisionalmente: {len(provisionales)}")
    print(f"✅ Pronóstico ECMWF ENS P50: {len(pronostico)} filas")
    print(
        "✅ Miembros válidos mínimos por día: "
        f"{estado['miembros_validos_min']}"
    )
    print(f"✅ Coordenadas: lat={LATITUD}, lon={LONGITUD}")
    print(combinado.tail(15).to_string(index=False))
    return combinado


def validar_siga(siga_file: Path | None = None) -> None:
    hoy = hoy_argentina()
    ayer = hoy - timedelta(days=1)
    observaciones, estado_siga = obtener_siga_dataframe(
        CAMPANIA_START,
        ayer,
        archivo_forzado=siga_file,
    )
    huecos = fechas_faltantes_siga(observaciones, CAMPANIA_START, ayer)
    print(f"✅ SIGA accesible: {estado_siga}")
    print(f"Filas observadas válidas: {len(observaciones)}")
    if not observaciones.empty:
        print(
            f"Rango observado: {observaciones['Fecha'].min()} "
            f"a {observaciones['Fecha'].max()}"
        )
    print(f"Huecos a cubrir provisionalmente: {len(huecos)}")
    if huecos:
        print(resumen_fechas([fecha.isoformat() for fecha in huecos]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Actualiza meteo_daily.csv para PREDWEEM Bordenave."
    )
    parser.add_argument(
        "--output",
        default=str(ARCHIVO_MAESTRO_DEFAULT),
        help="Archivo de salida CSV.",
    )
    parser.add_argument(
        "--siga-file",
        default=None,
        help="Archivo SIGA local opcional XLS/XLSX/CSV.",
    )
    parser.add_argument(
        "--solo-validar-siga",
        action="store_true",
        help="Solo valida SIGA y actualiza la caché observada.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    siga_file = Path(args.siga_file) if args.siga_file else None
    try:
        if args.solo_validar_siga:
            validar_siga(siga_file)
        else:
            construir_meteo_daily(output=output, siga_file=siga_file)
        return 0
    except Exception as error:
        print(
            f"❌ Error: {error}. No se reemplazó {output}.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
