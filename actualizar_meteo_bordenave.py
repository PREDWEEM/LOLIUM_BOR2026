# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 NODO CLIMÁTICO PREDWEEM — BORDENAVE
#
# Serie operativa:
#   • Fechas vencidas (< hoy): observaciones diarias SIGA–INTA.
#   • Hoy y próximos 6 días: Open-Meteo Ensemble API / ECMWF IFS ENS.
#
# Archivo final compatible con PREDWEEM:
#   meteo_daily.csv
#
# El script NO transforma pronósticos vencidos en observaciones.
# Si falta una fecha observada en SIGA, la informa como hueco.
# ===============================================================

from __future__ import annotations

import argparse
import io
import json
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


# ===============================================================
# CONFIGURACIÓN GENERAL
# ===============================================================

LATITUD = -37.85
LONGITUD = -63.02
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

RELLENAR_HUECOS_CON_PRONOSTICO_VENCIDO = (
    os.getenv("RELLENAR_HUECOS_CON_PRONOSTICO_VENCIDO", "false")
    .strip()
    .lower()
    in {"1", "true", "si", "sí", "yes"}
)

URL_ECMWF_ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODELO_ECMWF_ENS = "ecmwf_ifs025"
TIMEOUT_SEGUNDOS = 90
REINTENTOS = 4

COLUMNAS_COMPLETAS = [
    "Fecha",
    "TMAX",
    "TMIN",
    "Prec",
    "TMEDIA",
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


# ===============================================================
# UTILIDADES
# ===============================================================

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


# ===============================================================
# SIGA–INTA: DESCARGA Y PARSEO DE OBSERVACIONES
# ===============================================================

def buscar_archivo_siga_local(archivo_preferido: Path | None = None) -> Path | None:
    candidatos: list[Path] = []
    if archivo_preferido is not None:
        candidatos.append(archivo_preferido)
    candidatos.append(SIGA_ARCHIVO_LOCAL)
    for patron in ("A*.xls", "A*.xlsx", "*siga*.xls", "*siga*.xlsx", "*siga*.csv"):
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
    params = reemplazar_marcadores(parsear_json_entorno(SIGA_PARAMS_JSON, "SIGA_PARAMS_JSON"), contexto)
    data = reemplazar_marcadores(parsear_json_entorno(SIGA_POST_DATA_JSON, "SIGA_POST_DATA_JSON"), contexto)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0 Safari/537.36"
        ),
        "Accept": (
            "application/vnd.ms-excel,"
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
            "text/csv,*/*"
        ),
        "Referer": "https://siga.inta.gob.ar/",
    }
    headers.update(reemplazar_marcadores(parsear_json_entorno(SIGA_HEADERS_JSON, "SIGA_HEADERS_JSON"), contexto))

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
    nombre = coincidencia.group(1) if coincidencia else Path(url).name or "siga_bordenave_descarga.xls"

    inicio_texto = contenido[:300].lower()
    if b"<html" in inicio_texto or b"<!doctype html" in inicio_texto:
        raise ValueError("SIGA devolvió una página HTML y no un archivo de datos.")

    return contenido, nombre, tipo


def leer_tabla_siga_desde_bytes(contenido: bytes, nombre: str, tipo_contenido: str = "") -> pd.DataFrame:
    buffer = io.BytesIO(contenido)
    nombre_lower = nombre.lower()
    es_xls = contenido.startswith(b"\xd0\xcf\x11\xe0") or nombre_lower.endswith(".xls") or "application/vnd.ms-excel" in tipo_contenido
    es_xlsx = contenido.startswith(b"PK") or nombre_lower.endswith(".xlsx") or "spreadsheetml" in tipo_contenido
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


def normalizar_dataframe_siga(tabla: pd.DataFrame, fecha_limite_exclusiva: date) -> pd.DataFrame:
    if tabla.empty:
        raise ValueError("La tabla SIGA está vacía.")

    tabla = tabla.copy()
    tabla.columns = [normalizar_nombre_columna(c) for c in tabla.columns]

    # Alias explícitos para el XLS real de SIGA Bordenave:
    # temperatura_abrigo_150cm_maxima / minima / media y precipitacion_pluviometrica.
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

    seleccion = {destino: seleccionar_columna(tabla, candidatos) for destino, candidatos in alias.items()}
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
    fechas = pd.to_datetime(fechas_crudas, errors="coerce", yearfirst=True)
    faltantes_fecha = fechas.isna()
    if faltantes_fecha.any():
        fechas.loc[faltantes_fecha] = pd.to_datetime(
            fechas_crudas.loc[faltantes_fecha], errors="coerce", dayfirst=True
        )

    salida = pd.DataFrame({
        "Fecha": fechas,
        "TMAX": tabla[seleccion["tmax"]].map(to_float),
        "TMIN": tabla[seleccion["tmin"]].map(to_float),
        "Prec": tabla[seleccion["prec"]].map(to_float),
    })

    if seleccion.get("tmedia") is not None:
        salida["TMEDIA"] = tabla[seleccion["tmedia"]].map(to_float)
    else:
        salida["TMEDIA"] = (salida["TMAX"] + salida["TMIN"]) / 2.0

    salida = salida.dropna(subset=["Fecha", "TMAX", "TMIN"])
    salida["Fecha"] = salida["Fecha"].dt.normalize()

    salida.loc[~salida["TMAX"].between(-25, 55), "TMAX"] = np.nan
    salida.loc[~salida["TMIN"].between(-35, 45), "TMIN"] = np.nan
    salida.loc[salida["Prec"] < 0, "Prec"] = np.nan
    salida.loc[salida["Prec"] > 500, "Prec"] = np.nan
    salida = salida.loc[salida["TMAX"] >= salida["TMIN"]].copy()

    salida = salida.loc[
        (salida["Fecha"].dt.date >= CAMPANIA_START)
        & (salida["Fecha"].dt.date < fecha_limite_exclusiva)
    ].copy()

    salida["Fecha"] = salida["Fecha"].dt.strftime("%Y-%m-%d")
    salida = salida.drop_duplicates(subset=["Fecha"], keep="last")
    salida = salida.sort_values("Fecha").reset_index(drop=True)

    salida["GD_Tb2"] = np.maximum(0.0, salida["TMEDIA"] - TBASE)
    salida["Fuente"] = "SIGA_INTA_BORDENAVE"
    salida["TipoDato"] = "Observado"
    salida["CalidadDato"] = "Observado_estacion"
    salida["N_miembros"] = np.nan
    salida["Latitud_grilla"] = np.nan
    salida["Longitud_grilla"] = np.nan
    salida["Elevacion_grilla_m"] = np.nan
    salida["Emision_UTC"] = fecha_utc_iso()

    return asegurar_columnas(salida)


def obtener_siga_dataframe(
    fecha_inicio: date,
    fecha_fin: date,
    archivo_forzado: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Intenta: 1) SIGA remoto, 2) archivo local, 3) cache observado."""
    errores: list[str] = []

    if SIGA_URL_TEMPLATE and archivo_forzado is None:
        try:
            print("📡 Descargando observaciones diarias SIGA...")
            contenido, nombre, tipo = descargar_siga(fecha_inicio, fecha_fin)
            tabla = leer_tabla_siga_desde_bytes(contenido, nombre=nombre, tipo_contenido=tipo)
            df = normalizar_dataframe_siga(tabla, fecha_limite_exclusiva=fecha_fin + timedelta(days=1))
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
            df = normalizar_dataframe_siga(tabla, fecha_limite_exclusiva=fecha_fin + timedelta(days=1))
            escribir_csv_atomico(df, ARCHIVO_SIGA_CACHE)
            return df, f"SIGA_local:{archivo_local.name}"
        except Exception as error:
            errores.append(f"SIGA local: {error}")
            print(f"⚠️ Falló el archivo SIGA local: {error}")

    if ARCHIVO_SIGA_CACHE.exists():
        try:
            print("📦 Utilizando caché observado de SIGA.")
            cache = pd.read_csv(ARCHIVO_SIGA_CACHE, parse_dates=["Fecha"])
            cache = asegurar_columnas(cache)
            cache = cache.loc[cache["Fecha"].dt.date < fecha_fin + timedelta(days=1)].copy()
            cache["Fecha"] = cache["Fecha"].dt.strftime("%Y-%m-%d")
            return cache, "SIGA_cache"
        except Exception as error:
            errores.append(f"Caché SIGA: {error}")

    raise RuntimeError("No fue posible obtener datos SIGA. " + " | ".join(errores))


# ===============================================================
# OPEN-METEO ECMWF IFS ENS: PRONÓSTICO
# ===============================================================

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
        "wind_speed_unit": "kmh",
        "timeformat": "iso8601",
        "cell_selection": "land",
    }
    respuesta = solicitar_con_reintentos("GET", URL_ECMWF_ENS, params=params)
    return respuesta.json()


def seleccionar_columnas_ensamble(hourly: dict[str, Any], variable_base: str) -> list[str]:
    patron = re.compile(rf"^{re.escape(variable_base)}(_member\d+)?$")
    columnas = [
        clave
        for clave, valor in hourly.items()
        if clave != "time" and patron.match(clave) and isinstance(valor, list)
    ]

    def orden(clave: str) -> tuple[int, int]:
        if clave == variable_base:
            return (0, 0)
        m = re.search(r"_member(\d+)$", clave)
        return (1, int(m.group(1)) if m else 999)

    return sorted(columnas, key=orden)


def procesar_ecmwf_ens(datos: dict[str, Any]) -> pd.DataFrame:
    hourly = datos.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("La respuesta de Open-Meteo no contiene datos horarios.")

    tiempos = pd.to_datetime(hourly["time"], errors="coerce")
    if tiempos.isna().all():
        raise ValueError("No se pudieron interpretar las fechas del pronóstico.")

    cols_temp = seleccionar_columnas_ensamble(hourly, "temperature_2m")
    cols_prec = seleccionar_columnas_ensamble(hourly, "precipitation")
    if not cols_temp:
        raise ValueError("No hay miembros de temperatura en ECMWF ENS.")
    if not cols_prec:
        raise ValueError("No hay miembros de precipitación en ECMWF ENS.")

    n_miembros = min(len(cols_temp), len(cols_prec))
    matriz_diaria: list[pd.DataFrame] = []

    for i in range(n_miembros):
        temp = pd.to_numeric(pd.Series(hourly[cols_temp[i]]), errors="coerce")
        prec = pd.to_numeric(pd.Series(hourly[cols_prec[i]]), errors="coerce").fillna(0.0)
        miembro = pd.DataFrame({
            "Fecha": tiempos.dt.date.astype(str),
            "Temp": temp,
            "Prec_h": prec,
        })
        diario = miembro.groupby("Fecha", as_index=False).agg(
            TMAX=("Temp", "max"),
            TMIN=("Temp", "min"),
            TMEDIA=("Temp", "mean"),
            Prec=("Prec_h", "sum"),
        )
        diario["miembro"] = i
        matriz_diaria.append(diario)

    todos = pd.concat(matriz_diaria, ignore_index=True)
    registros = []
    emision = fecha_utc_iso()
    lat_grid = datos.get("latitude", np.nan)
    lon_grid = datos.get("longitude", np.nan)
    elev_grid = datos.get("elevation", np.nan)

    for fecha, grupo in todos.groupby("Fecha"):
        tmax = grupo["TMAX"]
        tmin = grupo["TMIN"]
        tmedia = grupo["TMEDIA"]
        prec = grupo["Prec"]
        registros.append({
            "Fecha": fecha,
            "TMAX": tmax.mean(),
            "TMIN": tmin.mean(),
            "Prec": prec.mean(),
            "TMEDIA": tmedia.mean(),
            "TMAX_P10": tmax.quantile(0.10),
            "TMAX_P50": tmax.quantile(0.50),
            "TMAX_P90": tmax.quantile(0.90),
            "TMIN_P10": tmin.quantile(0.10),
            "TMIN_P50": tmin.quantile(0.50),
            "TMIN_P90": tmin.quantile(0.90),
            "TMEDIA_P10": tmedia.quantile(0.10),
            "TMEDIA_P50": tmedia.quantile(0.50),
            "TMEDIA_P90": tmedia.quantile(0.90),
            "Prec_P10": prec.quantile(0.10),
            "Prec_P50": prec.quantile(0.50),
            "Prec_P90": prec.quantile(0.90),
            "Prob_Prec_ge_1mm": float((prec >= 1.0).mean() * 100.0),
            "Prob_Prec_ge_5mm": float((prec >= 5.0).mean() * 100.0),
            "Prob_Prec_ge_10mm": float((prec >= 10.0).mean() * 100.0),
            "Prob_Prec_ge_30mm": float((prec >= 30.0).mean() * 100.0),
            "GD_Tb2": max(0.0, float(tmedia.mean()) - TBASE),
            "Fuente": "ECMWF_IFS_ENS_025",
            "TipoDato": "Pronostico",
            "CalidadDato": "Media_ensamble",
            "N_miembros": int(n_miembros),
            "Latitud_grilla": lat_grid,
            "Longitud_grilla": lon_grid,
            "Elevacion_grilla_m": elev_grid,
            "Emision_UTC": emision,
        })

    salida = pd.DataFrame(registros)
    salida = asegurar_columnas(salida)
    salida = salida.sort_values("Fecha").reset_index(drop=True)
    return salida


def cargar_pronostico_ecmwf() -> pd.DataFrame:
    datos = consultar_ecmwf_ens()
    pronostico = procesar_ecmwf_ens(datos)
    DIRECTORIO_PRONOSTICOS.mkdir(parents=True, exist_ok=True)
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archivo = DIRECTORIO_PRONOSTICOS / f"ecmwf_ifs_ens_025_{marca}.csv"
    escribir_csv_atomico(pronostico, archivo)
    return pronostico


# ===============================================================
# ENSAMBLE FINAL OBSERVADO + PRONÓSTICO
# ===============================================================

def leer_maestro_existente(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNAS_COMPLETAS)
    try:
        df = pd.read_csv(path)
        df = asegurar_columnas(df)
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["Fecha"])
        return df
    except Exception as error:
        print(f"⚠️ No se pudo leer maestro existente: {error}")
        return pd.DataFrame(columns=COLUMNAS_COMPLETAS)


def calcular_huecos_observados(observaciones: pd.DataFrame, hasta_exclusivo: date) -> list[str]:
    if hasta_exclusivo <= CAMPANIA_START:
        return []
    esperadas = pd.date_range(
        CAMPANIA_START,
        hasta_exclusivo - timedelta(days=1),
        freq="D",
    ).strftime("%Y-%m-%d")
    presentes = set(observaciones["Fecha"].astype(str)) if not observaciones.empty else set()
    return [fecha for fecha in esperadas if fecha not in presentes]


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

    pronostico = cargar_pronostico_ecmwf()
    pronostico = pronostico.loc[pd.to_datetime(pronostico["Fecha"]).dt.date >= hoy].copy()

    maestro_anterior = leer_maestro_existente(output)

    if RELLENAR_HUECOS_CON_PRONOSTICO_VENCIDO and not maestro_anterior.empty:
        vencidos = maestro_anterior.loc[
            (pd.to_datetime(maestro_anterior["Fecha"]).dt.date < hoy)
            & (maestro_anterior["TipoDato"].astype(str).str.lower() == "pronostico")
        ].copy()
        if not vencidos.empty:
            fechas_obs = set(observaciones["Fecha"].astype(str))
            vencidos = vencidos.loc[~vencidos["Fecha"].astype(str).isin(fechas_obs)].copy()
            if not vencidos.empty:
                vencidos["CalidadDato"] = "Pronostico_vencido_sin_SIGA"
                observaciones = pd.concat([observaciones, vencidos], ignore_index=True)

    combinado = pd.concat([observaciones, pronostico], ignore_index=True)
    combinado = asegurar_columnas(combinado)
    combinado["Fecha_dt"] = pd.to_datetime(combinado["Fecha"], errors="coerce")
    combinado = combinado.dropna(subset=["Fecha_dt"])

    prioridad = combinado["TipoDato"].map({"Observado": 0, "Pronostico": 1}).fillna(2)
    combinado["_prioridad"] = prioridad
    combinado = combinado.sort_values(["Fecha_dt", "_prioridad"])
    combinado = combinado.drop_duplicates(subset=["Fecha"], keep="first")
    combinado = combinado.sort_values("Fecha_dt").drop(columns=["Fecha_dt", "_prioridad"])
    combinado = asegurar_columnas(combinado)

    escribir_csv_atomico(combinado, output)

    huecos = calcular_huecos_observados(observaciones, hoy)
    if huecos:
        print("⚠️ Fechas vencidas sin observación SIGA: " + ", ".join(huecos[-30:]))

    estado = {
        "ejecucion_utc": fecha_utc_iso(),
        "estado_siga": estado_siga,
        "ultima_observacion_siga": str(observaciones["Fecha"].max()) if not observaciones.empty else None,
        "fuente_pronostico": "ECMWF_IFS_ENS_025",
        "inicio_pronostico": str(pronostico["Fecha"].min()) if not pronostico.empty else None,
        "fin_pronostico": str(pronostico["Fecha"].max()) if not pronostico.empty else None,
        "huecos_observados": huecos,
    }

    ARCHIVO_ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_ESTADO.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Archivo actualizado: {output}")
    print(f"✅ Observaciones SIGA: {len(observaciones)} filas ({estado_siga})")
    print(f"✅ Pronóstico ECMWF ENS: {len(pronostico)} filas")
    print(f"✅ Estado: {ARCHIVO_ESTADO}")
    return combinado


def validar_siga(siga_file: Path | None = None) -> None:
    hoy = hoy_argentina()
    ayer = hoy - timedelta(days=1)
    observaciones, estado_siga = obtener_siga_dataframe(CAMPANIA_START, ayer, archivo_forzado=siga_file)
    print(f"✅ SIGA válido: {estado_siga}")
    print(f"Filas: {len(observaciones)}")
    if not observaciones.empty:
        print(f"Rango: {observaciones['Fecha'].min()} a {observaciones['Fecha'].max()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Actualiza meteo_daily.csv para PREDWEEM Bordenave.")
    parser.add_argument("--output", default=str(ARCHIVO_MAESTRO_DEFAULT), help="Archivo de salida CSV.")
    parser.add_argument("--siga-file", default=None, help="Archivo SIGA local opcional XLS/XLSX/CSV.")
    parser.add_argument("--solo-validar-siga", action="store_true", help="Solo valida SIGA y actualiza cache observado.")
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
        print(f"❌ Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
