# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 NODO CLIMÁTICO PREDWEEM — BORDENAVE
#
# Serie operativa:
#   • Fechas vencidas (< hoy): observaciones diarias SIGA–INTA.
#   • Hoy y próximos 6 días: ECMWF IFS ENS.
#   • Respaldo del pronóstico: MeteoBahía.
#
# Archivo final compatible con PREDWEEM:
#   meteo_daily.csv
#
# Columnas principales conservadas:
#   Fecha, TMAX, TMIN, Prec
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
import xml.etree.ElementTree as ET
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

ARCHIVO_MAESTRO = Path("meteo_daily.csv")
ARCHIVO_SIGA_CACHE = Path("data/siga_bordenave_observado.csv")
DIRECTORIO_PRONOSTICOS = Path("data/historico_pronosticos")
ARCHIVO_ESTADO = Path("data/estado_actualizacion_meteo.json")

# Respaldo local si SIGA no responde o aún no se configuró la URL.
SIGA_ARCHIVO_LOCAL = Path(
    os.getenv("SIGA_LOCAL_FILE", "A872817.xls")
)

# SIGA no publica una API estable documentada.
# Configure aquí la URL directa que genera la descarga XLS/CSV.
#
# La URL puede incluir:
#   {start_date}  -> 2026-01-01
#   {end_date}    -> ayer, por ejemplo 2026-07-06
#
# Es preferible guardar la URL como secreto de GitHub:
#   SIGA_DOWNLOAD_URL
SIGA_URL_TEMPLATE = os.getenv(
    "SIGA_DOWNLOAD_URL",
    "",
).strip()

# Se admiten consultas GET o POST para adaptarse al pedido real del portal.
SIGA_METHOD = os.getenv("SIGA_METHOD", "GET").strip().upper()
SIGA_PARAMS_JSON = os.getenv("SIGA_PARAMS_JSON", "").strip()
SIGA_POST_DATA_JSON = os.getenv("SIGA_POST_DATA_JSON", "").strip()
SIGA_HEADERS_JSON = os.getenv("SIGA_HEADERS_JSON", "").strip()

# Si SIGA no tiene una fecha vencida, por defecto NO se conserva el antiguo
# pronóstico en la serie operativa. Esto evita confundir estimaciones con datos.
RELLENAR_HUECOS_CON_PRONOSTICO_VENCIDO = (
    os.getenv(
        "RELLENAR_HUECOS_CON_PRONOSTICO_VENCIDO",
        "false",
    ).strip().lower()
    in {"1", "true", "si", "sí", "yes"}
)

URL_ECMWF_ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODELO_ECMWF_ENS = "ecmwf_ifs025"

URL_METEOBAHIA = (
    "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
)

TIMEOUT_SEGUNDOS = 90
REINTENTOS = 4

PATRON_MIEMBRO = re.compile(
    r"^(?P<variable>[a-z0-9_]+?)"
    r"(?:_member(?P<miembro>\d+))?$"
)

COLUMNAS_PRINCIPALES = [
    "Fecha",
    "TMAX",
    "TMIN",
    "Prec",
]

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
    """Devuelve la fecha calendario en Argentina."""
    return datetime.now(
        ZoneInfo(ZONA_HORARIA)
    ).date()


def fecha_utc_iso() -> str:
    """Marca temporal UTC sin microsegundos."""
    return datetime.now(
        timezone.utc
    ).replace(microsecond=0).isoformat()


def to_float(valor: Any) -> float | None:
    """Convierte números con punto o coma decimal."""
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


def normalizar_nombre_columna(nombre: Any) -> str:
    """Normaliza encabezados para detectar variantes de SIGA."""
    texto = str(nombre).strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    return texto.strip("_")


def parsear_json_entorno(
    texto: str,
    nombre: str,
) -> dict[str, Any]:
    """Interpreta una variable de entorno JSON."""
    if not texto:
        return {}

    try:
        valor = json.loads(texto)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{nombre} no contiene JSON válido: {error}"
        ) from error

    if not isinstance(valor, dict):
        raise ValueError(
            f"{nombre} debe contener un objeto JSON."
        )

    return valor


def reemplazar_marcadores(
    valor: Any,
    contexto: dict[str, str],
) -> Any:
    """
    Reemplaza {start_date} y {end_date} de forma recursiva.
    """
    if isinstance(valor, str):
        return valor.format_map(contexto)

    if isinstance(valor, dict):
        return {
            clave: reemplazar_marcadores(contenido, contexto)
            for clave, contenido in valor.items()
        }

    if isinstance(valor, list):
        return [
            reemplazar_marcadores(contenido, contexto)
            for contenido in valor
        ]

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
    """Realiza una consulta HTTP con reintentos."""
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
            print(
                f"⚠️ Intento HTTP {intento}/{intentos} "
                f"fallido: {error}"
            )

            if intento < intentos:
                time.sleep(5 * intento)

    raise RuntimeError(
        f"No fue posible consultar {url}"
    ) from ultimo_error


def asegurar_columnas(
    df: pd.DataFrame,
    columnas: list[str] = COLUMNAS_COMPLETAS,
) -> pd.DataFrame:
    """Agrega las columnas ausentes con valores nulos."""
    salida = df.copy()

    for columna in columnas:
        if columna not in salida.columns:
            salida[columna] = np.nan

    return salida[columnas]


def escribir_csv_atomico(
    df: pd.DataFrame,
    destino: Path,
) -> None:
    """Escribe un CSV sin exponer un archivo parcial."""
    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal = destino.with_suffix(
        destino.suffix + ".tmp"
    )

    df.to_csv(
        temporal,
        index=False,
        float_format="%.3f",
    )

    if destino.exists():
        respaldo = destino.with_suffix(
            destino.suffix + ".bak"
        )
        shutil.copy2(destino, respaldo)

    temporal.replace(destino)


def ultimo_no_nulo(serie: pd.Series) -> Any:
    """Devuelve el último valor no nulo de una serie."""
    valores = serie.dropna()
    return valores.iloc[-1] if not valores.empty else np.nan


# ===============================================================
# SIGA–INTA: DESCARGA Y PARSEO DE OBSERVACIONES
# ===============================================================

def buscar_archivo_siga_local(
    archivo_preferido: Path | None = None,
) -> Path | None:
    """Encuentra un XLS/XLSX/CSV local de SIGA."""
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
        candidatos.extend(
            Path(".").glob(patron)
        )

    existentes = {
        candidato.resolve()
        for candidato in candidatos
        if candidato.exists()
    }

    if not existentes:
        return None

    return max(
        existentes,
        key=lambda ruta: ruta.stat().st_mtime,
    )


def descargar_siga(
    fecha_inicio: date,
    fecha_fin: date,
) -> tuple[bytes, str, str]:
    """
    Descarga el archivo exportado por SIGA.

    Retorna:
        contenido, nombre sugerido, tipo de contenido.
    """
    if not SIGA_URL_TEMPLATE:
        raise RuntimeError(
            "SIGA_DOWNLOAD_URL no está configurada."
        )

    contexto = {
        "start": fecha_inicio.isoformat(),
        "end": fecha_fin.isoformat(),
        "start_date": fecha_inicio.isoformat(),
        "end_date": fecha_fin.isoformat(),
    }

    url = reemplazar_marcadores(
        SIGA_URL_TEMPLATE,
        contexto,
    )

    params = reemplazar_marcadores(
        parsear_json_entorno(
            SIGA_PARAMS_JSON,
            "SIGA_PARAMS_JSON",
        ),
        contexto,
    )

    data = reemplazar_marcadores(
        parsear_json_entorno(
            SIGA_POST_DATA_JSON,
            "SIGA_POST_DATA_JSON",
        ),
        contexto,
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/138.0 Safari/537.36"
        ),
        "Accept": (
            "application/vnd.ms-excel,"
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet,text/csv,*/*"
        ),
    }

    headers.update(
        reemplazar_marcadores(
            parsear_json_entorno(
                SIGA_HEADERS_JSON,
                "SIGA_HEADERS_JSON",
            ),
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
        raise ValueError(
            "La descarga SIGA es demasiado pequeña "
            "para contener una tabla."
        )

    tipo = respuesta.headers.get(
        "content-type",
        "",
    ).lower()

    disposicion = respuesta.headers.get(
        "content-disposition",
        "",
    )

    coincidencia = re.search(
        r'filename="?([^";]+)',
        disposicion,
        flags=re.IGNORECASE,
    )

    nombre = (
        coincidencia.group(1)
        if coincidencia
        else "siga_bordenave_descarga.xls"
    )

    inicio_texto = contenido[:300].lower()
    if (
        b"<html" in inicio_texto
        or b"<!doctype html" in inicio_texto
    ):
        raise ValueError(
            "SIGA devolvió una página HTML y no un archivo "
            "de datos. Revise la URL, cookies o encabezados."
        )

    return contenido, nombre, tipo


def leer_tabla_siga_desde_bytes(
    contenido: bytes,
    nombre: str = "siga.xls",
    tipo_contenido: str = "",
) -> pd.DataFrame:
    """Lee XLS, XLSX o CSV descargado desde SIGA."""
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
        return pd.read_excel(
            buffer,
            sheet_name="Datos diarios",
            engine="xlrd",
        )

    if es_xlsx:
        return pd.read_excel(
            buffer,
            sheet_name="Datos diarios",
            engine="openpyxl",
        )

    texto = contenido.decode(
        "utf-8-sig",
        errors="replace",
    )

    for separador in (";", ",", "\t"):
        candidato = pd.read_csv(
            io.StringIO(texto),
            sep=separador,
        )
        if candidato.shape[1] >= 4:
            return candidato

    raise ValueError(
        "No se pudo reconocer el formato de la descarga SIGA."
    )


def leer_tabla_siga_local(
    archivo: Path,
) -> pd.DataFrame:
    """Lee una exportación local de SIGA."""
    sufijo = archivo.suffix.lower()

    if sufijo == ".xls":
        return pd.read_excel(
            archivo,
            sheet_name="Datos diarios",
            engine="xlrd",
        )

    if sufijo == ".xlsx":
        return pd.read_excel(
            archivo,
            sheet_name="Datos diarios",
            engine="openpyxl",
        )

    if sufijo == ".csv":
        for separador in (";", ",", "\t"):
            candidato = pd.read_csv(
                archivo,
                sep=separador,
            )
            if candidato.shape[1] >= 4:
                return candidato

    raise ValueError(
        f"Formato SIGA local no soportado: {archivo}"
    )


def normalizar_dataframe_siga(
    tabla: pd.DataFrame,
    fecha_limite_exclusiva: date,
) -> pd.DataFrame:
    """Convierte la exportación SIGA al formato PREDWEEM."""
    if tabla.empty:
        raise ValueError(
            "La tabla SIGA está vacía."
        )

    tabla = tabla.copy()
    tabla.columns = [
        normalizar_nombre_columna(columna)
        for columna in tabla.columns
    ]

    alias = {
        "fecha": [
            "fecha",
            "date",
        ],
        "tmedia": [
            "temperatura_media",
            "temperatura_promedio",
            "tmedia",
            "temp_media",
        ],
        "tmax": [
            "temperatura_maxima",
            "temperatura_max",
            "tmax",
            "temp_max",
        ],
        "tmin": [
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

    seleccion: dict[str, str] = {}

    for destino, candidatos in alias.items():
        for candidato in candidatos:
            if candidato in tabla.columns:
                seleccion[destino] = candidato
                break

    obligatorias = {"fecha", "tmax", "tmin", "prec"}
    faltantes = obligatorias - set(seleccion)

    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias en SIGA: "
            + ", ".join(sorted(faltantes))
            + ". Columnas encontradas: "
            + ", ".join(tabla.columns)
        )

    fechas_crudas = tabla[seleccion["fecha"]]

    # SIGA suele exportar ISO YYYY-MM-DD. Se analiza primero con
    # prioridad de año para evitar intercambiar día y mes. Solo los
    # valores que no pudieron interpretarse se reintentan con dayfirst.
    fechas = pd.to_datetime(
        fechas_crudas,
        errors="coerce",
        yearfirst=True,
    )

    faltantes_fecha = fechas.isna()
    if faltantes_fecha.any():
        fechas.loc[faltantes_fecha] = pd.to_datetime(
            fechas_crudas.loc[faltantes_fecha],
            errors="coerce",
            dayfirst=True,
        )

    salida = pd.DataFrame({
        "Fecha": fechas,
        "TMAX": tabla[seleccion["tmax"]].map(to_float),
        "TMIN": tabla[seleccion["tmin"]].map(to_float),
        "Prec": tabla[seleccion["prec"]].map(to_float),
    })

    if "tmedia" in seleccion:
        salida["TMEDIA"] = tabla[
            seleccion["tmedia"]
        ].map(to_float)
    else:
        salida["TMEDIA"] = (
            salida["TMAX"] + salida["TMIN"]
        ) / 2.0

    salida = salida.dropna(
        subset=["Fecha", "TMAX", "TMIN"]
    )

    salida["Fecha"] = salida["Fecha"].dt.normalize()

    # Controles físicos básicos.
    salida.loc[
        ~salida["TMAX"].between(-25, 55),
        "TMAX",
    ] = np.nan

    salida.loc[
        ~salida["TMIN"].between(-35, 45),
        "TMIN",
    ] = np.nan

    salida.loc[
        salida["Prec"] < 0,
        "Prec",
    ] = np.nan

    salida.loc[
        salida["Prec"] > 500,
        "Prec",
    ] = np.nan

    salida = salida.loc[
        salida["TMAX"] >= salida["TMIN"]
    ].copy()

    salida = salida.loc[
        (salida["Fecha"].dt.date >= CAMPANIA_START)
        & (
            salida["Fecha"].dt.date
            < fecha_limite_exclusiva
        )
    ].copy()

    salida["Fuente"] = "SIGA_INTA_BORDENAVE"
    salida["TipoDato"] = "Observado"
    salida["CalidadDato"] = "Observado_estacion"
    salida["Emision_UTC"] = fecha_utc_iso()
    salida["GD_Tb2"] = np.maximum(
        0.0,
        salida["TMEDIA"] - TBASE,
    )

    salida = (
        salida.drop_duplicates(
            subset=["Fecha"],
            keep="last",
        )
        .sort_values("Fecha")
        .reset_index(drop=True)
    )

    return asegurar_columnas(salida)


def obtener_siga_dataframe(
    fecha_inicio: date,
    fecha_fin: date,
    archivo_forzado: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Intenta:
      1. URL directa SIGA.
      2. Archivo local.
      3. Caché observado anterior.
    """
    errores: list[str] = []

    if SIGA_URL_TEMPLATE and archivo_forzado is None:
        try:
            print(
                "📡 Descargando observaciones diarias SIGA..."
            )

            contenido, nombre, tipo = descargar_siga(
                fecha_inicio,
                fecha_fin,
            )

            tabla = leer_tabla_siga_desde_bytes(
                contenido,
                nombre=nombre,
                tipo_contenido=tipo,
            )

            df = normalizar_dataframe_siga(
                tabla,
                fecha_limite_exclusiva=fecha_fin
                + timedelta(days=1),
            )

            return df, "SIGA_remoto"

        except Exception as error:
            errores.append(
                f"SIGA remoto: {error}"
            )
            print(
                f"⚠️ Falló la consulta remota SIGA: {error}"
            )

    archivo_local = buscar_archivo_siga_local(
        archivo_forzado
    )

    if archivo_local is not None:
        try:
            print(
                f"📄 Leyendo respaldo SIGA local: "
                f"{archivo_local}"
            )

            tabla = leer_tabla_siga_local(
                archivo_local
            )

            df = normalizar_dataframe_siga(
                tabla,
                fecha_limite_exclusiva=fecha_fin
                + timedelta(days=1),
            )

            return df, f"SIGA_local:{archivo_local.name}"

        except Exception as error:
            errores.append(
                f"SIGA local: {error}"
            )
            print(
                f"⚠️ Falló el archivo SIGA local: {error}"
            )

    if ARCHIVO_SIGA_CACHE.exists():
        try:
            print(
                "📦 Utilizando caché observado de SIGA."
            )

            cache = pd.read_csv(
                ARCHIVO_SIGA_CACHE,
                parse_dates=["Fecha"],
            )

            cache = asegurar_columnas(cache)
            cache = cache.loc[
                cache["Fecha"].dt.date
                < fecha_fin + timedelta(days=1)
            ].copy()

            return cache, "SIGA_cache"

        except Exception as error:
            errores.append(
                f"Caché SIGA: {error}"
            )

    raise RuntimeError(
        "No fue posible obtener datos SIGA. "
        + " | ".join(errores)
    )


# ===============================================================
# ECMWF IFS ENS: PRONÓSTICO PRINCIPAL
# ===============================================================

def solicitar_json_ecmwf(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Descarga y valida la respuesta ECMWF ENS."""
    respuesta = solicitar_con_reintentos(
        "GET",
        URL_ECMWF_ENS,
        params=params,
        headers={
            "User-Agent": "PREDWEEM/2026",
        },
    )

    try:
        datos = respuesta.json()
    except ValueError as error:
        raise ValueError(
            "ECMWF ENS no devolvió JSON válido."
        ) from error

    if datos.get("error"):
        raise RuntimeError(
            str(datos.get("reason", datos["error"]))
        )

    if "hourly" not in datos:
        raise ValueError(
            "ECMWF ENS no devolvió datos horarios."
        )

    return datos


def extraer_miembros(
    hourly: dict[str, Any],
    variable: str,
) -> dict[str, np.ndarray]:
    """Extrae todos los miembros disponibles de una variable."""
    miembros: dict[str, np.ndarray] = {}

    for clave, valores in hourly.items():
        if clave == "time":
            continue

        coincidencia = PATRON_MIEMBRO.match(clave)

        if not coincidencia:
            continue

        if coincidencia.group("variable") != variable:
            continue

        numero = coincidencia.group("miembro")

        nombre = (
            "control"
            if numero is None
            else f"member{int(numero):02d}"
        )

        miembros[nombre] = np.asarray(
            valores,
            dtype=float,
        )

    if not miembros:
        raise KeyError(
            f"No se encontraron miembros para {variable}."
        )

    return miembros


def fetch_ecmwf_ens_dataframe() -> pd.DataFrame:
    """Genera siete días diarios a partir de los miembros ECMWF."""
    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "hourly": "temperature_2m,precipitation",
        "models": MODELO_ECMWF_ENS,
        "timezone": ZONA_HORARIA,
        "forecast_days": HORIZONTE_DIAS,
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }

    datos = solicitar_json_ecmwf(params)
    hourly = datos["hourly"]

    if "time" not in hourly:
        raise ValueError(
            "ECMWF ENS no contiene el eje temporal."
        )

    tiempos = pd.to_datetime(
        hourly["time"],
        errors="raise",
    )

    miembros_t = extraer_miembros(
        hourly,
        "temperature_2m",
    )
    miembros_p = extraer_miembros(
        hourly,
        "precipitation",
    )

    miembros = sorted(
        set(miembros_t) & set(miembros_p)
    )

    if not miembros:
        raise RuntimeError(
            "No existen miembros comunes de temperatura "
            "y precipitación."
        )

    base = pd.DataFrame({
        "FechaHora": tiempos,
    })
    base["Fecha"] = base["FechaHora"].dt.date

    fechas_referencia: list[date] | None = None
    tmax_lista: list[np.ndarray] = []
    tmin_lista: list[np.ndarray] = []
    tmedia_lista: list[np.ndarray] = []
    prec_lista: list[np.ndarray] = []

    for miembro in miembros:
        if (
            len(miembros_t[miembro]) != len(base)
            or len(miembros_p[miembro]) != len(base)
        ):
            raise ValueError(
                f"Longitud inconsistente para {miembro}."
            )

        df = base.copy()
        df["T"] = miembros_t[miembro]
        df["P"] = np.maximum(
            miembros_p[miembro],
            0.0,
        )

        diario = (
            df.groupby(
                "Fecha",
                as_index=False,
            )
            .agg(
                TMAX=("T", "max"),
                TMIN=("T", "min"),
                Prec=("P", "sum"),
                Horas=("T", "count"),
            )
        )

        diario = diario.loc[
            diario["Horas"] >= 20
        ].reset_index(drop=True)

        diario["TMEDIA"] = (
            diario["TMAX"] + diario["TMIN"]
        ) / 2.0

        fechas_miembro = diario[
            "Fecha"
        ].tolist()

        if fechas_referencia is None:
            fechas_referencia = fechas_miembro
        elif fechas_miembro != fechas_referencia:
            raise RuntimeError(
                f"Las fechas del miembro {miembro} "
                "no coinciden."
            )

        tmax_lista.append(
            diario["TMAX"].to_numpy(float)
        )
        tmin_lista.append(
            diario["TMIN"].to_numpy(float)
        )
        tmedia_lista.append(
            diario["TMEDIA"].to_numpy(float)
        )
        prec_lista.append(
            diario["Prec"].to_numpy(float)
        )

    if fechas_referencia is None:
        raise RuntimeError(
            "No se pudieron construir días ECMWF."
        )

    tmax = np.column_stack(tmax_lista)
    tmin = np.column_stack(tmin_lista)
    tmedia = np.column_stack(tmedia_lista)
    prec = np.column_stack(prec_lista)

    salida = pd.DataFrame({
        "Fecha": pd.to_datetime(
            fechas_referencia
        ),
    })

    matrices = {
        "TMAX": tmax,
        "TMIN": tmin,
        "TMEDIA": tmedia,
        "Prec": prec,
    }

    for nombre, matriz in matrices.items():
        salida[nombre] = np.nanmean(
            matriz,
            axis=1,
        )
        salida[f"{nombre}_P10"] = np.nanpercentile(
            matriz,
            10,
            axis=1,
        )
        salida[f"{nombre}_P50"] = np.nanpercentile(
            matriz,
            50,
            axis=1,
        )
        salida[f"{nombre}_P90"] = np.nanpercentile(
            matriz,
            90,
            axis=1,
        )

    for umbral in (1, 5, 10, 30):
        salida[
            f"Prob_Prec_ge_{umbral}mm"
        ] = (
            np.nanmean(
                prec >= umbral,
                axis=1,
            )
            * 100.0
        )

    salida["GD_Tb2"] = np.maximum(
        0.0,
        salida["TMEDIA"] - TBASE,
    )

    salida["Fuente"] = "ECMWF_IFS_ENS_025"
    salida["TipoDato"] = "Pronostico"
    salida["CalidadDato"] = "Media_ensamble"
    salida["N_miembros"] = len(miembros)
    salida["Latitud_grilla"] = datos.get(
        "latitude"
    )
    salida["Longitud_grilla"] = datos.get(
        "longitude"
    )
    salida["Elevacion_grilla_m"] = datos.get(
        "elevation"
    )
    salida["Emision_UTC"] = fecha_utc_iso()

    hoy = pd.Timestamp(hoy_argentina())
    limite = hoy + pd.Timedelta(
        days=HORIZONTE_DIAS - 1
    )

    salida = salida.loc[
        (salida["Fecha"] >= hoy)
        & (salida["Fecha"] <= limite)
    ].copy()

    if salida.empty:
        raise ValueError(
            "ECMWF ENS no produjo días dentro "
            "del horizonte solicitado."
        )

    return asegurar_columnas(
        salida.sort_values("Fecha")
        .reset_index(drop=True)
    )


# ===============================================================
# METEOBAHÍA: RESPALDO DE PRONÓSTICO
# ===============================================================

def fetch_meteobahia_dataframe() -> pd.DataFrame:
    """Descarga el pronóstico XML de MeteoBahía."""
    respuesta = solicitar_con_reintentos(
        "GET",
        URL_METEOBAHIA,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0 Safari/537.36"
            )
        },
        timeout=30,
    )

    root = ET.fromstring(
        respuesta.content
    )

    filas: list[dict[str, Any]] = []

    for elemento in root.findall(
        ".//forecast/tabular/day"
    ):
        fecha = elemento.find("fecha")
        tmax = elemento.find("tmax")
        tmin = elemento.find("tmin")
        precip = elemento.find("precip")

        if None in (
            fecha,
            tmax,
            tmin,
            precip,
        ):
            continue

        filas.append({
            "Fecha": pd.to_datetime(
                fecha.get("value"),
                errors="coerce",
            ),
            "TMAX": to_float(
                tmax.get("value")
            ),
            "TMIN": to_float(
                tmin.get("value")
            ),
            "Prec": to_float(
                precip.get("value")
            ),
        })

    if not filas:
        raise ValueError(
            "MeteoBahía no devolvió registros."
        )

    salida = pd.DataFrame(filas)
    salida = salida.dropna(
        subset=["Fecha", "TMAX", "TMIN"]
    )

    salida["TMEDIA"] = (
        salida["TMAX"] + salida["TMIN"]
    ) / 2.0

    salida["GD_Tb2"] = np.maximum(
        0.0,
        salida["TMEDIA"] - TBASE,
    )

    salida["Fuente"] = "MeteoBahia_respaldo"
    salida["TipoDato"] = "Pronostico"
    salida["CalidadDato"] = "Pronostico_deterministico"
    salida["Emision_UTC"] = fecha_utc_iso()

    hoy = pd.Timestamp(hoy_argentina())
    limite = hoy + pd.Timedelta(
        days=HORIZONTE_DIAS - 1
    )

    salida = salida.loc[
        (salida["Fecha"] >= hoy)
        & (salida["Fecha"] <= limite)
    ].copy()

    return asegurar_columnas(
        salida.sort_values("Fecha")
        .reset_index(drop=True)
    )


def obtener_pronostico() -> tuple[pd.DataFrame, str]:
    """Usa ECMWF ENS y activa MeteoBahía si falla."""
    try:
        print(
            "📡 Descargando pronóstico ECMWF IFS ENS..."
        )

        df = fetch_ecmwf_ens_dataframe()

        miembros = df["N_miembros"].dropna()
        if not miembros.empty:
            print(
                f"✅ ECMWF ENS: "
                f"{int(miembros.iloc[0])} miembros."
            )

        return df, "ECMWF_IFS_ENS_025"

    except Exception as error_ecmwf:
        print(
            f"⚠️ ECMWF ENS falló: {error_ecmwf}"
        )
        print(
            "📡 Activando respaldo MeteoBahía..."
        )

        try:
            df = fetch_meteobahia_dataframe()
            return df, "MeteoBahia_respaldo"

        except Exception as error_mb:
            raise RuntimeError(
                "Fallaron ambas fuentes de pronóstico. "
                f"ECMWF: {error_ecmwf}. "
                f"MeteoBahía: {error_mb}"
            ) from error_mb


# ===============================================================
# ARCHIVOS, FUSIÓN Y CONTROL DE CALIDAD
# ===============================================================

def cargar_archivo_maestro(
    archivo: Path,
) -> pd.DataFrame:
    """Carga el archivo maestro anterior."""
    if not archivo.exists():
        return pd.DataFrame(
            columns=COLUMNAS_COMPLETAS
        )

    df = pd.read_csv(archivo)

    if "Fecha" not in df.columns:
        raise ValueError(
            f"{archivo} no contiene la columna Fecha."
        )

    df["Fecha"] = pd.to_datetime(
        df["Fecha"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Fecha"]
    )

    return asegurar_columnas(df)


def consolidar_observaciones(
    observaciones_nuevas: pd.DataFrame,
    maestro_anterior: pd.DataFrame,
    hoy: date,
) -> pd.DataFrame:
    """Conserva y actualiza exclusivamente observaciones reales."""
    bloques: list[pd.DataFrame] = []

    if not maestro_anterior.empty:
        fuente = maestro_anterior[
            "Fuente"
        ].fillna("").astype(str)

        tipo = maestro_anterior[
            "TipoDato"
        ].fillna("").astype(str)

        observadas_previas = maestro_anterior.loc[
            tipo.str.lower().eq("observado")
            | fuente.str.contains(
                "SIGA",
                case=False,
                regex=False,
            )
        ].copy()

        observadas_previas["_prioridad"] = 1
        bloques.append(observadas_previas)

    if ARCHIVO_SIGA_CACHE.exists():
        try:
            cache = pd.read_csv(
                ARCHIVO_SIGA_CACHE,
                parse_dates=["Fecha"],
            )
            cache = asegurar_columnas(cache)
            cache["_prioridad"] = 2
            bloques.append(cache)
        except Exception as error:
            print(
                f"⚠️ No se pudo leer la caché SIGA: {error}"
            )

    nuevas = asegurar_columnas(
        observaciones_nuevas
    )
    nuevas["_prioridad"] = 3
    bloques.append(nuevas)

    combinado = pd.concat(
        bloques,
        ignore_index=True,
        sort=False,
    )

    combinado["Fecha"] = pd.to_datetime(
        combinado["Fecha"],
        errors="coerce",
    )

    combinado = combinado.dropna(
        subset=["Fecha"]
    )

    combinado = combinado.loc[
        (combinado["Fecha"].dt.date >= CAMPANIA_START)
        & (combinado["Fecha"].dt.date < hoy)
    ].copy()

    combinado = combinado.sort_values(
        ["Fecha", "_prioridad"]
    )

    # Para cada día, toma el último valor no nulo de cada columna.
    consolidado = (
        combinado.groupby(
            "Fecha",
            as_index=False,
        )
        .agg(ultimo_no_nulo)
    )

    consolidado["Fuente"] = (
        "SIGA_INTA_BORDENAVE"
    )
    consolidado["TipoDato"] = "Observado"
    consolidado["CalidadDato"] = (
        "Observado_estacion"
    )

    if "_prioridad" in consolidado.columns:
        consolidado = consolidado.drop(
            columns=["_prioridad"]
        )

    return asegurar_columnas(
        consolidado.sort_values("Fecha")
        .reset_index(drop=True)
    )


def detectar_huecos_observados(
    observaciones: pd.DataFrame,
    hoy: date,
) -> list[date]:
    """Lista fechas vencidas sin observación."""
    if hoy <= CAMPANIA_START:
        return []

    esperadas = pd.date_range(
        CAMPANIA_START,
        hoy - timedelta(days=1),
        freq="D",
    )

    disponibles = set(
        pd.to_datetime(
            observaciones["Fecha"],
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
    )

    return [
        marca.date()
        for marca in esperadas
        if marca.normalize() not in disponibles
    ]


def completar_huecos_con_pronostico_vencido(
    observaciones: pd.DataFrame,
    maestro_anterior: pd.DataFrame,
    huecos: list[date],
) -> pd.DataFrame:
    """
    Opcional: conserva valores pronosticados vencidos, marcándolos
    explícitamente como estimados. Está desactivado por defecto.
    """
    if (
        not RELLENAR_HUECOS_CON_PRONOSTICO_VENCIDO
        or not huecos
        or maestro_anterior.empty
    ):
        return observaciones

    huecos_ts = {
        pd.Timestamp(fecha)
        for fecha in huecos
    }

    candidatos = maestro_anterior.loc[
        maestro_anterior["Fecha"].dt.normalize().isin(
            huecos_ts
        )
    ].copy()

    if candidatos.empty:
        return observaciones

    candidatos["Fuente"] = (
        candidatos["Fuente"]
        .fillna("Pronostico_anterior")
        .astype(str)
    )

    candidatos["TipoDato"] = (
        "Estimado"
    )
    candidatos["CalidadDato"] = (
        "Pronostico_vencido_sin_SIGA"
    )

    return asegurar_columnas(
        pd.concat(
            [observaciones, candidatos],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["Fecha"],
            keep="first",
        )
        .sort_values("Fecha")
        .reset_index(drop=True)
    )


def archivar_pronostico(
    df: pd.DataFrame,
) -> Path:
    """Guarda cada emisión para validación D+1 a D+7."""
    DIRECTORIO_PRONOSTICOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    sello = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    fuente = (
        str(df["Fuente"].iloc[0])
        if not df.empty
        else "pronostico"
    )

    fuente = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        fuente,
    ).lower()

    destino = (
        DIRECTORIO_PRONOSTICOS
        / f"{fuente}_{sello}.csv"
    )

    copia = df.copy()
    copia["Fecha"] = pd.to_datetime(
        copia["Fecha"]
    ).dt.strftime("%Y-%m-%d")

    copia.to_csv(
        destino,
        index=False,
        float_format="%.3f",
    )

    return destino


def guardar_cache_siga(
    observaciones: pd.DataFrame,
) -> None:
    """Persiste exclusivamente las observaciones SIGA."""
    cache = asegurar_columnas(
        observaciones
    ).copy()

    cache["Fecha"] = pd.to_datetime(
        cache["Fecha"]
    ).dt.strftime("%Y-%m-%d")

    escribir_csv_atomico(
        cache,
        ARCHIVO_SIGA_CACHE,
    )


def guardar_estado(
    *,
    estado_siga: str,
    fuente_pronostico: str,
    observaciones: pd.DataFrame,
    pronostico: pd.DataFrame,
    huecos: list[date],
) -> None:
    """Guarda un diagnóstico legible por la aplicación."""
    ARCHIVO_ESTADO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ultima_observacion = None
    if not observaciones.empty:
        ultima_observacion = (
            pd.to_datetime(
                observaciones["Fecha"]
            )
            .max()
            .date()
            .isoformat()
        )

    inicio_pronostico = None
    fin_pronostico = None

    if not pronostico.empty:
        fechas = pd.to_datetime(
            pronostico["Fecha"]
        )
        inicio_pronostico = (
            fechas.min().date().isoformat()
        )
        fin_pronostico = (
            fechas.max().date().isoformat()
        )

    contenido = {
        "ejecucion_utc": fecha_utc_iso(),
        "estado_siga": estado_siga,
        "ultima_observacion_siga": ultima_observacion,
        "fuente_pronostico": fuente_pronostico,
        "inicio_pronostico": inicio_pronostico,
        "fin_pronostico": fin_pronostico,
        "huecos_observados": [
            fecha.isoformat()
            for fecha in huecos
        ],
    }

    temporal = ARCHIVO_ESTADO.with_suffix(
        ".json.tmp"
    )

    temporal.write_text(
        json.dumps(
            contenido,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporal.replace(ARCHIVO_ESTADO)


def construir_archivo_operativo(
    observaciones: pd.DataFrame,
    pronostico: pd.DataFrame,
    hoy: date,
) -> pd.DataFrame:
    """Une pasado observado con presente y futuro pronosticado."""
    pasado = observaciones.loc[
        pd.to_datetime(
            observaciones["Fecha"]
        ).dt.date < hoy
    ].copy()

    futuro = pronostico.loc[
        pd.to_datetime(
            pronostico["Fecha"]
        ).dt.date >= hoy
    ].copy()

    combinado = pd.concat(
        [pasado, futuro],
        ignore_index=True,
        sort=False,
    )

    combinado = asegurar_columnas(
        combinado
    )

    combinado["Fecha"] = pd.to_datetime(
        combinado["Fecha"],
        errors="coerce",
    )

    combinado = (
        combinado.dropna(
            subset=["Fecha", "TMAX", "TMIN"]
        )
        .drop_duplicates(
            subset=["Fecha"],
            keep="last",
        )
        .sort_values("Fecha")
        .reset_index(drop=True)
    )

    combinado = combinado.loc[
        combinado["Fecha"].dt.date
        >= CAMPANIA_START
    ].copy()

    combinado["Fecha"] = combinado[
        "Fecha"
    ].dt.strftime("%Y-%m-%d")

    return combinado


# ===============================================================
# EJECUCIÓN
# ===============================================================

def update_file(
    archivo_maestro: Path = ARCHIVO_MAESTRO,
    archivo_siga_forzado: Path | None = None,
    solo_validar_siga: bool = False,
) -> None:
    """Ejecuta la actualización integral."""
    hoy = hoy_argentina()
    ayer = hoy - timedelta(days=1)

    if hoy < CAMPANIA_START:
        print(
            f"⏳ Esperando inicio de campaña: "
            f"{CAMPANIA_START}"
        )
        return

    maestro_anterior = cargar_archivo_maestro(
        archivo_maestro
    )

    estado_siga = "sin_actualizar"

    try:
        observaciones_nuevas, estado_siga = (
            obtener_siga_dataframe(
                CAMPANIA_START,
                ayer,
                archivo_forzado=archivo_siga_forzado,
            )
        )

        print(
            f"✅ SIGA cargado: "
            f"{len(observaciones_nuevas)} días."
        )

    except Exception as error_siga:
        print(
            f"⚠️ No se pudo actualizar SIGA: {error_siga}"
        )

        # Permite actualizar el pronóstico usando observaciones
        # previamente guardadas.
        if ARCHIVO_SIGA_CACHE.exists():
            observaciones_nuevas = pd.read_csv(
                ARCHIVO_SIGA_CACHE,
                parse_dates=["Fecha"],
            )
            observaciones_nuevas = asegurar_columnas(
                observaciones_nuevas
            )
            estado_siga = "cache_por_falla"
        else:
            observaciones_nuevas = pd.DataFrame(
                columns=COLUMNAS_COMPLETAS
            )
            estado_siga = "no_disponible"

    observaciones = consolidar_observaciones(
        observaciones_nuevas,
        maestro_anterior,
        hoy,
    )

    huecos = detectar_huecos_observados(
        observaciones,
        hoy,
    )

    observaciones = completar_huecos_con_pronostico_vencido(
        observaciones,
        maestro_anterior,
        huecos,
    )

    # Recalcular huecos después del posible relleno.
    huecos = detectar_huecos_observados(
        observaciones,
        hoy,
    )

    guardar_cache_siga(
        observaciones.loc[
            observaciones["TipoDato"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("observado")
        ].copy()
    )

    if solo_validar_siga:
        print(
            "\nÚltimos registros SIGA normalizados:"
        )
        print(
            observaciones[
                [
                    "Fecha",
                    "TMAX",
                    "TMIN",
                    "TMEDIA",
                    "Prec",
                    "Fuente",
                ]
            ]
            .tail(10)
            .to_string(index=False)
        )

        if huecos:
            print(
                "\n⚠️ Fechas vencidas sin SIGA: "
                + ", ".join(
                    fecha.isoformat()
                    for fecha in huecos
                )
            )
        else:
            print(
                "\n✅ No se detectaron huecos observados."
            )

        return

    pronostico, fuente_pronostico = (
        obtener_pronostico()
    )

    archivo_emision = archivar_pronostico(
        pronostico
    )

    print(
        f"🗂️ Pronóstico archivado en: "
        f"{archivo_emision}"
    )

    operativo = construir_archivo_operativo(
        observaciones,
        pronostico,
        hoy,
    )

    escribir_csv_atomico(
        operativo,
        archivo_maestro,
    )

    guardar_estado(
        estado_siga=estado_siga,
        fuente_pronostico=fuente_pronostico,
        observaciones=observaciones,
        pronostico=pronostico,
        huecos=huecos,
    )

    print(
        f"\n[OK] Archivo actualizado: "
        f"{archivo_maestro}"
    )
    print(
        f"Filas totales: {len(operativo)}"
    )

    if huecos:
        print(
            "⚠️ Fechas vencidas sin observación SIGA: "
            + ", ".join(
                fecha.isoformat()
                for fecha in huecos[-20:]
            )
        )
    else:
        print(
            "✅ Todas las fechas vencidas poseen "
            "observación SIGA."
        )

    columnas_mostrar = [
        "Fecha",
        "TMAX",
        "TMIN",
        "Prec",
        "Fuente",
        "TipoDato",
        "Prob_Prec_ge_10mm",
        "Prob_Prec_ge_30mm",
    ]

    print(
        "\nÚltimos registros operativos:"
    )
    print(
        operativo[
            [
                columna
                for columna in columnas_mostrar
                if columna in operativo.columns
            ]
        ]
        .tail(12)
        .to_string(index=False)
    )


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actualiza observaciones SIGA y pronóstico "
            "ECMWF ENS para PREDWEEM."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ARCHIVO_MAESTRO,
        help="Archivo CSV operativo de salida.",
    )

    parser.add_argument(
        "--siga-file",
        type=Path,
        default=None,
        help=(
            "Fuerza el uso de un XLS/XLSX/CSV local "
            "de SIGA."
        ),
    )

    parser.add_argument(
        "--solo-validar-siga",
        action="store_true",
        help=(
            "Procesa SIGA y finaliza sin consultar "
            "el pronóstico."
        ),
    )

    return parser


if __name__ == "__main__":
    argumentos = crear_parser().parse_args()

    try:
        update_file(
            archivo_maestro=argumentos.output,
            archivo_siga_forzado=argumentos.siga_file,
            solo_validar_siga=argumentos.solo_validar_siga,
        )

    except Exception as error:
        print(
            f"❌ Error durante la actualización "
            f"meteorológica: {error}"
        )
        sys.exit(1)
