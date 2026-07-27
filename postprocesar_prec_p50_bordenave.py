# -*- coding: utf-8 -*-
"""Compatibilidad PREDWEEM — Bordenave.

El actualizador principal ya genera el pronóstico con P50 operativo para
TMAX, TMIN, TMEDIA y Prec. Este script queda como herramienta manual para
normalizar archivos antiguos sin inventar valores ni completar datos nulos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVO = Path("meteo_daily.csv")

PARES = (
    ("TMAX", "TMAX_P50", "TMAX_Media_Ens"),
    ("TMIN", "TMIN_P50", "TMIN_Media_Ens"),
    ("TMEDIA", "TMEDIA_P50", "TMEDIA_Media_Ens"),
    ("Prec", "Prec_P50", "Prec_Media_Ens"),
)


def aplicar_p50_coherente(path: Path = ARCHIVO) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} está vacío.")

    tipo = df.get("TipoDato", pd.Series("", index=df.index)).astype(str)
    mascara = tipo.str.lower().eq("pronostico")
    if not mascara.any():
        print("ℹ️ No hay filas de pronóstico para normalizar.")
        return False

    for operativo, percentil, media in PARES:
        if operativo not in df.columns or percentil not in df.columns:
            raise ValueError(
                f"Faltan columnas requeridas para {operativo}: "
                f"{operativo}, {percentil}."
            )
        valores_p50 = pd.to_numeric(df.loc[mascara, percentil], errors="coerce")
        if valores_p50.isna().any():
            raise ValueError(
                f"{percentil} contiene valores nulos; no se modificó {path}."
            )

        if media not in df.columns:
            df[media] = np.nan
        original = pd.to_numeric(df.loc[mascara, operativo], errors="coerce")
        faltan_medias = df.loc[mascara, media].isna()
        indices = df.index[mascara][faltan_medias]
        df.loc[indices, media] = original.loc[indices]
        df.loc[mascara, operativo] = valores_p50.to_numpy()

    if "CalidadDato" in df.columns:
        df.loc[mascara, "CalidadDato"] = "Mediana_ensamble_P50"

    temporal = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporal, index=False, float_format="%.3f")
    temporal.replace(path)
    print(f"✅ P50 aplicado coherentemente a TMAX, TMIN, TMEDIA y Prec en {path}.")
    return True


def main() -> int:
    try:
        aplicar_p50_coherente()
        return 0
    except Exception as error:
        print(f"❌ Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
