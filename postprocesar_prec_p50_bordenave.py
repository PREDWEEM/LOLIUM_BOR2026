# -*- coding: utf-8 -*-
"""
Postproceso PREDWEEM — Bordenave.

Objetivo:
- Mantener SIGA observado sin cambios.
- Para filas futuras ECMWF ENS, usar la mediana del ensamble como lluvia operativa:
      Prec = Prec_P50
- Conservar la media original del ensamble en:
      Prec_Media_Ens

Este archivo se ejecuta después de actualizar_meteo_bordenave.py dentro del workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVOS = [Path("meteo_daily.csv")]
ARCHIVOS.extend(sorted(Path("data/historico_pronosticos").glob("ecmwf_ifs_ens_025*.csv")))


def reordenar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Ubica Prec_Media_Ens inmediatamente después de Prec_P90."""
    columnas = list(df.columns)
    if "Prec_Media_Ens" not in columnas:
        return df

    columnas.remove("Prec_Media_Ens")
    if "Prec_P90" in columnas:
        idx = columnas.index("Prec_P90") + 1
        columnas.insert(idx, "Prec_Media_Ens")
    else:
        columnas.append("Prec_Media_Ens")
    return df[columnas]


def corregir_archivo(path: Path) -> bool:
    if not path.exists():
        return False

    df = pd.read_csv(path)
    if df.empty or "Prec" not in df.columns or "Prec_P50" not in df.columns:
        return False

    if "Prec_Media_Ens" not in df.columns:
        df["Prec_Media_Ens"] = np.nan

    tipo = df.get("TipoDato", pd.Series("", index=df.index)).astype(str).str.lower()
    mascara_pronostico = tipo.eq("pronostico") & df["Prec_P50"].notna()

    if not mascara_pronostico.any():
        df = reordenar_columnas(df)
        df.to_csv(path, index=False, float_format="%.3f")
        return True

    # Antes de reemplazar Prec por Prec_P50, guardar la media original del ensamble.
    faltante_media = mascara_pronostico & df["Prec_Media_Ens"].isna()
    df.loc[faltante_media, "Prec_Media_Ens"] = df.loc[faltante_media, "Prec"]

    # Lluvia operativa futura: mediana del ensamble.
    df.loc[mascara_pronostico, "Prec"] = df.loc[mascara_pronostico, "Prec_P50"]

    if "CalidadDato" in df.columns:
        df.loc[mascara_pronostico, "CalidadDato"] = "Mediana_ensamble"

    df = reordenar_columnas(df)
    df.to_csv(path, index=False, float_format="%.3f")
    return True


def main() -> int:
    corregidos = []
    for path in ARCHIVOS:
        if corregir_archivo(path):
            corregidos.append(str(path))

    if corregidos:
        print("✅ Prec corregida a Prec_P50 en:")
        for item in corregidos:
            print(f"   - {item}")
    else:
        print("ℹ️ No hubo archivos para corregir.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
