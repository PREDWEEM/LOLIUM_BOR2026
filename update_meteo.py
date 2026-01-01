# update_meteo.py
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
OUT = Path("meteo_daily.csv")

START = datetime(2026, 1, 1)

def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

def fetch_meteobahia():
    # Definimos un "User-Agent" para parecer un navegador normal
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Agregamos los headers a la petición
    r = requests.get(URL, headers=headers, timeout=20)
    
    # Esto ahora debería funcionar sin el error 403
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    for d in root.findall(".//forecast/tabular/day"):
        fecha = d.find("fecha").get("value")
        tmax  = d.find("tmax").get("value")
        tmin  = d.find("tmin").get("value")
        prec  = d.find("precip").get("value")

        rows.append({
            "Fecha": pd.to_datetime(fecha),
            "TMAX": to_float(tmax),
            "TMIN": to_float(tmin),
            "Prec": to_float(prec),
        })

    df = pd.DataFrame(rows).sort_values("Fecha")
    return df

def update_file():
    today = datetime.utcnow().date()

    # 1) Antes del 01/01/2026 → NO HACER NADA
    if today < START.date():
        print("⏳ Antes del 01/01/2026 → no se actualiza meteo_daily.csv")
        return

    # 2) EXACTAMENTE EL 01/01/2026 → BORRAR ARCHIVO
    if today == START.date():
        if OUT.exists():
            OUT.unlink()
            print("🆕 meteo_daily.csv reiniciado el 01/01/2026.")

    # 3) Descargar datos nuevos
    df_new = fetch_meteobahia()

    # 4) Si ya existe el archivo (post-reinicio) → concatenar
    if OUT.exists():
        df_old = pd.read_csv(OUT, parse_dates=["Fecha"])
        df_all = pd.concat([df_old, df_new]).drop_duplicates("Fecha").sort_values("Fecha")
    else:
        df_all = df_new

    df_all.to_csv(OUT, index=False)
    print(f"[OK] Archivo actualizado: {len(df_all)} registros.")

if __name__ == "__main__":
    update_file()
