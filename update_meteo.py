# update_meteo.py
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Configuración y constantes
URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
OUT = Path("meteo_daily.csv")
START = datetime(2026, 1, 1)

def to_float(x):
    """Convierte strings con coma decimal a float."""
    try:
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError):
        return None

def fetch_meteobahia():
    """Descarga y procesa el XML de MeteoBahía."""
    # User-Agent para evitar errores 403
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    r = requests.get(URL, headers=headers, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    # Navegamos por la estructura del XML
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
    """Lee el CSV viejo, lo actualiza con datos nuevos y lo guarda."""
    today = datetime.utcnow().date()

    # 1) Restricción de fecha de inicio
    if today < START.date():
        print(f"⏳ Esperando al {START.date()} para iniciar actualizaciones.")
        return

    # 2) Reinicio anual: Si es exactamente el día de inicio, borramos el historial previo
    if today == START.date():
        if OUT.exists():
            OUT.unlink()
            print("🆕 Historial reiniciado para el nuevo ciclo (2026).")

    # 3) Obtener datos frescos del sitio web
    print("📡 Descargando datos actuales...")
    df_new = fetch_meteobahia()

    # 4) Lógica de actualización (Merge)
    if OUT.exists():
        # Leemos el archivo actual
        df_old = pd.read_csv(OUT, parse_dates=["Fecha"])
        
        # Concatenamos. Al poner df_new al final y usar keep='last', 
        # los datos nuevos sobrescriben a los viejos si la fecha coincide.
        df_all = pd.concat([df_old, df_new]).drop_duplicates("Fecha", keep="last").sort_values("Fecha")
        print("🔄 Actualizando registros existentes y agregando nuevos...")
    else:
        df_all = df_new
        print("📝 Creando nuevo archivo meteo_daily.csv...")

    # 5) Guardar resultado final
    df_all.to_csv(OUT, index=False)
    print(f"[OK] Proceso completado. Total de registros: {len(df_all)}.")

if __name__ == "__main__":
    try:
        update_file()
    except Exception as e:
        print(f"❌ Error durante la actualización: {e}")
