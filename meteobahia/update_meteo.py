# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 NODOS CLIMÁTICOS PREDWEEM — SCON SENSOR BORDENAVE 2026
# Procesamiento e Integración Estricta de la Red MeteoBahía
# ===============================================================

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
import sys

# CONFIGURACIÓN DE RUTAS Y CONSTANTES (Endpoint específico: for-bd.xml)
URL_XML = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
ARCHIVO_CSV = Path("meteo_daily.csv")
CAMPANIA_START = datetime(2026, 1, 1).date()

def to_float(x):
    """Convierte strings del XML con coma decimal a floats limpios."""
    try:
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError):
        return None

def fetch_meteobahia_dataframe():
    """Descarga el XML de Bordenave, corrige etiquetas y devuelve un DataFrame."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = requests.get(URL_XML, headers=headers, timeout=20)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    rows = []
    
    # Recorrido del árbol XML de MeteoBahía para Bordenave
    for d in root.findall(".//forecast/tabular/day"):
        fecha_str = d.find("fecha").get("value")  # Formato YYYY-MM-DD
        tmax = d.find("tmax").get("value")
        tmin = d.find("tmin").get("value")
        prec = d.find("precip").get("value")

        rows.append({
            "Fecha": pd.to_datetime(fecha_str),
            "TMAX": to_float(tmax),
            "TMIN": to_float(tmin),
            "Prec": to_float(prec),
        })

    if not rows:
        raise ValueError("El XML de Bordenave no contenía registros procesables.")

    df = pd.DataFrame(rows)
    return df

def update_file():
    """Lee el historial, fusiona con el XML aplicando la purga de duplicados por ISO-string."""
    # Ajuste de zona horaria local argentina (ART = UTC-3) para evitar saltos nocturnos
    hoy_local = (datetime.utcnow() - timedelta(hours=3)).date()

    # 1) Control de inicio de campaña
    if hoy_local < CAMPANIA_START:
        print(f"⏳ Esperando fecha de inicio de campaña: {CAMPANIA_START}")
        return

    # 2) Blanqueo de ciclo anual (Punto de control 1 de Enero)
    if hoy_local == CAMPANIA_START and ARCHIVO_CSV.exists():
        ARCHIVO_CSV.unlink()
        print("🆕 Ciclo 2026: Historial previo reiniciado para nueva calibración en Bordenave.")

    # 3) Captura remota
    print("📡 Descargando datos frescos desde MeteoBahía para Bordenave...")
    df_new = fetch_meteobahia_dataframe()

    # 4) Filtro de horizonte: Descartar ruidos predictivos más allá de 7 días
    limite_futuro = pd.Timestamp(hoy_local + timedelta(days=7))
    df_new = df_new[df_new["Fecha"] <= limite_futuro].copy()

    # 5) Fusión lógica (Merge) con blindaje de tipos
    if ARCHIVO_CSV.exists():
        print(f"Leyendo historial existente desde {ARCHIVO_CSV}...")
        df_old = pd.read_csv(ARCHIVO_CSV)
        
        # Forzamos conversión temporal a ambos bloques para un merge seguro
        df_old["Fecha"] = pd.to_datetime(df_old["Fecha"])
        df_new["Fecha"] = pd.to_datetime(df_new["Fecha"])
        
        # Concatenamos poniendo las actualizaciones al final
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        
        # Eliminamos duplicados basados en datetime real (el dato nuevo pisa al pronóstico viejo)
        df_all = df_all.drop_duplicates(subset=["Fecha"], keep="last")
        df_all = df_all.sort_values(by="Fecha").reset_index(drop=True)
        print("🔄 Actualizando registros existentes y agregando nuevos consolidados...")
    else:
        print("📝 No se detectó historial previo. Creando archivo maestro para Bordenave...")
        df_all = df_new

    # Purga de filas corruptas antes de la persistencia
    df_all = df_all.dropna(subset=["Fecha", "TMAX", "TMIN"])

    # CORRECCIÓN DE FORMATO CRÍTICA:
    # Forzamos la escritura con formato string ISO estricto para evitar fallos de re-lectura en Pandas
    df_all["Fecha"] = df_all["Fecha"].dt.strftime("%Y-%m-%d")
    
    # Escritura física definitiva
    df_all.to_csv(ARCHIVO_CSV, index=False)
    print(f"[OK] Sincronización exitosa. Total de días registrados en Bordenave: {len(df_all)}.")
    print("Últimos 5 registros consolidados:")
    print(df_all.tail(5))

if __name__ == "__main__":
    try:
        update_file()
    except Exception as e:
        print(f"❌ Error durante la actualización del nodo Bordenave: {e}")
        sys.exit(1)
