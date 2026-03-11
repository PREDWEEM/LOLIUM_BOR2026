import streamlit as st
import pandas as pd
import zipfile
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Conversor de Datos PREDWEEM", layout="centered")

st.title("📂 Procesador de Series Históricas")
st.markdown("""
Sube todos los archivos `.csv` anuales que deseas convertir. 
El sistema generará el formato: `Fecha, TMAX, TMIN, Prec`.
""")

# 1. CARGADOR DE ARCHIVOS MÚLTIPLES
uploaded_files = st.file_uploader(
    "Selecciona los archivos (puedes arrastrar varios a la vez)", 
    type=['csv'], 
    accept_multiple_files=True
)

if uploaded_files:
    # Buffer para el archivo ZIP
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for uploaded_file in uploaded_files:
            try:
                # Leer archivo original
                df = pd.read_csv(uploaded_file)
                
                # Detectar columnas por posición para evitar errores de nombres
                # Estructura actual: [Fecha/Unnamed, Julian, Tmin, Tmax, prec]
                # Reasignamos a los nombres estándar del modelo
                df.columns = ['Fecha', 'JULIANO', 'TMIN', 'TMAX', 'Prec']
                
                # Seleccionar y ordenar columnas para PREDWEEM
                df_clean = df[['Fecha', 'TMAX', 'TMIN', 'Prec']].copy()
                
                # Limpieza de fechas
                df_clean['Fecha'] = pd.to_datetime(df_clean['Fecha']).dt.strftime('%Y-%m-%d')
                
                # Convertir a CSV en memoria
                csv_data = df_clean.to_csv(index=False).encode('utf-8')
                
                # Generar nombre de archivo limpio (ej: meteo_2008.csv)
                original_name = uploaded_file.name
                year_part = "".join(filter(str.isdigit, original_name))[-4:] # Extrae el año
                file_name = f"meteo_{year_part}.csv"
                
                # Añadir al ZIP
                zip_file.writestr(file_name, csv_data)
                
            except Exception as e:
                st.error(f"Error procesando {uploaded_file.name}: {e}")

    # 2. BOTÓN DE DESCARGA DEL ZIP
    st.success(f"✅ Se han procesado {len(uploaded_files)} archivos correctamente.")
    
    st.download_button(
        label="📥 Descargar todos los archivos (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="datos_meteo_PREDWEEM.zip",
        mime="application/zip"
    )

st.info("💡 Consejo: Una vez descargado el ZIP, extrae los archivos y súbelos a tu repositorio de GitHub para que el Validador Cruzado pueda utilizarlos.")
