import streamlit as st
import pandas as pd
import zipfile
import io

st.title("📦 Generador de ZIP PREDWEEM")
st.markdown("Sube los archivos meteorológicos anuales para estandarizarlos y descargarlos en un solo ZIP.")

# Cargador de múltiples archivos
archivos_subidos = st.file_uploader(
    "Selecciona los archivos CSV (ej: CLIMA... o datos LOLIUM...)", 
    type=['csv'], 
    accept_multiple_files=True
)

if archivos_subidos:
    # Buffer en memoria para el archivo ZIP
    buf_zip = io.BytesIO()
    
    with zipfile.ZipFile(buf_zip, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for f in archivos_subidos:
            try:
                df = pd.read_csv(f)
                # Limpiar nombres de columnas (quitar espacios)
                df.columns = [str(c).strip() for c in df.columns]
                
                # Mapeo inteligente de columnas según el formato detectado
                mapeo = {}
                cols_low = [c.lower() for c in df.columns]
                
                # Identificar Fecha (primera columna o columna 'fecha')
                if 'fecha' in cols_low:
                    idx = cols_low.index('fecha')
                    mapeo[df.columns[idx]] = 'Fecha'
                else:
                    mapeo[df.columns[0]] = 'Fecha'
                
                # Identificar TMAX y TMIN (independiente del orden original)
                idx_tmax = cols_low.index('tmax')
                idx_tmin = cols_low.index('tmin')
                mapeo[df.columns[idx_tmax]] = 'TMAX'
                mapeo[df.columns[idx_tmin]] = 'TMIN'
                
                # Identificar Precipitacion
                if 'prec' in cols_low:
                    idx_p = cols_low.index('prec')
                    mapeo[df.columns[idx_p]] = 'Prec'
                
                # Aplicar cambios y seleccionar solo lo necesario
                df_std = df.rename(columns=mapeo)
                df_final = df_std[['Fecha', 'TMAX', 'TMIN', 'Prec']].copy()
                
                # Asegurar formato de fecha limpio
                df_final['Fecha'] = pd.to_datetime(df_final['Fecha']).dt.strftime('%Y-%m-%d')
                
                # Convertir a CSV y añadir al ZIP
                csv_str = df_final.to_csv(index=False)
                zip_file.writestr(f.name, csv_str)
                
            except Exception as e:
                st.error(f"Error procesando {f.name}: {e}")

    # Botón de descarga
    st.success(f"¡{len(archivos_subidos)} archivos procesados con éxito!")
    st.download_button(
        label="📥 Descargar Serie Estandarizada (ZIP)",
        data=buf_zip.getvalue(),
        file_name="meteo_PREDWEEM_ajustado.zip",
        mime="application/zip"
    )
