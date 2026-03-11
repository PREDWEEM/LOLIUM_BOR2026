import streamlit as st
import pandas as pd
import zipfile
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Procesador Excel PREDWEEM", layout="centered")

st.title("📂 Procesador Meteorológico de Excel")
st.markdown("""
Sube tu archivo `CLIMA 2017-2022.xlsx`. El sistema convertirá cada hoja en un 
archivo `.csv` estandarizado con el formato: `Fecha, TMAX, TMIN, Prec`.
""")

# 1. CARGADOR DE ARCHIVO EXCEL
f_excel = st.file_uploader("Subir archivo Excel (.xlsx)", type=['xlsx'])

if f_excel:
    try:
        # Leer todas las hojas del Excel
        # sheet_name=None devuelve un diccionario {nombre_hoja: dataframe}
        excel_data = pd.read_excel(f_excel, sheet_name=None)
        
        st.info(f"Hojas detectadas: {', '.join(excel_data.keys())}")
        
        # Buffer para el archivo ZIP
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for nombre_hoja, df in excel_data.items():
                # Limpiar nombres de columnas (quitar espacios en blanco)
                df.columns = [str(c).strip() for c in df.columns]
                cols_low = [c.lower() for c in df.columns]
                
                # Mapeo inteligente de columnas
                mapeo = {}
                
                # Buscar Fecha
                if 'fecha' in cols_low:
                    mapeo[df.columns[cols_low.index('fecha')]] = 'Fecha'
                else:
                    mapeo[df.columns[0]] = 'Fecha' # Asumir primera columna
                
                # Buscar Temperaturas (independiente de mayúsculas/minúsculas)
                if 'tmax' in cols_low:
                    mapeo[df.columns[cols_low.index('tmax')]] = 'TMAX'
                if 'tmin' in cols_low:
                    mapeo[df.columns[cols_low.index('tmin')]] = 'TMIN'
                
                # Buscar Precipitación
                if 'prec' in cols_low:
                    mapeo[df.columns[cols_low.index('prec')]] = 'Prec'
                elif 'precipitacion' in cols_low:
                    mapeo[df.columns[cols_low.index('precipitacion')]] = 'Prec'

                # Aplicar estandarización
                df_std = df.rename(columns=mapeo)
                
                # Verificar si tenemos las columnas necesarias
                columnas_necesarias = ['Fecha', 'TMAX', 'TMIN', 'Prec']
                if all(col in df_std.columns for col in columnas_necesarias):
                    df_final = df_std[columnas_necesarias].copy()
                    
                    # Limpiar formato de fecha (eliminar horas si existen)
                    df_final['Fecha'] = pd.to_datetime(df_final['Fecha']).dt.strftime('%Y-%m-%d')
                    
                    # Convertir a CSV para el ZIP
                    csv_data = df_final.to_csv(index=False).encode('utf-8')
                    zip_file.writestr(f"meteo_{nombre_hoja}.csv", csv_data)
                else:
                    st.warning(f"⚠️ La hoja '{nombre_hoja}' no contiene todas las columnas necesarias.")

        st.success(f"✅ Se han procesado {len(excel_data)} años correctamente.")
        
        # 2. BOTÓN DE DESCARGA
        st.download_button(
            label="📥 Descargar Serie Histórica (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Serie_Meteo_PREDWEEM.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Error al procesar el Excel: {e}")

st.divider()
st.info("""
**Nota técnica:** Este procesador asegura que la columna `TMAX` siempre preceda a `TMIN` 
y que los nombres coincidan exactamente con lo que espera el motor de la Red Neuronal.
""")
