import streamlit as st
import pandas as pd
import zipfile
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Procesador Multi-Hoja PREDWEEM", layout="centered")

st.title("📑 Procesador de Excel Histórico")
st.markdown("""
Sube tu archivo `datos LOLIUM METEORO.xlsx`. El sistema procesará cada hoja 
como un año independiente y corregirá el formato de columnas automáticamente.
""")

# 1. CARGADOR DE ARCHIVO EXCEL
f_excel = st.file_uploader("Subir archivo Excel (.xlsx)", type=['xlsx'])

if f_excel:
    try:
        # Leer todas las hojas del Excel
        # sheet_name=None devuelve un diccionario: {nombre_hoja: dataframe}
        excel_data = pd.read_excel(f_excel, sheet_name=None)
        
        st.write(f"📂 Hojas detectadas: {', '.join(excel_data.keys())}")
        
        # Buffer para el ZIP
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for sheet_name, df in excel_data.items():
                # Limpieza de nombres de columnas (quitar espacios)
                df.columns = df.columns.str.strip()
                
                # REASIGNACIÓN BASADA EN TU ESTRUCTURA:
                # [Unnamed: 0, DIA JULIANO, Tmin, Tmax, prec]
                # Usamos iloc para ser robustos ante cambios de nombre menores
                
                new_df = pd.DataFrame()
                new_df['Fecha'] = pd.to_datetime(df.iloc[:, 0])
                new_df['TMAX'] = df['Tmax']
                new_df['TMIN'] = df['Tmin']
                new_df['Prec'] = df['prec']
                
                # Ordenar por fecha y limpiar formato
                new_df = new_df.sort_values('Fecha')
                new_df['Fecha'] = new_df['Fecha'].dt.strftime('%Y-%m-%d')
                
                # Convertir a CSV para el ZIP
                csv_data = new_df.to_csv(index=False).encode('utf-8')
                
                # Nombre del archivo dentro del ZIP (ej: meteo_2025.csv)
                file_name = f"meteo_{sheet_name}.csv"
                zip_file.writestr(file_name, csv_data)
        
        st.success(f"✅ Se han procesado {len(excel_data)} años correctamente.")
        
        # 2. BOTÓN DE DESCARGA
        st.download_button(
            label="📥 Descargar Serie Histórica (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Serie_Historica_PREDWEEM.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Error al procesar el Excel: {e}")
        st.info("Asegúrate de que las hojas tengan las columnas: Tmin, Tmax y prec.")

st.divider()
st.info("""
**Nota científica:** Este proceso invierte el orden de tus columnas originales para cumplir 
con el requisito de la Red Neuronal (`TMAX` antes que `TMIN`). Esto asegura que el 
escalamiento de los sensores de la RNA sea térmicamente coherente.
""")
