import streamlit as st
import pandas as pd
import zipfile
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Procesador de Campo PREDWEEM", layout="centered")

st.title("📊 Unificador de Datos de Campo (Patrones)")
st.markdown("""
Sube tu archivo `Patron de emergencia LOLMU.xlsx`. El sistema detectará automáticamente 
las columnas de fecha y densidad en cada hoja, unificándolas al formato `FECHA, PLM2`.
""")

f_excel = st.file_uploader("Subir Patrón de Emergencia (.xlsx)", type=['xlsx'])

if f_excel:
    try:
        # Leer todas las hojas
        excel_data = pd.read_excel(f_excel, sheet_name=None)
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for sheet_name, df in excel_data.items():
                
                # 1. Limpieza inicial: Eliminar filas y columnas completamente vacías
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                # 2. Búsqueda inteligente de columnas
                # Buscamos una columna que se parezca a 'fecha' y otra a 'plantas/promedio'
                col_fecha = None
                col_plantas = None
                
                # Iteramos sobre los nombres de las columnas actuales
                for col in df.columns:
                    c_low = str(col).lower()
                    if 'fech' in c_low:
                        col_fecha = col
                    if any(x in c_low for x in ['plant', 'prom', 'pl.m-2']):
                        col_plantas = col
                
                # Si no las encuentra en el encabezado, intentamos buscar en la primera fila de datos
                if col_fecha is None or col_plantas is None:
                    # Esto sucede cuando el Excel tiene filas vacías arriba (como en tu 2015.csv)
                    # Forzamos la búsqueda en las primeras filas
                    df.columns = df.iloc[0] # Intentamos subir el encabezado
                    df = df[1:]
                    for col in df.columns:
                        c_low = str(col).lower()
                        if 'fech' in c_low: col_fecha = col
                        if any(x in c_low for x in ['plant', 'prom', 'pl.m-2']): col_plantas = col

                if col_fecha and col_plantas:
                    # 3. Formateo Final
                    df_final = pd.DataFrame()
                    df_final['FECHA'] = pd.to_datetime(df[col_fecha])
                    df_final['PLM2'] = pd.to_numeric(df[col_plantas], errors='coerce')
                    
                    # Eliminar filas con datos nulos y ordenar
                    df_final = df_final.dropna().sort_values('FECHA')
                    df_final['FECHA'] = df_final['FECHA'].dt.strftime('%Y-%m-%d')
                    
                    # Guardar en ZIP
                    csv_data = df_final.to_csv(index=False).encode('utf-8')
                    zip_file.writestr(f"valida_{sheet_name}.csv", csv_data)
                else:
                    st.warning(f"⚠️ No se pudieron identificar las columnas en la hoja: {sheet_name}")

        st.success(f"✅ Se han procesado {len(excel_data)} años de datos de campo.")
        
        # 4. BOTÓN DE DESCARGA
        st.download_button(
            label="📥 Descargar Datos de Campo (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Campo_Unificado_PREDWEEM.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Error crítico: {e}")

st.divider()
st.info("""
**Instrucciones para Validación Cruzada:**
1. Descarga este ZIP con los datos de campo unificados.
2. Usa estos archivos junto con los de Clima (Meteo) que procesamos antes.
3. Ahora ambos archivos coincidirán perfectamente en nombres de columnas y formatos de fecha.
""")
