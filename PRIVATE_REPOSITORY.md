# Preparación para repositorio privado

Este repositorio fue acondicionado para funcionar desde un checkout privado.

## Antes de cambiar la visibilidad

1. Autorizar a Streamlit Community Cloud para acceder a los repositorios privados de PREDWEEM.
2. Confirmar que la aplicación utiliza la rama `main` y el archivo `app_emergenciacombinado.py`.
3. Verificar que GitHub Actions esté habilitado y que los secretos SIGA opcionales continúen configurados.

## Después de privatizar

1. Ejecutar manualmente `Actualizar SIGA y ECMWF ENS`.
2. Confirmar que `meteo_daily.csv` y los archivos de `data/` se actualicen.
3. Verificar que Streamlit cargue los modelos y los datos locales.
4. Mantener la aplicación pública solo desde la configuración de Streamlit, no desde GitHub.

## Recursos requeridos

La aplicación necesita los archivos `IW.npy`, `LW.npy`, `bias_IW.npy`, `bias_out.npy`, `modelo_clusters_k3.pkl` y `meteo_daily.csv` en el checkout privado.
