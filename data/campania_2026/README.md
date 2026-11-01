# Campaña 2026 de Bordenave: emergencia y clima

Copia fija de los datos de la campaña 2026, la última de las 17 campañas con que se calibró y validó la alerta de flushes
(ver el README principal). La actualización automática no modifica estos archivos; `meteo_daily.csv`, en cambio, se reemplaza
con la campaña nueva el 1 de enero de cada año.

## Emergencia de *Lolium multiflorum*

| Archivo | Contenido |
|---|---|
| `emergencia_bordenave_2026_original.xlsx` | Planilla aportada (hoja «2026»), sin cambios. |
| `emergencia_bordenave_2026.csv` | Los mismos conteos en formato de tabla, con columnas derivadas. |

Son conteos semanales de plántulas nuevas, de 02/02/2026 a 14/09/2026 (33 fechas). El 08/06/2026 no tiene conteo.
- `SR_marco1`–`SR_marco3`: plántulas por marco **sin remoción** (sin labranza).
- `CR_marco4`–`CR_marco6`: plántulas por marco **con remoción** (suelo labrado).
- `SR_pl_m2` / `CR_pl_m2`: media de los marcos con dato expresada en plántulas/m², tal como figura en la planilla (media × 4, marcos de 0,25 m²).
- `*_fraccion_pct` / `*_acumulada_pct`: cada conteo y su acumulado, sobre el total de la campaña.
  Totales: 7041.4 pl/m² sin remoción y 6801.3 pl/m² con remoción.

La alerta se calibró con la serie **sin remoción**.

## Clima

`clima_bordenave_2026.csv` es una copia de `meteo_daily.csv` al cierre de la campaña: 274 días, del 01/01/2026 al 01/10/2026, con 367 mm de lluvia.
Los datos son observaciones de la estación SIGA–INTA Bordenave (`TipoDato` = Observado), salvo 7 días (04/06 – 10/06)
que SIGA no tenía publicados y se completaron con ECMWF IFS histórico (`TipoDato` = Provisional). Las columnas de ensamble quedan vacías en los días observados.
La serie SIGA descargada es `data/siga_bordenave_observado.csv`.

## Huellas SHA-256

| Archivo | SHA-256 |
|---|---|
| `emergencia_bordenave_2026_original.xlsx` | `a8c8bacb03a56498165a38aa8cdcdffd104fd3d2652538435a2e90ae8fd0adbf` |
| `emergencia_bordenave_2026.csv` | `4f4bf858e450ca3d11b71af88a8d84d138e27e402e5054cad89150e11fa13741` |
| `clima_bordenave_2026.csv` | `5ed7224bb9ba2fbf3a1dd14e4e592af0054e0f9916906c9730df405118661902` |
