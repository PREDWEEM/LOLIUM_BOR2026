# PREDWEEM — Lolium Bordenave 2026

Repositorio correspondiente a la implementación de **PREDWEEM** para la predicción de la emergencia y la dinámica fenológica de *Lolium multiflorum* en Bordenave, provincia de Buenos Aires, Argentina.

> **Propiedad intelectual**  
> Copyright © 2026 Guillermo R. Chantre / PREDWEEM.  
> Todos los derechos reservados.
>
> Este repositorio constituye software propietario. Su disponibilidad pública no concede autorización para utilizar, copiar, modificar, redistribuir, sublicenciar, realizar ingeniería inversa ni explotar comercialmente el código, los modelos, los parámetros, los pesos neuronales, la documentación o los datos incluidos.
>
> Consulte el aviso completo en [COPYRIGHT.md](COPYRIGHT.md).

## Finalidad

PREDWEEM es una herramienta de apoyo a la toma de decisiones agronómicas basada en la integración de datos meteorológicos, modelos predictivos y filtros ecofisiológicos para anticipar los flujos de emergencia de raigrás anual.

La implementación de este repositorio está orientada a **Bordenave** y debe utilizarse considerando el dominio geográfico, climático y agronómico para el cual fue configurada, así como su estado específico de validación.

## Preparación para repositorio privado

La aplicación carga los datos, pesos y modelos desde el checkout local. Antes de cambiar la visibilidad, autorice a Streamlit Community Cloud para acceder a los repositorios privados de `PREDWEEM`. El procedimiento completo se encuentra en [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md).

La automatización `Actualizar SIGA Bordenave y ECMWF ENS` continúa descargando datos desde SIGA–INTA y actualizando `meteo_daily.csv` mediante GitHub Actions.

## Arquitectura meteorológica operativa

La serie diaria mantiene una jerarquía explícita de fuentes:

1. **SIGA–INTA Bordenave:** fuente observada prioritaria y definitiva.
2. **ECMWF IFS histórico:** completa provisionalmente cualquier fecha vencida que todavía no tenga una observación SIGA válida, incluidos huecos internos.
3. **ECMWF IFS ENS 0,25°:** pronóstico desde el día actual hasta seis días posteriores.

Las filas provisionales se identifican mediante:

- `Fuente=ECMWF_IFS_HISTORICO`
- `TipoDato=Provisional`
- `CalidadDato=Provisional_hasta_reemplazo_SIGA`

Cuando SIGA publica posteriormente una fecha provisional, la observación de estación tiene prioridad y reemplaza automáticamente el valor modelado.

### Estadístico operativo del pronóstico

Para las filas futuras, PREDWEEM utiliza coherentemente la mediana del ensamble:

- `TMAX = TMAX_P50`
- `TMIN = TMIN_P50`
- `TMEDIA = TMEDIA_P50`
- `Prec = Prec_P50`

Las medias del ensamble se conservan en `TMAX_Media_Ens`, `TMIN_Media_Ens`, `TMEDIA_Media_Ens` y `Prec_Media_Ens`.

La precipitación horaria faltante no se convierte en cero. Temperatura y precipitación se emparejan por identificador de miembro; cada miembro y día debe aportar 24 horas válidas. El pronóstico exige al menos 30 miembros válidos y el 80 % de los miembros emparejados disponibles.

Antes de guardar `meteo_daily.csv`, el workflow verifica continuidad diaria, ausencia de nulos, precipitación no negativa, `TMAX >= TMIN`, ubicación temporal de observados/provisionales/pronósticos y correspondencia exacta entre las variables operativas y sus P50.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**
