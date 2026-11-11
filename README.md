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

### Campaña vigente y actualización automática

Cada campaña carga clima del **1 de enero al 1 de octubre** (inclusive) del año en curso, hora de Argentina. Mientras
la campaña está abierta, la serie se actualiza automáticamente dos veces por día (07:30 y 15:30): SIGA–INTA hasta
ayer, puente provisional ECMWF IFS para los días que SIGA todavía no publicó y pronóstico ECMWF ENS (P50) de los
siete días siguientes; el pronóstico se recorta al 1 de octubre.

Del 2 de octubre al 31 de diciembre la campaña queda cerrada: no se consulta el ensamble y la serie se conserva. El
**1 de enero se reinicia automáticamente la carga** con la campaña nueva (la de 2027 empieza el 1/01/2027). Durante
las primeras tres semanas se admite una serie sin observaciones SIGA del año, cubierta con el puente provisional y
el pronóstico ECMWF, porque SIGA publica con demora. Las variables `CAMPANIA_INICIO` y `CAMPANIA_FIN` (AAAA-MM-DD)
permiten fijar otras fechas. Los archivos de pronóstico históricos se conservan.

## Alerta validada de grandes flujos (sección independiente)

Debajo del motor PREDWEEM, la app muestra una **alerta de flushes validada**: la probabilidad de que en los
próximos 7 días emerja ≥ 5 % de la campaña, con dos niveles fijos (*Atención* desde 15 %, *Alerta* desde 25 %).
No modifica el motor científico ni sus resultados (`alerta_lolmu.py`, `visualizacion_alerta_lolmu.py`,
`alerta_lolmu_params.json`).

Combina, con la información disponible a la fecha de emisión:

1. el **calendario medio** de emergencia de LOLMU (17 campañas);
2. la **respuesta a las lluvias ya registradas** (modelo de pulsos ajustado con las campañas de *Lolium* de Bordenave:
   rezago gamma en tiempo térmico de suelo, mayor respuesta sobre suelo seco y tamaño del pulso según la época del año);
3. opcionalmente, la **fecha en que se vieron las primeras plántulas** del lote (un solo dato por campaña), que además
   ancla el reloj térmico de la ventana de control (600–800 °Cd);
4. opcionalmente, los **conteos** del lote: se cargan en la sección de la alerta (tabla o planilla Excel/CSV con una
   columna «Fecha» y una por marco; se pueden elegir los marcos a usar).

Validación dejando cada campaña afuera (17 campañas: 2015, 2017–2022 y 2026 del ensayo de Bordenave; 2008–2014, 2023
y 2024 de las curvas históricas de PREDWEEM; 386 semanas, 97 flushes):

| Información disponible | AUC | Flushes detectados con Alerta | Semanas marcadas |
|---|---|---|---|
| Solo clima | 0,77 | 77 % | 44 % |
| + fecha de primeras plántulas | 0,83 | 81 % | 44 % |
| + conteos semanales | 0,87 | 82 % | 40 % |

Las probabilidades están calibradas. Con las mismas campañas, el motor PREDWEEM anticipa flushes con AUC ~0,70, pero
anticipa bien la **primera cohorte**; por eso se usa como inicio estimado del reloj térmico hasta que se informa la fecha
de las primeras plántulas. Límites: un sitio, lotes sin labranza; no anticipa flushes sin lluvia registrada.

La barra lateral conserva solo los parámetros que usa la alerta: temperaturas base, óptima y crítica del tiempo
térmico y los objetivos de la ventana de control (600 y 800 °Cd). Los parámetros del motor PREDWEEM quedan en sus
valores por defecto.

## Curva acumulada en tiempo térmico desde las primeras plántulas

Cuando se informa la fecha de las primeras plántulas (o hay un primer conteo con plántulas), la curva de emergencia
acumulada deja de seguir el calendario y sigue el **tiempo térmico acumulado desde esa fecha** (media y rango 10–90 % de
17 campañas de Bordenave, `curva_tt` en `alerta_lolmu_params.json`); más allá del clima disponible se suman temperaturas
medias históricas. Error medio de la emergencia acumulada en las fechas de conteo desde las primeras plántulas (puntos %):

| Localidad | Calendario por fecha | Tiempo térmico desde el inicio |
|---|---|---|
| Bordenave (17 campañas, dejando cada una afuera) | 12,9 | 11,3 |
| Lartigau 2026 | 9,9 | 5,8 |
| Olavarría 2026 | 13,7 | 10,7 |
| Pergamino 2024 y 2026 | 17,5 | 7,2 |
| San Pedro 2023, 2024 y 2026 | 19,5 | 14,4 |

Mejora sobre todo donde la fecha de inicio varía mucho entre años. En Tres Arroyos y Balcarce sigue el calendario propio
del sitio (error 6,1 y 6,9 puntos) y en Azul el calendario de Bordenave (5,3 contra 15,8 en su única campaña).

## Señal de lluvia (actualización de noviembre de 2026)

La señal de lluvia usa ahora el **modelo de pulsos ajustado con las 17 campañas de *Lolium multiflorum* de Bordenave**.
El tamaño de cada pulso depende del agua que entró al suelo, de cuán seco estaba y de la época del año (máximo para
lluvias de fines de mayo); la mitad de la respuesta emerge dentro de 114 °Cd de tiempo térmico de suelo y el 90 %
dentro de 252 °Cd. Con el nuevo núcleo se reajustaron los modelos logísticos, los percentiles de referencia y la
recalibración de flujos tardíos (`alerta_lolmu_params.json`). El balance de agua y la temperatura de suelo no cambian.

Validación con cada campaña de Bordenave predicha con el ajuste sin ella (incluido el núcleo de lluvia): AUC 0,77 solo
con clima, 0,83 con primeras plántulas y 0,87 con conteos. En las ocho localidades de validación, sin reajustar: 0,89,
0,92 y 0,93 (calendario solo, 0,89).

## Aviso de flujos tardíos

Del 1 de mayo al 30 de junio la app muestra además la **probabilidad de flujos tardíos**: que en los próximos 7 días
emerja al menos 2 % de la campaña. El aviso se da desde 20 %; sin datos del lote, desde 25 % (`tardia.umbral_tramos`).
En esa época los grandes flujos (≥ 5 %) son raros, pero pocas plantas por semana importan para decidir la fecha de siembra.
Es la misma alerta recalibrada por nivel de información (`tardia` en `alerta_lolmu_params.json`). Aparece en un recuadro
debajo de la alerta, en el **semáforo semanal** del gráfico (fila «Flujos tardíos»: violeta fuerte = flujo tardío probable;
la fila «Grandes flujos» repite el nivel de cada semana) y en la descarga en Excel.

| Información | AUC Bordenave (17 campañas, 130 semanas de mayo–junio, 42 con ≥ 2 %) | Detectadas con el aviso | Semanas marcadas |
|---|---|---|---|
| Solo clima (aviso desde 25 %) | 0,57 | 67 % | 68 % |
| + primeras plántulas | 0,79 | 90 % | 62 % |
| + conteos semanales | 0,87 | 86 % | 51 % |

Cada campaña se predijo con el ajuste sin ella. Sin datos del lote el aviso casi no distingue semanas: conviene cargar la
fecha de primeras plántulas. En otras seis localidades (58 intervalos de mayo–junio, 10 con ≥ 2 % por semana) la AUC fue
0,85 con primeras plántulas y 0,89 con conteos, con 90 % de detección; allí la probabilidad tiende a ser alta (30 % contra
17 % observado).

**Del 1 de julio al 1 de octubre no se muestra probabilidad** (`tardia.recorrida`): en Bordenave (80 semanas, 11 con
flujo tardío) no distinguió las semanas con emergencia (AUC 0,41–0,50), y bajar el umbral solo marcaba más semanas al
azar. En su lugar, la app recomienda **recorrer los lotes cada 2 semanas**, sobre todo donde suele haber emergencia
invernal (sudoeste bonaerense), antes de definir la fecha de siembra o un control. En el semáforo esas semanas aparecen
en gris.

## Datos de la campaña 2026

`data/campania_2026/` guarda una copia fija de la emergencia de *Lolium* (planilla original y CSV por marco, sin y con
remoción, con fracción y acumulada) y del clima de la campaña 2026 (01/01–01/10, SIGA–INTA). La actualización automática no
los modifica; ver `data/campania_2026/README.md`.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**
