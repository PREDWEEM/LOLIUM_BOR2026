# Decaimiento tardío desde junio

La versión operativa conserva **sin modificaciones** la simulación entre enero y mayo. En ese período `Factor_Decaimiento_Junio = 1.0`, por lo que `EMERREL` es exactamente el mismo que en la versión previa sin decaimiento.

A partir del 1 de junio se aplica:

`F(t) = (1-I) + I * exp(-(t/tau)^beta)`

con `t` expresado en días desde el 1 de junio. El 1 de junio `F=1`, por lo que no existe una discontinuidad en la serie.

Parámetros iniciales experimentales:

- `tau = 60 días`
- `beta = 1.0`
- `intensidad = 0.75`

El factor modifica `EMERREL` antes de calcular `EMERAC`, Event-to-Event, T50 y métricas. Las columnas de auditoría son `EMERREL_ANTES_DECAIMIENTO_JUNIO`, `Dias_Desde_1Jun` y `Factor_Decaimiento_Junio`.
