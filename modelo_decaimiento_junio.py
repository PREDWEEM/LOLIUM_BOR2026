# -*- coding: utf-8 -*-
"""Decaimiento tardío de PREDWEEM Bordenave aplicado desde el 15 de abril.

Objetivo operativo
------------------
- Hasta el 14 de abril inclusive: conservar exactamente el EMERREL de la
  versión previa.
- Desde el 15 de abril: ningún pulso puede superar el 50 % del máximo de
  EMERREL alcanzado antes del 15 de abril.
- A partir de ese techo del 50 %, el límite permitido sigue disminuyendo con
  una función Weibull/exponencial configurable.

Formulación
-----------
Sea M_pre el máximo EMERREL previo al 15-abr y t los días desde 15-abr:

    techo_0 = 0.50 * M_pre
    F(t) = (1-I) + I*exp(-(t/tau)**beta)
    techo(t) = techo_0 * F(t)
    EMERREL_final(t) = min(EMERREL_original(t), techo(t))

El 15-abr F(0)=1, por lo que el techo comienza exactamente en 50 % del máximo
previo y luego cae progresivamente. El ajuste actúa antes de EMERAC,
Event-to-Event, T50 y las métricas.
"""

from __future__ import annotations

import numpy as np


TAU_DECAIMIENTO_JUNIO_DEFAULT = 60.0
BETA_DECAIMIENTO_JUNIO_DEFAULT = 1.0
INTENSIDAD_DECAIMIENTO_JUNIO_DEFAULT = 0.75
FRACCION_MAX_JUNIO = 0.50


def factor_decaimiento_desde_junio(
    dias_desde_inicio,
    activo,
    tau: float = TAU_DECAIMIENTO_JUNIO_DEFAULT,
    beta: float = BETA_DECAIMIENTO_JUNIO_DEFAULT,
    intensidad: float = INTENSIDAD_DECAIMIENTO_JUNIO_DEFAULT,
):
    """Factor temporal: 1 antes del 15-abr y Weibull desde esa fecha."""
    dias = np.maximum(np.asarray(dias_desde_inicio, dtype=float), 0.0)
    activo_arr = np.asarray(activo, dtype=bool)
    tau = max(float(tau), 1e-9)
    beta = max(float(beta), 1e-9)
    intensidad = float(np.clip(intensidad, 0.0, 1.0))

    factor = np.ones_like(dias, dtype=float)
    factor[activo_arr] = (
        (1.0 - intensidad)
        + intensidad * np.exp(-((dias[activo_arr] / tau) ** beta))
    )
    return np.clip(factor, 0.0, 1.0)


def _reemplazar_unico(source: str, old: str, new: str, etiqueta: str) -> str:
    cantidad = source.count(old)
    if cantidad != 1:
        raise RuntimeError(
            f"Parche de decaimiento desde 15-abr no aplicado: '{etiqueta}' aparece "
            f"{cantidad} veces; se esperaba exactamente una coincidencia."
        )
    return source.replace(old, new, 1)


def parchear_modelo_decaimiento_junio(source: str) -> str:
    """Integra techo 50 % + decaimiento desde 15-abr, preservando lo anterior."""

    sidebar_old = '''st.sidebar.markdown("**Validación del Primer Pico**")
st.sidebar.info(
    f"El inicio de la campaña se habilita únicamente cuando "
    f"EMERREL > {UMBRAL_PRIMER_PICO:.2f}."
)

residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 0)'''

    sidebar_new = '''st.sidebar.markdown("**Validación del Primer Pico**")
st.sidebar.info(
    f"El inicio de la campaña se habilita únicamente cuando "
    f"EMERREL > {UMBRAL_PRIMER_PICO:.2f}."
)

st.sidebar.markdown("**Decaimiento tardío (desde 15 de abril)**")
st.sidebar.info(
    "Desde el 15 de abril, los pulsos quedan limitados como máximo al 50 % "
    "del mayor pulso simulado previo. Desde ese techo, el límite continúa "
    "disminuyendo con el decaimiento tardío."
)
tau_decaimiento = st.sidebar.number_input(
    "Tau de decaimiento desde 15-abr (días)",
    min_value=5.0,
    max_value=180.0,
    value=60.0,
    step=5.0,
    help="No afecta fechas anteriores al 15-abr. Menor tau = caída más rápida.",
)
beta_decaimiento = st.sidebar.number_input(
    "Forma beta del decaimiento tardío",
    min_value=0.3,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="beta=1 equivale a decaimiento exponencial desde el 15 de abril.",
)
intensidad_decaimiento = st.sidebar.slider(
    "Intensidad del decaimiento tardío",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Controla cuánto sigue reduciéndose el techo de 50 % después del 15-abr.",
)
st.sidebar.caption(
    "Hasta el 14-abr el modelo permanece intacto. El 15-abr el techo es 50 % "
    "del máximo previo; luego techo(t)=0,50·Máx_pre·[(1−I)+I·exp(−(días/tau)^beta)]."
)

residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 0)'''

    source = _reemplazar_unico(source, sidebar_old, sidebar_new, "controles tardíos")

    optimizer_call_old = '''                    exponente_kr=exponente_kr,
                    calentamiento_suelo=calentamiento_suelo,
                )'''
    optimizer_call_new = '''                    exponente_kr=exponente_kr,
                    calentamiento_suelo=calentamiento_suelo,
                    tau_decaimiento=tau_decaimiento,
                    beta_decaimiento=beta_decaimiento,
                    intensidad_decaimiento=intensidad_decaimiento,
                )'''
    source = _reemplazar_unico(source, optimizer_call_old, optimizer_call_new, "paso al optimizador")

    main_call_old = '''        exponente_kr=exponente_kr,
        calentamiento_suelo=calentamiento_suelo,
    )'''
    main_call_new = '''        exponente_kr=exponente_kr,
        calentamiento_suelo=calentamiento_suelo,
        tau_decaimiento=tau_decaimiento,
        beta_decaimiento=beta_decaimiento,
        intensidad_decaimiento=intensidad_decaimiento,
    )'''
    source = _reemplazar_unico(source, main_call_old, main_call_new, "paso al motor principal")

    motor_old = '''    df, idx_primer_pico = aplicar_filtro_primer_pico(df, umbral=UMBRAL_PRIMER_PICO)

    df["EMERAC"] = df["EMERREL"].cumsum()'''

    motor_new = '''    df, idx_primer_pico = aplicar_filtro_primer_pico(df, umbral=UMBRAL_PRIMER_PICO)

    # Control tardío: hasta el 14 de abril queda exactamente sin modificar.
    df["EMERREL_ANTES_DECAIMIENTO_15ABR"] = df["EMERREL"].copy()
    df["Dias_Desde_15Abr"] = 0.0
    df["Factor_Decaimiento_15Abr"] = 1.0
    df["Techo_EMERREL_15Abr"] = np.nan

    tau_d = max(float(tau_decaimiento) if tau_decaimiento is not None else 60.0, 1e-9)
    beta_d = max(float(beta_decaimiento) if beta_decaimiento is not None else 1.0, 1e-9)
    intensidad_d = float(np.clip(
        intensidad_decaimiento if intensidad_decaimiento is not None else 0.0,
        0.0,
        1.0,
    ))

    inicio_decaimiento = pd.to_datetime({
        "year": df["Fecha"].dt.year,
        "month": np.full(len(df), 4),
        "day": np.full(len(df), 15),
    })
    mascara_decay = df["Fecha"] >= inicio_decaimiento
    dias_decay = (df["Fecha"] - inicio_decaimiento).dt.days.clip(lower=0).astype(float)

    # Máximo previo al 15 de abril por campaña/año. Ese máximo define 100 %
    # visual y el techo del 15-abr arranca en exactamente 50 % de ese valor.
    max_pre_por_anio = {}
    for anio in sorted(df["Fecha"].dt.year.dropna().unique()):
        fecha_inicio_anio = pd.Timestamp(year=int(anio), month=4, day=15)
        mascara_pre = (df["Fecha"].dt.year == anio) & (df["Fecha"] < fecha_inicio_anio)
        max_pre = float(df.loc[mascara_pre, "EMERREL"].clip(lower=0.0).max()) if mascara_pre.any() else 0.0
        max_pre_por_anio[int(anio)] = max_pre

    factor_decay = np.ones(len(df), dtype=float)
    factor_decay[mascara_decay.to_numpy()] = (
        (1.0 - intensidad_d)
        + intensidad_d * np.exp(
            -((dias_decay[mascara_decay].to_numpy() / tau_d) ** beta_d)
        )
    )

    techo_decay = np.full(len(df), np.nan, dtype=float)
    for pos, (_, fila) in enumerate(df.iterrows()):
        if bool(mascara_decay.iloc[pos]):
            max_pre = max_pre_por_anio.get(int(fila["Fecha"].year), 0.0)
            techo_decay[pos] = 0.50 * max_pre * factor_decay[pos]

    df["Dias_Desde_15Abr"] = dias_decay
    df["Factor_Decaimiento_15Abr"] = np.clip(factor_decay, 0.0, 1.0)
    df["Techo_EMERREL_15Abr"] = techo_decay

    if mascara_decay.any():
        valores_originales = df.loc[mascara_decay, "EMERREL"].clip(lower=0.0).to_numpy()
        techos_activos = df.loc[mascara_decay, "Techo_EMERREL_15Abr"].fillna(0.0).to_numpy()
        df.loc[mascara_decay, "EMERREL"] = np.minimum(
            valores_originales,
            techos_activos,
        )

    df["Tau_Decaimiento_15Abr_d"] = tau_d
    df["Beta_Decaimiento_15Abr"] = beta_d
    df["Intensidad_Decaimiento_15Abr"] = intensidad_d
    df["Fraccion_Maxima_15Abr"] = 0.50

    df["EMERAC"] = df["EMERREL"].cumsum()'''

    source = _reemplazar_unico(source, motor_old, motor_new, "motor desde 15-abr")
    return source
