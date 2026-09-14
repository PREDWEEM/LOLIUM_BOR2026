# -*- coding: utf-8 -*-
"""Decaimiento tardío de PREDWEEM Bordenave aplicado desde junio.

Objetivo operativo
------------------
- Enero–mayo: conservar exactamente el EMERREL de la versión previa.
- Desde el 1 de junio: ningún pulso puede superar el 50 % del máximo de
  EMERREL alcanzado antes de junio.
- A partir de ese techo del 50 %, el límite permitido sigue disminuyendo con
  una función Weibull/exponencial configurable.

Formulación
-----------
Sea M_pre el máximo EMERREL previo al 1-jun y t los días desde 1-jun:

    techo_0 = 0.50 * M_pre
    F(t) = (1-I) + I*exp(-(t/tau)**beta)
    techo(t) = techo_0 * F(t)
    EMERREL_final(t) = min(EMERREL_original(t), techo(t))

El 1-jun F(0)=1, por lo que el techo comienza exactamente en 50 % del máximo
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
    dias_desde_junio,
    activo,
    tau: float = TAU_DECAIMIENTO_JUNIO_DEFAULT,
    beta: float = BETA_DECAIMIENTO_JUNIO_DEFAULT,
    intensidad: float = INTENSIDAD_DECAIMIENTO_JUNIO_DEFAULT,
):
    """Factor temporal: 1 antes de junio y Weibull desde junio."""
    dias = np.maximum(np.asarray(dias_desde_junio, dtype=float), 0.0)
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
            f"Parche de decaimiento desde junio no aplicado: '{etiqueta}' aparece "
            f"{cantidad} veces; se esperaba exactamente una coincidencia."
        )
    return source.replace(old, new, 1)


def parchear_modelo_decaimiento_junio(source: str) -> str:
    """Integra el techo 50 % + decaimiento desde junio, preservando enero–mayo."""

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

st.sidebar.markdown("**Decaimiento tardío (desde junio)**")
st.sidebar.info(
    "Desde el 1 de junio, los pulsos quedan limitados como máximo al 50 % "
    "del mayor pulso simulado entre enero y mayo. Desde ese techo, el límite "
    "continúa disminuyendo con el decaimiento tardío."
)
tau_decaimiento = st.sidebar.number_input(
    "Tau de decaimiento desde junio (días)",
    min_value=5.0,
    max_value=180.0,
    value=60.0,
    step=5.0,
    help="No afecta enero–mayo. Menor tau = caída más rápida desde el 1 de junio.",
)
beta_decaimiento = st.sidebar.number_input(
    "Forma beta del decaimiento tardío",
    min_value=0.3,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="beta=1 equivale a decaimiento exponencial desde junio.",
)
intensidad_decaimiento = st.sidebar.slider(
    "Intensidad del decaimiento tardío",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Controla cuánto sigue reduciéndose el techo de 50 % después del 1-jun.",
)
st.sidebar.caption(
    "Enero–mayo permanecen intactos. El 1-jun el techo es 50 % del máximo "
    "previo; luego techo(t)=0,50·Máx_pre·[(1−I)+I·exp(−(días/tau)^beta)]."
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

    # Control tardío: enero–mayo quedan exactamente sin modificar.
    df["EMERREL_ANTES_DECAIMIENTO_JUNIO"] = df["EMERREL"].copy()
    df["Dias_Desde_1Jun"] = 0.0
    df["Factor_Decaimiento_Junio"] = 1.0
    df["Techo_EMERREL_Junio"] = np.nan

    tau_d = max(float(tau_decaimiento) if tau_decaimiento is not None else 60.0, 1e-9)
    beta_d = max(float(beta_decaimiento) if beta_decaimiento is not None else 1.0, 1e-9)
    intensidad_d = float(np.clip(
        intensidad_decaimiento if intensidad_decaimiento is not None else 0.0,
        0.0,
        1.0,
    ))

    inicio_junio = pd.to_datetime({
        "year": df["Fecha"].dt.year,
        "month": np.full(len(df), 6),
        "day": np.ones(len(df), dtype=int),
    })
    mascara_junio = df["Fecha"] >= inicio_junio
    dias_junio = (df["Fecha"] - inicio_junio).dt.days.clip(lower=0).astype(float)

    # Máximo previo por campaña/año. En 2026, el máximo enero–mayo define 100 %
    # visual y el techo de junio arranca en exactamente 50 % de ese valor.
    max_pre_junio_por_anio = {}
    for anio in sorted(df["Fecha"].dt.year.dropna().unique()):
        mascara_pre = (df["Fecha"].dt.year == anio) & (df["Fecha"].dt.month < 6)
        max_pre = float(df.loc[mascara_pre, "EMERREL"].clip(lower=0.0).max()) if mascara_pre.any() else 0.0
        max_pre_junio_por_anio[int(anio)] = max_pre

    factor_junio = np.ones(len(df), dtype=float)
    factor_junio[mascara_junio.to_numpy()] = (
        (1.0 - intensidad_d)
        + intensidad_d * np.exp(
            -((dias_junio[mascara_junio].to_numpy() / tau_d) ** beta_d)
        )
    )

    techo_junio = np.full(len(df), np.nan, dtype=float)
    for pos, (_, fila) in enumerate(df.iterrows()):
        if bool(mascara_junio.iloc[pos]):
            max_pre = max_pre_junio_por_anio.get(int(fila["Fecha"].year), 0.0)
            techo_junio[pos] = 0.50 * max_pre * factor_junio[pos]

    df["Dias_Desde_1Jun"] = dias_junio
    df["Factor_Decaimiento_Junio"] = np.clip(factor_junio, 0.0, 1.0)
    df["Techo_EMERREL_Junio"] = techo_junio

    if mascara_junio.any():
        valores_originales = df.loc[mascara_junio, "EMERREL"].clip(lower=0.0).to_numpy()
        techos_activos = df.loc[mascara_junio, "Techo_EMERREL_Junio"].fillna(0.0).to_numpy()
        df.loc[mascara_junio, "EMERREL"] = np.minimum(
            valores_originales,
            techos_activos,
        )

    df["Tau_Decaimiento_Junio_d"] = tau_d
    df["Beta_Decaimiento_Junio"] = beta_d
    df["Intensidad_Decaimiento_Junio"] = intensidad_d
    df["Fraccion_Maxima_Junio"] = 0.50

    df["EMERAC"] = df["EMERREL"].cumsum()'''

    source = _reemplazar_unico(source, motor_old, motor_new, "motor desde junio")
    return source
