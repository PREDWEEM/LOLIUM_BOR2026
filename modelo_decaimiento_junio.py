# -*- coding: utf-8 -*-
"""Decaimiento tardío de PREDWEEM Bordenave aplicado sólo desde junio.

Conserva exactamente la versión previa del modelo entre enero y mayo y atenúa
únicamente los pulsos tardíos a partir del 1 de junio.

Factor:
    F(t) = 1                                      si fecha < 1-jun
    F(t) = (1-I) + I*exp(-(t/tau)**beta)         si fecha >= 1-jun

con t = días transcurridos desde el 1 de junio. El 1-jun F=1, por lo que no
existe discontinuidad. La modificación actúa sobre EMERREL antes de EMERAC,
Event-to-Event, T50 y las métricas.
"""

from __future__ import annotations

import numpy as np


TAU_DECAIMIENTO_JUNIO_DEFAULT = 60.0
BETA_DECAIMIENTO_JUNIO_DEFAULT = 1.0
INTENSIDAD_DECAIMIENTO_JUNIO_DEFAULT = 0.75


def factor_decaimiento_desde_junio(
    dias_desde_junio,
    activo,
    tau: float = TAU_DECAIMIENTO_JUNIO_DEFAULT,
    beta: float = BETA_DECAIMIENTO_JUNIO_DEFAULT,
    intensidad: float = INTENSIDAD_DECAIMIENTO_JUNIO_DEFAULT,
):
    """Factor 1 antes de junio y Weibull desde junio en adelante."""
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
    """Integra el decaimiento sólo desde junio, preservando enero–mayo."""

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
    help="0 = sin decaimiento; 1 = el factor puede tender a cero a largo plazo.",
)
st.sidebar.caption(
    "Enero–mayo se conservan idénticos a la versión anterior. "
    "Desde el 1-jun: (1−I) + I·exp[−(días/tau)^beta]."
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

    # Decaimiento tardío: enero–mayo quedan exactamente sin modificar.
    df["EMERREL_ANTES_DECAIMIENTO_JUNIO"] = df["EMERREL"].copy()
    df["Dias_Desde_1Jun"] = 0.0
    df["Factor_Decaimiento_Junio"] = 1.0

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
    factor_junio = np.ones(len(df), dtype=float)
    factor_junio[mascara_junio.to_numpy()] = (
        (1.0 - intensidad_d)
        + intensidad_d * np.exp(
            -((dias_junio[mascara_junio].to_numpy() / tau_d) ** beta_d)
        )
    )

    df["Dias_Desde_1Jun"] = dias_junio
    df["Factor_Decaimiento_Junio"] = np.clip(factor_junio, 0.0, 1.0)
    df.loc[mascara_junio, "EMERREL"] = (
        df.loc[mascara_junio, "EMERREL"]
        * df.loc[mascara_junio, "Factor_Decaimiento_Junio"]
    ).clip(0.0, 1.0)
    df["Tau_Decaimiento_Junio_d"] = tau_d
    df["Beta_Decaimiento_Junio"] = beta_d
    df["Intensidad_Decaimiento_Junio"] = intensidad_d

    df["EMERAC"] = df["EMERREL"].cumsum()'''

    source = _reemplazar_unico(source, motor_old, motor_new, "motor desde junio")
    return source
