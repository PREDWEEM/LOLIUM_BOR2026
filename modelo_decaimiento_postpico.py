# -*- coding: utf-8 -*-
"""Decaimiento post-pico integrado al motor PREDWEEM Bordenave.

Este módulo NO altera los pesos de la ANN. Inyecta, de forma auditable, un
factor de decaimiento causal luego del primer pico validado del modelo y hace
que el EMERREL modificado sea el que continúa hacia EMERAC, Event-to-Event,
T50 y las métricas de validación.

Factor:
    F(t) = (1 - I) + I * exp(- (t / tau) ** beta)

con t = días desde el primer pico validado, tau > 0, beta > 0 e I en [0, 1].
F(0)=1 y, a largo plazo, F(t) -> 1-I. Por lo tanto, la intensidad controla qué
fracción máxima del potencial post-pico puede agotarse sin forzar a cero todos
los pulsos tardíos.

Los valores iniciales son experimentales y editables desde la interfaz:
    tau = 60 días, beta = 1.0, intensidad = 0.75.
Deben recalibrarse/validarse antes de considerarse parámetros definitivos.
"""

from __future__ import annotations

import numpy as np


TAU_DECAIMIENTO_DEFAULT = 60.0
BETA_DECAIMIENTO_DEFAULT = 1.0
INTENSIDAD_DECAIMIENTO_DEFAULT = 0.75


def factor_decaimiento_weibull(
    dias,
    tau: float = TAU_DECAIMIENTO_DEFAULT,
    beta: float = BETA_DECAIMIENTO_DEFAULT,
    intensidad: float = INTENSIDAD_DECAIMIENTO_DEFAULT,
):
    """Devuelve el factor multiplicativo post-pico en [0, 1]."""
    dias_arr = np.maximum(np.asarray(dias, dtype=float), 0.0)
    tau = max(float(tau), 1e-9)
    beta = max(float(beta), 1e-9)
    intensidad = float(np.clip(intensidad, 0.0, 1.0))
    return (1.0 - intensidad) + intensidad * np.exp(-((dias_arr / tau) ** beta))


def _reemplazar_unico(source: str, old: str, new: str, etiqueta: str) -> str:
    cantidad = source.count(old)
    if cantidad != 1:
        raise RuntimeError(
            f"Parche de decaimiento no aplicado: '{etiqueta}' aparece {cantidad} veces; "
            "se esperaba exactamente una coincidencia."
        )
    return source.replace(old, new, 1)


def parchear_modelo_decaimiento_postpico(source: str) -> str:
    """Integra el decaimiento en EMERREL antes de acumulación/validación."""

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

st.sidebar.markdown("**Decaimiento post-pico del modelo**")
tau_decaimiento = st.sidebar.number_input(
    "Tau de decaimiento (días)",
    min_value=5.0,
    max_value=180.0,
    value=60.0,
    step=5.0,
    help="Escala temporal del agotamiento post-pico. Menor tau = caída más rápida.",
)
beta_decaimiento = st.sidebar.number_input(
    "Forma beta del decaimiento",
    min_value=0.3,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="beta=1 corresponde a decaimiento exponencial; beta>1 acelera la caída tardía.",
)
intensidad_decaimiento = st.sidebar.slider(
    "Intensidad del decaimiento",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="0 desactiva el decaimiento; 1 permite que el factor tienda a cero a largo plazo.",
)
st.sidebar.caption(
    "Factor post-pico: (1−I) + I·exp[−(días/tau)^beta]. "
    "Parámetros experimentales; afectan EMERREL y por lo tanto también la validación."
)

residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 0)'''

    source = _reemplazar_unico(source, sidebar_old, sidebar_new, "controles de decaimiento")

    optimizer_call_old = '''                    exponente_kr=exponente_kr,
                    calentamiento_suelo=calentamiento_suelo,
                )'''
    optimizer_call_new = '''                    exponente_kr=exponente_kr,
                    calentamiento_suelo=calentamiento_suelo,
                    tau_decaimiento=tau_decaimiento,
                    beta_decaimiento=beta_decaimiento,
                    intensidad_decaimiento=intensidad_decaimiento,
                )'''
    source = _reemplazar_unico(
        source,
        optimizer_call_old,
        optimizer_call_new,
        "paso de decaimiento al optimizador",
    )

    main_call_old = '''        exponente_kr=exponente_kr,
        calentamiento_suelo=calentamiento_suelo,
    )'''
    main_call_new = '''        exponente_kr=exponente_kr,
        calentamiento_suelo=calentamiento_suelo,
        tau_decaimiento=tau_decaimiento,
        beta_decaimiento=beta_decaimiento,
        intensidad_decaimiento=intensidad_decaimiento,
    )'''
    source = _reemplazar_unico(
        source,
        main_call_old,
        main_call_new,
        "paso de decaimiento al motor principal",
    )

    motor_old = '''    df, idx_primer_pico = aplicar_filtro_primer_pico(df, umbral=UMBRAL_PRIMER_PICO)

    df["EMERAC"] = df["EMERREL"].cumsum()'''

    motor_new = '''    df, idx_primer_pico = aplicar_filtro_primer_pico(df, umbral=UMBRAL_PRIMER_PICO)

    # Decaimiento causal post-pico: forma parte del EMERREL operativo.
    df["EMERREL_ANTES_DECAIMIENTO"] = df["EMERREL"].copy()
    df["Dias_Desde_Pico"] = 0.0
    df["Factor_Decaimiento"] = 1.0

    if idx_primer_pico is not None:
        tau_d = max(float(tau_decaimiento) if tau_decaimiento is not None else 60.0, 1e-9)
        beta_d = max(float(beta_decaimiento) if beta_decaimiento is not None else 1.0, 1e-9)
        intensidad_d = float(np.clip(
            intensidad_decaimiento if intensidad_decaimiento is not None else 0.0,
            0.0,
            1.0,
        ))
        fecha_pico_decay = pd.Timestamp(df.loc[idx_primer_pico, "Fecha"])
        dias_decay = (df["Fecha"] - fecha_pico_decay).dt.days.clip(lower=0).astype(float)
        factor_decay = (
            (1.0 - intensidad_d)
            + intensidad_d * np.exp(-((dias_decay / tau_d) ** beta_d))
        )
        factor_decay.loc[df.index < idx_primer_pico] = 1.0

        df["Dias_Desde_Pico"] = dias_decay
        df["Factor_Decaimiento"] = np.clip(factor_decay, 0.0, 1.0)
        df["EMERREL"] = (
            df["EMERREL"] * df["Factor_Decaimiento"]
        ).clip(0.0, 1.0)

    df["Tau_Decaimiento_d"] = float(tau_decaimiento) if tau_decaimiento is not None else np.nan
    df["Beta_Decaimiento"] = float(beta_decaimiento) if beta_decaimiento is not None else np.nan
    df["Intensidad_Decaimiento"] = float(intensidad_decaimiento) if intensidad_decaimiento is not None else 0.0

    df["EMERAC"] = df["EMERREL"].cumsum()'''

    source = _reemplazar_unico(source, motor_old, motor_new, "motor post-pico")
    return source
