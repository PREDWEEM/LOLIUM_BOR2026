# -*- coding: utf-8 -*-
"""Control de acceso de las apps PREDWEEM: ingreso con cuenta de Google y lista de usuarios aprobados.

Se activa solo cuando los secretos de la app (Streamlit Community Cloud → Settings → Secrets)
incluyen la sección ``[acceso]`` con ``activo = true`` y la sección ``[auth]`` de Google.
Mientras no estén configurados, la app funciona como siempre (sin login).

Secretos esperados::

    [auth]
    redirect_uri = "https://<app>.streamlit.app/oauth2callback"
    cookie_secret = "<texto aleatorio largo>"
    client_id = "<id de cliente OAuth de Google>"
    client_secret = "<secreto de cliente OAuth de Google>"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

    [acceso]
    activo = true
    administradores = ["chantreguillermo@gmail.com"]      # siempre habilitados
    lista_url = "https://docs.google.com/spreadsheets/d/e/.../pub?output=csv"  # planilla de aprobados
    # (o varias: lista_url = ["<planilla manual>", "<respuestas del formulario>"])
    contacto = "chantreguillermo@gmail.com"               # a quién se pide acceso
    formulario_url = "https://forms.gle/..."              # opcional: formulario de solicitud

La planilla publicada como CSV debe tener una columna ``email``; si además tiene una columna
``estado``, solo se habilitan las filas con ``aprobado`` (o ``sí``/``si``/``ok``).
"""

from __future__ import annotations

import csv
import io
import urllib.parse
import urllib.request

import streamlit as st

_ESTADOS_OK = {"aprobado", "aprobada", "si", "sí", "ok", "habilitado", "habilitada", "1", "true"}
_CONTACTO_POR_DEFECTO = "chantreguillermo@gmail.com"


def _config() -> dict:
    try:
        return dict(st.secrets.get("acceso", {}))
    except Exception:  # noqa: BLE001  (sin archivo de secretos)
        return {}


def emails_aprobados_desde_csv(texto: str) -> set[str]:
    """Emails habilitados en el CSV de la planilla (columna email y, opcional, estado)."""
    filas = list(csv.reader(io.StringIO(texto.lstrip("﻿"))))
    if not filas:
        return set()
    encabezado = [c.strip().lower() for c in filas[0]]
    col_email = next((i for i, c in enumerate(encabezado) if "mail" in c), None)
    col_estado = next((i for i, c in enumerate(encabezado) if c in ("estado", "aprobado", "habilitado")), None)
    datos = filas[1:] if col_email is not None else filas
    col_email = 0 if col_email is None else col_email
    aprobados = set()
    for fila in datos:
        if len(fila) <= col_email:
            continue
        email = fila[col_email].strip().lower()
        if "@" not in email:
            continue
        if col_estado is not None:
            estado = fila[col_estado].strip().lower() if len(fila) > col_estado else ""
            if estado not in _ESTADOS_OK:
                continue
        aprobados.add(email)
    return aprobados


@st.cache_data(ttl=300, show_spinner=False)
def _leer_planilla(url: str) -> set[str]:
    with urllib.request.urlopen(url, timeout=15) as respuesta:  # noqa: S310 (URL fija de los secretos)
        return emails_aprobados_desde_csv(respuesta.read().decode("utf-8", errors="replace"))


def _aprobados(cfg: dict) -> tuple[set[str], list[int], int]:
    """Devuelve (emails habilitados, números de lista que no se pudieron leer, cantidad de listas)."""
    emails = {str(e).strip().lower() for e in cfg.get("administradores", [])}
    emails |= {str(e).strip().lower() for e in cfg.get("emails", [])}
    urls = cfg.get("lista_url", [])
    urls = [urls] if isinstance(urls, str) else list(urls)
    urls = [str(u).strip() for u in urls if str(u).strip()]
    fallidas = []
    for n, url in enumerate(urls, start=1):
        try:
            emails |= _leer_planilla(url)
        except Exception:  # noqa: BLE001
            fallidas.append(n)
    return emails, fallidas, len(urls)


def _encabezado(nombre_app: str) -> None:
    st.set_page_config(page_title=f"PREDWEEM {nombre_app} · Ingreso", page_icon="🌾", layout="centered")
    st.markdown(
        f"<h2 style='margin-bottom:0'>🌾 PREDWEEM — {nombre_app}</h2>"
        "<p style='color:#4b5563;margin-top:.3rem'>Predicción de emergencia de <i>Lolium multiflorum</i></p>",
        unsafe_allow_html=True,
    )


def _conectado() -> bool:
    try:
        return bool(st.user.get("is_logged_in", False))
    except Exception:  # noqa: BLE001
        return False


def requerir_acceso(nombre_app: str) -> None:
    """Detiene la app hasta que el usuario ingresa con Google y su email está aprobado."""
    cfg = _config()
    if not cfg.get("activo", False):
        return  # control de acceso todavía no configurado: la app sigue abierta

    contacto = str(cfg.get("contacto", _CONTACTO_POR_DEFECTO))

    if not _conectado():
        _encabezado(nombre_app)
        st.info("El acceso a las aplicaciones PREDWEEM es para usuarios registrados. Ingresá con tu cuenta de Google.")
        st.button("Ingresar con Google", type="primary", icon=":material/login:", on_click=st.login, width="stretch")
        st.caption(
            "Si es la primera vez, al ingresar vas a poder solicitar el acceso. "
            "Solo se usa tu nombre y tu email para identificarte."
        )
        st.stop()

    email = str(st.user.get("email", "")).strip().lower()
    aprobados, fallidas, total = _aprobados(cfg)
    if email and email in aprobados:
        return

    _encabezado(nombre_app)
    nombre = st.user.get("name") or email
    formulario = str(cfg.get("formulario_url", "")).strip()
    if fallidas and len(fallidas) == total:
        st.error("No se pudo verificar la lista de usuarios habilitados. Probá de nuevo en unos minutos.")
        st.caption(f"Detalle para el administrador: no responde ninguna de las {total} listas de `lista_url`.")
    elif formulario:
        st.warning(f"Hola {nombre}: tu cuenta **{email}** todavía no está habilitada para usar PREDWEEM.")
        st.link_button("Solicitar acceso", formulario, type="primary", icon=":material/how_to_reg:", width="stretch")
        st.caption(
            f"Completá el formulario con esta misma cuenta de Google ({email}). "
            "Cuando se apruebe tu solicitud vas a poder entrar a todas las apps PREDWEEM; volvé a abrir esta página."
        )
    else:
        st.warning(f"Hola {nombre}: tu cuenta **{email}** todavía no está habilitada para usar PREDWEEM.")
        asunto = f"Solicitud de acceso a PREDWEEM {nombre_app}"
        cuerpo = (
            f"Hola, solicito acceso a la app PREDWEEM {nombre_app}.\n\n"
            f"Nombre: {nombre}\nEmail de la cuenta de Google: {email}\n"
            "Institución / empresa:\nLocalidad:\nUso previsto:\n"
        )
        enlace = f"mailto:{contacto}?subject={urllib.parse.quote(asunto)}&body={urllib.parse.quote(cuerpo)}"
        st.link_button("Solicitar acceso por email", enlace, type="primary", icon=":material/mail:", width="stretch")
        st.caption(f"La solicitud llega a {contacto}. Cuando se apruebe, volvé a abrir esta página.")
    if fallidas and len(fallidas) < total:
        st.caption(
            "Detalle para el administrador: no se pudo leer la lista n.º "
            + ", ".join(str(n) for n in fallidas) + f" de las {total} de `lista_url`."
        )
    st.button("Cerrar sesión", on_click=st.logout)
    st.stop()


def mostrar_sesion() -> None:
    """Muestra en la barra lateral el usuario conectado y el botón para cerrar sesión."""
    if not _config().get("activo", False) or not _conectado():
        return
    with st.sidebar:
        st.divider()
        st.caption(f"Sesión: {st.user.get('email', '')}")
        st.button("Cerrar sesión", on_click=st.logout, key="predweem_cerrar_sesion")
