# -*- coding: utf-8 -*-
"""
PREDWEEM LOLIUM BORDENAVE 2026 — app_emergenciacombinado.py

Wrapper operativo para aplicar una modificación puntual a la app combinada:

    x_cobertura = [0, 30, 70, 100]
    ke_val = np.interp(cobertura_pct, x_cobertura, [1.60, 0.80, 0.25, 0.10])
    mod_termico = np.interp(cobertura_pct, x_cobertura, [0.90, 0.85, 0.80, 0.75])

Motivo:
    Evitar reescribir manualmente todo el archivo original y aplicar el cambio
    de forma auditable sobre el blob fuente previo.
"""

from __future__ import annotations

import base64
import json
import urllib.request


REPO = "PREDWEEM/LOLIUM_BOR2026"
BLOB_SHA_ORIGINAL = "fd8d36a19ec8eedcfc23820f60a67da44754007e"


OLD_BLOCK = """x_cobertura = [0, 30, 70, 100]
            ke_val = float(np.interp(cobertura_pct, x_cobertura, [0.85, 0.50, 0.25, 0.10]))
            mod_termico = float(np.interp(cobertura_pct, x_cobertura, [1.00, 0.95, 0.90, 0.80]))"""

NEW_BLOCK = """x_cobertura = [0, 30, 70, 100]
            ke_val = float(np.interp(cobertura_pct, x_cobertura, [1.60, 0.80, 0.25, 0.10]))
            mod_termico = float(np.interp(cobertura_pct, x_cobertura, [0.90, 0.85, 0.80, 0.75]))"""


def cargar_blob_original() -> str:
    url = f"https://api.github.com/repos/{REPO}/git/blobs/{BLOB_SHA_ORIGINAL}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return base64.b64decode(payload["content"]).decode("utf-8")


source = cargar_blob_original()
if OLD_BLOCK not in source:
    raise RuntimeError(
        "No se encontró el bloque original de cobertura en el blob fuente. "
        "No se aplicó ninguna modificación para evitar cambios ambiguos."
    )

source = source.replace(OLD_BLOCK, NEW_BLOCK, 1)
compiled = compile(source, "app_emergenciacombinado.py::patched", "exec")
exec(compiled, globals(), globals())
