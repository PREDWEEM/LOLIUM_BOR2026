"""Puente al módulo común ``lolium_comun.alerta_lolmu`` (repositorio PREDWEEM/LOLIUM-COMUN).

No editar acá: los cambios se hacen en LOLIUM-COMUN y se actualiza la versión en ``requirements.txt``. Este archivo indica
dónde están los parámetros de esta app (``alerta_lolmu_params.json``, en esta carpeta) y hace que ``import alerta_lolmu``
devuelva el módulo común.
"""
import os
import sys
from pathlib import Path

os.environ["LOLIUM_ALERTA_PARAMS"] = str(Path(__file__).with_name("alerta_lolmu_params.json"))

from lolium_comun import alerta_lolmu as _modulo  # noqa: E402

sys.modules[__name__] = _modulo
