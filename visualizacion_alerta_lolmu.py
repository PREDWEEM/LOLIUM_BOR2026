"""Puente al módulo común ``lolium_comun.visualizacion_alerta_lolmu`` (repositorio PREDWEEM/LOLIUM-COMUN).

No editar acá: los cambios se hacen en LOLIUM-COMUN y se actualiza la versión en ``requirements.txt``.
"""
import sys

import alerta_lolmu  # noqa: F401  (fija los parámetros de esta app antes de cargar el módulo común)
from lolium_comun import visualizacion_alerta_lolmu as _modulo  # noqa: E402

sys.modules[__name__] = _modulo
