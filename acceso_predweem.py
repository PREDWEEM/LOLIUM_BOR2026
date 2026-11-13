"""Puente al módulo común ``lolium_comun.acceso_predweem`` (repositorio PREDWEEM/LOLIUM-COMUN).

No editar acá: los cambios se hacen en LOLIUM-COMUN y se actualiza la versión en ``requirements.txt``.
"""
import sys

from lolium_comun import acceso_predweem as _modulo

sys.modules[__name__] = _modulo
