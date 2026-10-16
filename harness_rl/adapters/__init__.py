"""Built-in adapters. Importing this package imports every subpackage, and each subpackage registers its adapter with
harness_rl.core.registry at import. A subpackage is any directory here with an __init__.py."""

import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    if _m.ispkg:
        importlib.import_module(f"{__name__}.{_m.name}")
