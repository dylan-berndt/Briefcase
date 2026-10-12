"""
utils.config.loadRunConfig, loaded straight from utils/config.py. Importing it as utils.config would run
utils/__init__.py, which pulls in torch models, transformers and spaCy; these scripts only need the config loader.
"""
import importlib.util
import os

_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils", "config.py")
_spec = importlib.util.spec_from_file_location("utilsConfigStandalone", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
loadRunConfig = _module.loadRunConfig
