import importlib
import os
import pkgutil
import pytest

MODULES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modules')

packages = [name for _, name, ispkg in pkgutil.iter_modules([MODULES_DIR]) if ispkg]

@pytest.mark.parametrize('package', packages)
def test_module_import(package):
    importlib.import_module(f"modules.{package}")
