"""ScGen - Predicting single cell perturbations"""

from .models import *
from .read_load import load_file
from . import plotting


__author__ = ', '.join([
    'Mohammad  Lotfollahi',
    'Mohsen Naghipourfar'
])

__email__ = ', '.join([
    'Mohammad.lotfollahi@helmholtz-muenchen.de',
    'mohsen.naghipourfar@gmail.com'
])

try:
    from get_version import get_version
    __version__ = get_version(__file__)
    del get_version
except Exception:
    __version__ = "0.0.0-sagemaker"




