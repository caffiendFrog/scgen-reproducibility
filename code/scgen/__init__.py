"""ScGen - Predicting single cell perturbations"""

# Enable TensorFlow 1.x compatibility mode for TensorFlow 2.x
# This must be imported before any TensorFlow operations
from . import tf_compat  # Auto-configures TF compatibility

from .models import *
from .read_load import load_file
from . import plotting
from .file_utils import ensure_dir_for_file, ensure_dir, to_dense, get_dense_X


__author__ = ', '.join([
    'Mohammad  Lotfollahi',
    'Mohsen Naghipourfar'
])

__email__ = ', '.join([
    'Mohammad.lotfollahi@helmholtz-muenchen.de',
    'mohsen.naghipourfar@gmail.com'
])

from get_version import get_version
__version__ = get_version(__file__)
del get_version




