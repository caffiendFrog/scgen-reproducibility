"""ScGen - Predicting single cell perturbations"""

import logging

# Enable TensorFlow 1.x compatibility mode for TensorFlow 2.x
# This must be imported before any TensorFlow operations
from . import tf_compat  # Auto-configures TF compatibility

from .models import *
from .read_load import load_file
try:
    from . import plotting
except Exception as exc:
    logging.getLogger(__name__).warning(
        "Optional dependency for plotting is not available; "
        "scgen.plotting will be disabled. Error: %s",
        exc,
    )
    plotting = None
from .file_utils import ensure_dir_for_file, ensure_dir, to_dense, get_dense_X
from .constants import DEFAULT_BATCH_SIZE, STGAN_BATCH_SIZE


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




