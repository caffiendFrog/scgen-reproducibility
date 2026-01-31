import ctypes
import json
import os
import platform
import sys
import time

import scgen


conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
conda_libstdcxx = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
if os.path.isfile(conda_libstdcxx):
    try:
        ctypes.CDLL(conda_libstdcxx, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise ImportError(
            "ctypes.CDLL is required for visualization utilities. "
        ) from e
try:
    import scanpy as sc
except Exception as e:
    raise ImportError(
        "scanpy is required for visualization utilities. "
        "Install scanpy (and matplotlib) or avoid plotting functions."
    ) from e
import numpy as np
from scgen.file_utils import ensure_dir_for_file

train = sc.read("../data/train_pbmc.h5ad")
valid = sc.read("../data/valid_pbmc.h5ad")
train = train[~((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "stimulated"))]
valid = valid[~((valid.obs["cell_type"] == "CD4T") & (valid.obs["condition"] == "stimulated"))]
z_dim = 20
network = scgen.CVAE(x_dimension=train.X.shape[1], z_dimension=z_dim, alpha=0.1, model_path="../models/CVAE/pbmc/all/models/scgen")
network.train(train, use_validation=True, valid_data=valid, n_epochs=100)
labels, _ = scgen.label_encoder(train)
train = sc.read("../data/train_pbmc.h5ad")
CD4T = train[train.obs["cell_type"] == "CD4T"]
unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
fake_labels = np.ones((len(unperturbed_data), 1))
predicted_cells = network.predict(unperturbed_data, fake_labels)
adata = sc.AnnData(predicted_cells, obs={"condition": ["pred"]*len(fake_labels)})
adata.var_names = CD4T.var_names
all_adata = CD4T.concatenate(adata)
all_adata.write(ensure_dir_for_file("../data/reconstructed/CVAE_CD4T.h5ad"))