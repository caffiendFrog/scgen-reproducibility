import json
import os
import platform
import sys
import time

import scgen

# #region agent log
def _log_debug(message, data, hypothesis_id, run_id="pre-fix"):
    payload = {
        "sessionId": "debug-session",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "train_cvae.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    log_path = r"c:\Users\silly\GitHub\scgen-reproducibility\.cursor\debug.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
# #endregion

# #region agent log
_log_debug(
    "pre_scanpy_import",
    {
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    },
    "H1",
)
# #endregion
try:
    import scanpy as sc
except Exception as e:
    # #region agent log
    _log_debug(
        "scanpy_import_error",
        {"error_type": type(e).__name__, "error_message": str(e)},
        "H1",
    )
    # #endregion
    raise
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