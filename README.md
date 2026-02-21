# scGen predicts single-cell perturbations.

<img align="center"  src="/sketch/sketch.png?raw=true">



This repository includes python scripts in [code](https://github.com/theislab/scGen/tree/master/code) and notebooks in the [Jupyter Notebooks](https://github.com/caffiendFrog/scgen-reproducibility/tree/main/Jupyter%20Notebooks) folder to reproduce figures from the paper [(bioRxiv, 2018)](https://www.biorxiv.org/content/10.1101/478503v2) and related analyses. Notebooks (with links to this repository) are listed below.

| Figure / notebook | Path |
|-------------------|------|
| [*Figure 2*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/Fig2.ipynb) | Jupyter Notebooks/Fig2.ipynb |
| [*Figure 3*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/Fig3.ipynb) | Jupyter Notebooks/Fig3.ipynb |
| [*Figure 4*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/Fig4.ipynb) | Jupyter Notebooks/Fig4.ipynb |
| [*Figure 5*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/Fig5.ipynb) | Jupyter Notebooks/Fig5.ipynb |
| [*Figure 6*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/Fig6.ipynb) | Jupyter Notebooks/Fig6.ipynb |
| [*Supplemental Figure 1*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig1.ipynb) | Jupyter Notebooks/SupplFig1.ipynb |
| [*Supplemental Figure 2*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig2.ipynb) | Jupyter Notebooks/SupplFig2.ipynb |
| [*Supplemental Figure 4*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig4.ipynb) | Jupyter Notebooks/SupplFig4.ipynb |
| [*Supplemental Figure 5*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig5.ipynb) | Jupyter Notebooks/SupplFig5.ipynb |
| [*Supplemental Figure 6*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig6.ipynb) | Jupyter Notebooks/SupplFig6.ipynb |
| [*Supplemental Figure 7*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig7.ipynb) | Jupyter Notebooks/SupplFig7.ipynb |
| [*Supplemental Figure 8*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig8.ipynb) | Jupyter Notebooks/SupplFig8.ipynb |
| [*Supplemental Figure 9*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig9.ipynb) | Jupyter Notebooks/SupplFig9.ipynb |
| [*Supplemental Figure 10*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig10.ipynb) | Jupyter Notebooks/SupplFig10.ipynb |
| [*Supplemental Figure 11*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig11.ipynb) | Jupyter Notebooks/SupplFig11.ipynb |
| [*Supplemental Figure 12*](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/SupplFig12.ipynb) | Jupyter Notebooks/SupplFig12.ipynb |
| *Pancreas (Scanorama)* | [Jupyter Notebooks/pancreas-4-Scanorama.ipynb](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/pancreas-4-Scanorama.ipynb) |
| *MNN* | [Jupyter Notebooks/mnn.ipynb](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/mnn.ipynb) |
| *CCA* | [Jupyter Notebooks/cca.ipynb](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/cca.ipynb) |
| *BBKNN* | [Jupyter Notebooks/bbknn.ipynb](https://nbviewer.jupyter.org/github/caffiendFrog/scgen-reproducibility/blob/main/Jupyter%20Notebooks/bbknn.ipynb) | 

## Setup

**Verified platform: Linux only.** Setup has not been verified on macOS or Windows.

Both `environment.yml` and `conda-lock.yml` in this repo contain the **full set of dependencies** at versions verified to run all notebooks and scripts on Linux.

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda) on Linux
- For **exact reproducibility**: [conda-lock](https://github.com/conda/conda-lock) — `pip install conda-lock` or `conda install -c conda-forge conda-lock`

### First: initialize conda in your shell (one-time)

Do this once so `conda` and `conda activate` work in your terminal:

```bash
conda init bash
```

Then start a new shell or run `source ~/.bashrc` (use `source ~/.zshrc` if you use zsh). Without this, the `conda activate` steps below will not work.

## Notebook Quickstart (recommended order)

### 1. Change to the repository root first

Run setup commands from the repo root so `conda-lock.yml`, `environment.yml`, and `scripts/` paths resolve correctly:

```bash
cd /path/to/scgen-reproducibility
```

### 2. Create the conda environment

**Option A — conda-lock (recommended):** same package versions and builds every time.

```bash
cd /path/to/scgen-reproducibility
conda-lock install -n scgen-repro-env conda-lock.yml
conda activate scgen-repro-env
```

**Option B — environment.yml:**

```bash
cd /path/to/scgen-reproducibility
conda env create -f environment.yml
conda activate scgen-repro-env
```

If the env already exists and you want to update: `conda env update -f environment.yml --prune`.

### 3. Run notebook environment setup

**With `scgen-repro-env` activated**, run (from repo root):

```bash
bash scripts/setup_notebook_environment.sh
```

This script is idempotent and performs notebook-specific setup in one place:
- creates `Jupyter Notebooks/scgen` -> `code/scgen`
- installs an IPython startup hook used by Jupyter kernels
- installs env-local conda activation/deactivation hooks for `LD_LIBRARY_PATH`
- registers `Python (scgen-repro-env)` in Jupyter kernelspecs
- bootstraps Scanorama for `pancreas-4-Scanorama.ipynb`:
  - clones `https://github.com/brianhie/scanorama.git` to a sibling `../scanorama` (if missing)
  - creates `../scanorama/conf/4panc.txt`
  - creates `../scanorama/bin/4panc.py`

Optional flags:

```bash
# Skip Scanorama bootstrap
bash scripts/setup_notebook_environment.sh --skip-scanorama

# Override clone target and overwrite existing 4panc files
bash scripts/setup_notebook_environment.sh --scanorama-dir /custom/path/scanorama --force-scanorama-files
```

After running it, reactivate the env (or open a new shell) and restart notebook kernels:

```bash
conda deactivate
conda activate scgen-repro-env
```

### Verification

After setup, verify imports and kernel registration.

From the repo root (Python import check):

```bash
cd code && python -c "import scgen; import scanpy; import tensorflow; print('All imports successful')"
```

Kernel registration check:

```bash
conda activate scgen-repro-env
python -c "import sys; print(sys.executable)"
jupyter kernelspec list
```

In a notebook cell:

```python
import sys
print(sys.executable)
```

If the kernel is missing, run `bash scripts/setup_notebook_environment.sh` again (or register manually with `python -m ipykernel install --user --name scgen-repro-env --display-name "Python (scgen-repro-env)"`).

## Data + Model Artifacts (for full notebook reproduction)

**Reproducibility checklist:** (1) `cd` to repo root. (2) Create + activate `scgen-repro-env`. (3) Run `scripts/setup_notebook_environment.sh`. (4) Confirm with `jupyter kernelspec list` and import checks. (5) Download data/train models. (6) Run notebooks with kernel `Python (scgen-repro-env)`.

### Scanorama bootstrap (automated by setup script)

For [pancreas-4-Scanorama](Jupyter%20Notebooks/pancreas-4-Scanorama.ipynb), `scripts/setup_notebook_environment.sh` now handles the required Scanorama bootstrap by default (sibling clone + `conf/4panc.txt` + `bin/4panc.py`).

If cloning was skipped (e.g., no network) or you used `--skip-scanorama`, run one of:

```bash
# Re-run just with default behavior
bash scripts/setup_notebook_environment.sh

# Or do it manually:
cd /path/to/parent   # parent of scgen-reproducibility
git clone https://github.com/brianhie/scanorama.git scanorama
```

For troubleshooting (conda install, dependencies, paths), see [notes/pancreas-4-scanorama-compatibility.md](notes/pancreas-4-scanorama-compatibility.md).

### Download data and train models

Once the environment setup script has completed:

```bash
cd code/
python DataDownloader.py
python ModelTrainer.py all
```

**Before running the notebooks**, confirm that the expected data files exist. The notebooks expect the following (paths relative to the repo root). If any are missing, the training pipeline may have been interrupted—for example by a session timeout or resource limit—before it finished. In that case, re-run from the repo root: `cd code && python DataDownloader.py && python ModelTrainer.py all`.

**Raw inputs** (downloaded by `DataDownloader.py`):  
`data/train_pbmc.h5ad`, `data/valid_pbmc.h5ad`, `data/train_hpoly.h5ad`, `data/valid_hpoly.h5ad`, `data/train_salmonella.h5ad`, `data/valid_salmonella.h5ad`, `data/train_species.h5ad`, `data/valid_species.h5ad`, `data/train_study.h5ad`, `data/valid_study.h5ad`, `data/train_zheng.h5ad`, `data/pancreas.h5ad`, `data/MouseAtlas.subset.h5ad`.

**Reconstructed / model outputs** (produced by `ModelTrainer.py all` or batch-correction notebooks):  
- **scGen:** `data/reconstructed/scGen/pbmc.h5ad`, `hpoly.h5ad`, `salmonella.h5ad`, `pancreas.h5ad`, `mouse_atlas.h5ad`  
- **Other models:** `data/reconstructed/PCAVecArithm/PCA_pbmc.h5ad`, `PCA_CD4T.h5ad`, `PCA_CD4T_biased.h5ad`; `data/reconstructed/VecArithm/VecArithm_CD4T.h5ad`; `data/reconstructed/CVAE/CVAE_CD4T.h5ad`; `data/reconstructed/CGAN/cgan_cd4t.h5ad`  
- **Batch correction (SupplFig10):** `data/bbknn.h5ad`, `data/cca.h5ad`, `data/mnn.h5ad`, `data/scanorama.h5ad` (these are produced by the bbknn, cca, mnn, and pancreas-4-Scanorama notebooks respectively if you run them first).

Not every notebook uses every file; the list above is the full set that at least one notebook expects.

Then run the notebooks in `Jupyter Notebooks/` to reproduce the results. Use the kernel `Python (scgen-repro-env)` so all dependencies are available.

### Re-running reconstructions

Reconstruction outputs are cached by default. To force regeneration, pass
`--overwrite` to `ModelTrainer.py`:

```bash
python ModelTrainer.py all --overwrite
```

**Note:** The `scgen` module lives in `code/scgen`; `scripts/setup_notebook_environment.sh` sets up the `Jupyter Notebooks/scgen` symlink, installs an IPython startup hook, configures env-local `LD_LIBRARY_PATH` hooks, registers the active environment as a Jupyter kernel, and bootstraps Scanorama files used by `pancreas-4-Scanorama.ipynb`.

### Troubleshooting

- **AWS GPU setup (EC2/SageMaker)**: TensorFlow will only use GPU if the host has an NVIDIA driver and the CUDA runtime libraries are available to the process.
  1. **Verify GPU + driver** (host-level):
     ```bash
     nvidia-smi
     ```
     If this fails, install the AWS-provided NVIDIA driver for your instance type (Deep Learning AMI or the SageMaker GPU base images already include it).
  2. **Ensure CUDA runtime libs are in your env**: The locked environment and `environment.yml` include compatible `cudatoolkit` and `cudnn`; if you installed from the lock file they are already present. Otherwise (e.g. custom env):
     ```bash
     conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
     export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
     ```
     To make `LD_LIBRARY_PATH` persistent for this env, prefer rerunning:
     ```bash
     bash scripts/setup_notebook_environment.sh
     ```
     **After updating hooks, deactivate and reactivate the env** (or open a new shell) so it takes effect:
     ```bash
     conda deactivate
     conda activate scgen-repro-env
     ```
  3. **(Optional) TensorRT**: Only needed for TF-TRT optimizations. If you want to remove TensorRT warnings:
     ```bash
     conda install -c conda-forge tensorrt
     ```
  4. **Verify TensorFlow sees the GPU**:
     ```bash
     python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
     ```
     If the list is empty, the driver or CUDA libs are still missing.

- **`get_version` import error**: If you encounter an error importing `get_version` when using `scgen`, you may need to install it separately or modify `code/scgen/__init__.py` to handle versioning differently. This does not affect the analysis functionality.
- **Linux/SageMaker `CXXABI_1.3.15` error**: This means the system `libstdc++.so.6` is older than what `matplotlib` (via `scanpy`) was built against. Ensure the environment provides a newer `libstdc++` and that it is picked first:
  ```bash
  conda install -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
  ```
  Then re-run `python -c "import matplotlib; import scanpy"`.

All datasets are available in this drive [directory](https://drive.google.com/drive/folders/1v3qySFECxtqWLRhRTSbfQDFqdUCAXql3).
`MouseAtlas.subset.h5ad` is now hosted as a Google Drive file and is downloaded from
https://drive.google.com/file/d/1IiLFYEs4a8OS2nqT4FSk5BsB3UO3UHPZ/view?usp=drive_link.