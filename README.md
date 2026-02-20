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

**Supported platform: Linux only.** Setup has not been verified on macOS or Windows.

Both `environment.yml` and `conda-lock.yml` in this repo contain the **full set of dependencies** at versions verified to run all notebooks and scripts on Linux.

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda) on Linux
- For **exact reproducibility**: [conda-lock](https://github.com/conda/conda-lock) — `pip install conda-lock` or `conda install -c conda-forge conda-lock`

### 1. Create the conda environment

**Option A — conda-lock (recommended):** same package versions and builds every time.

```bash
conda-lock install -n scgen-repro-env conda-lock.yml
conda activate scgen-repro-env
```

**Option B — environment.yml:**

```bash
conda env create -f environment.yml
conda activate scgen-repro-env
```

If the env already exists and you want to update: `conda env update -f environment.yml --prune`.

### 2. Create the scgen symlink

So notebooks can import `code/scgen`:

```bash
bash scripts/create_symlink.sh
```

### Verification

After setup, verify the environment works:

```bash
python -c "import scgen; import scanpy; import tensorflow; print('All imports successful!')"
```

### Jupyter kernel check (conda)

If a notebook cannot import a dependency, confirm the kernel is using the
`scgen-repro-env` conda environment:

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

If the kernel is missing, register it:

```bash
python -m ipykernel install --user --name scgen-repro-env --display-name "Python (scgen-repro-env)"
```

## Getting Started

**Reproducibility checklist:** (1) Create the environment from `conda-lock.yml` or `environment.yml`. (2) Run `scripts/create_symlink.sh` so notebooks find `code/scgen`. (3) Download data and train models (below). (4) Run notebooks with the `scgen-repro-env` kernel.

Once the environment is set up:

```bash
cd code/
python DataDownloader.py
python ModelTrainer.py all
```

Then run the notebooks in `Jupyter Notebooks/` to reproduce the results. Use the kernel `Python (scgen-repro-env)` so all dependencies are available.

### Re-running reconstructions

Reconstruction outputs are cached by default. To force regeneration, pass
`--overwrite` to `ModelTrainer.py`:

```bash
python ModelTrainer.py all --overwrite
```

**Note:** The `scgen` module lives in `code/scgen`; `scripts/create_symlink.sh` creates a symlink at `Jupyter Notebooks/scgen` so notebooks can import it.

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
     To make the `LD_LIBRARY_PATH` change permanent for this env:
     ```bash
     mkdir -p $CONDA_PREFIX/etc/conda/activate.d
     printf 'export LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH}"\n' "$CONDA_PREFIX" \
       > $CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh
     ```
     **After adding this hook, deactivate and reactivate the env** (or open a new shell) so it takes effect:
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