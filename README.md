# scGen predicts single-cell perturbations.

<img align="center"  src="/sketch/sketch.png?raw=true">



This repository includes python scripts in [code](https://github.com/theislab/scGen/tree/master/code) and notebooks in the [Jupyter Notebooks](https://github.com/M0hammadL/scGen_reproducibility/tree/master/Jupyter%20Notebooks) folder to reproduce figures from the paper [(bioRxiv, 2018)](https://www.biorxiv.org/content/10.1101/478503v2) according to the table bellow.



figure       | notebook path     
---------------| ---------------
| [*Figure 2*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/Fig2.ipynb)| Jupyter Notebooks/Fig2.ipynb| 
| [*Figure 3*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/Fig3.ipynb)| Jupyter Notebooks/Fig3.ipynb| 
| [*Figure 4*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/Fig4.ipynb)| Jupyter Notebooks/Fig4.ipynb| 
| [*Figure 5*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/Fig5.ipynb)| Jupyter Notebooks/Fig5.ipynb| 
| [*Figure 6*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/Fig6.ipynb)| Jupyter Notebooks/Fig6.ipynb| 
| [*Supplemental Figure 1*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig1.ipynb)| Jupyter Notebooks/SupplFig1.ipynb| 
| [*Supplemental Figure 2*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig2.ipynb)| Jupyter Notebooks/SupplFig2.ipynb| 
| [*Supplemental Figure 4*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig4.ipynb)| Jupyter Notebooks/SupplFig4.ipynb| 
| [*Supplemental Figure 5*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig5.ipynb)| Jupyter Notebooks/SupplFig5.ipynb| 
| [*Supplemental Figure 6*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig6.ipynb)| Jupyter Notebooks/SupplFig6.ipynb| 
| [*Supplemental Figure 7*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig7.ipynb)| Jupyter Notebooks/SupplFig7.ipynb| 
| [*Supplemental Figure 8*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig8.ipynb)| Jupyter Notebooks/SupplFig8.ipynb| 
| [*Supplemental Figure 9*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig9.ipynb)| Jupyter Notebooks/SupplFig9.ipynb| 
| [*Supplemental Figure 10*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig10.ipynb)| Jupyter Notebooks/SupplFig10.ipynb| 
| [*Supplemental Figure 11*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig11.ipynb)| Jupyter Notebooks/SupplFig11.ipynb| 
| [*Supplemental Figure 12*](https://nbviewer.jupyter.org/github/M0hammadL/scGen_reproducibility/blob/master/Jupyter%20Notebooks/SupplFig12.ipynb)| Jupyter Notebooks/SupplFig12.ipynb| 

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)

### Environment Setup

#### macOS

1. Create and activate the conda environment:
   ```bash
   bash scripts/bootstrap_mac.sh
   conda activate scgen-repor-env
   ```

2. Create the symlink (replaces duplicate `scgen` directory):
   ```bash
   bash scripts/create_symlink_mac.sh
   ```

#### Windows

1. Create and activate the conda environment:
   ```powershell
   .\scripts\bootstrap_win.ps1
   conda activate scgen-repor-env
   ```
   Or use the convenience batch file:
   ```cmd
   scripts\bootstrap_win.bat
   ```

2. Create the symlink/junction (replaces duplicate `scgen` directory):
   ```powershell
   .\scripts\create_symlink_win.ps1
   ```
   **Note:** On Windows, if symbolic link creation fails (requires Developer Mode or admin privileges), the script will automatically create a directory junction instead, which works without special privileges.

### Verification

After setup, verify the environment works:

```bash
python -c "import scgen; import scanpy; import tensorflow; print('All imports successful!')"
```

## Getting Started

Once the environment is set up:

```bash
cd code/
python DataDownloader.py
python ModelTrainer.py all
```

Then you can run each notebook and reproduce the results.

**Note:** The `scgen` module is now available as a single source of truth in `code/scgen`, with a symlink/junction in `Jupyter Notebooks/scgen` for notebook compatibility.

### Troubleshooting

- **`get_version` import error**: If you encounter an error importing `get_version` when using `scgen`, you may need to install it separately or modify `code/scgen/__init__.py` to handle versioning differently. This does not affect the analysis functionality.
- **Windows symlink issues**: If symbolic link creation fails on Windows, the script automatically falls back to a directory junction, which works without special privileges.

All datasets are available in this drive [directory](https://drive.google.com/drive/folders/1v3qySFECxtqWLRhRTSbfQDFqdUCAXql3).
