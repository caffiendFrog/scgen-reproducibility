# Batch Size Configuration Locations

This document identifies all places in the codebase where batch size can be specified or modified.

## 1. ST-GAN Training (`code/st_gan.py`)

### Direct Assignment
- **Line 54**: `batch_size = 512`
  - This is the main batch size for ST-GAN training
  - Used in the training loop for generator and discriminator updates
  - Currently set to 512

### Usage
- **Line 58**: Used in `arch` dictionary: `"bsize": batch_size`
- **Line 296-298**: Used when sampling batches: `np.random.choice(range(len(train_real_stim)), size=eq, replace=False)`
  - Note: `eq` is the minimum of control/stimulated data sizes, not directly batch_size

---

## 2. VAE Training (`code/train_scGen.py`)

### Function Parameters (Default Values)

#### `test_train_whole_data_one_celltype_out()`
- **Line 14**: `batch_size=32` (function parameter default)
- **Line 56**: Passed to `network.train(..., batch_size=batch_size)`
- **Lines 283-290**: Function calls with `batch_size=32`:
  - Line 283: `test_train_whole_data_one_celltype_out("pbmc", ..., batch_size=32, ...)`
  - Line 285: `test_train_whole_data_one_celltype_out("hpoly", ..., batch_size=32, ...)`
  - Line 287: `test_train_whole_data_one_celltype_out("salmonella", ..., batch_size=32, ...)`
  - Line 289: `test_train_whole_data_one_celltype_out("species", ..., batch_size=32, ...)`

#### `test_train_whole_data_some_celltypes_out()`
- **Line 228**: `batch_size=32` (function parameter default)
- **Line 254**: Passed to `network.train(..., batch_size=batch_size)`
- **Lines 304, 316, 328**: Function calls with `batch_size=32`:
  - Line 304: First call with `batch_size=32`
  - Line 316: Second call with `batch_size=32`
  - Line 328: Third call with `batch_size=32`

#### `train_cross_study()`
- **Line 263**: `batch_size=32` (function parameter default)
- **Line 277**: Passed to `network.train(..., batch_size=batch_size)`
- **Line 291**: Function call: `train_cross_study("study", ..., batch_size=32, ...)`

---

## 3. VAE Model Class (`code/scgen/models/_vae.py`)

### Method Signature
- **Line 440**: `def train(..., batch_size=32, ...)`
  - Default batch size parameter in the `VAEArith.train()` method
  - Used in training loop at:
    - **Line 505**: `for lower in range(0, train_data.shape[0], batch_size):`
    - **Line 506**: `upper = min(lower + batch_size, train_data.shape[0])`
    - **Line 518**: Validation loop: `for lower in range(0, valid_data.shape[0], batch_size):`
    - **Line 519**: `upper = min(lower + batch_size, valid_data.shape[0])`
    - **Line 536**: Loss calculation: `train_loss / (train_data.shape[0] // batch_size)`

---

## 4. CVAE Model Class (`code/scgen/models/_cvae.py`)

### Method Signature
- **Line 294**: `def train(..., batch_size=32, ...)`
  - Default batch size parameter in the `CVAE.train()` method
  - Used in training loop at:
    - **Line 355**: `for lower in range(0, train_data.shape[0], batch_size):`
    - **Line 356**: `upper = min(lower + batch_size, train_data.shape[0])`
    - **Line 370**: Validation loop: `for lower in range(0, valid_data.shape[0], batch_size):`
    - **Line 371**: `upper = min(lower + batch_size, valid_data.shape[0])`

### Usage in `train_cvae.py`
- **Line 12**: `network.train(train, use_validation=True, valid_data=valid, n_epochs=100)`
  - **Note**: Batch size not explicitly specified, uses default of 32 from method signature

---

## 5. Mouse Atlas Training (`code/mouse_atlas.py`)

### Direct Assignment
- **Line 25**: `batch_size = 32`
  - Used in training function at:
    - **Line 164**: `input_matrix = train_data[0:train_data.shape[0] // batch_size * batch_size, :]`
    - **Line 165**: `for lower in range(0, input_matrix.shape[0], batch_size):`
    - **Line 166**: `upper = min(lower + batch_size, input_matrix.shape[0])`
    - **Line 171**: `size: batch_size` in feed_dict

---

## 6. Pancreas Training (`code/pancreas.py`)

### Direct Assignment
- **Line 27**: `batch_size = 32`
  - Used in training function at:
    - **Line 178**: `input_matrix = train_data[0:train_data.shape[0] // batch_size * batch_size, :]`
    - **Line 179**: `for lower in range(0, input_matrix.shape[0], batch_size):`
    - **Line 180**: `upper = min(lower + batch_size, input_matrix.shape[0])`
    - **Line 185**: `size: batch_size` in feed_dict

---

## 7. Hyperparameter Optimization (`code/scgen/hyperoptim.py`)

### Hyperparameter Search Space
- **Line 27**: `batch_size={{choice([32, 64, 128, 256])}}`
  - Used in hyperparameter optimization
  - Tests batch sizes: 32, 64, 128, 256

### Default Value
- **Line 52**: `batch_size = 256`
  - Default batch size for hyperparameter optimization runs

---

## 8. VAE Keras Model (`code/scgen/models/_vae_keras.py`)

### Method Signature
- **Line 431**: `batch_size=32` (parameter default)
- **Lines 516, 525**: Used in `model.fit()` calls:
  - `batch_size=batch_size` parameter

### Dynamic Batch Size
- **Line 160**: `batch_size = K.shape(mu)[0]`
  - Dynamic batch size calculation from tensor shape

---

## Summary Table

| File | Line | Type | Current Value | Usage |
|------|------|------|---------------|-------|
| `st_gan.py` | 54 | Direct assignment | 512 | ST-GAN training |
| `train_scGen.py` | 14 | Function parameter | 32 | VAE training (one celltype) |
| `train_scGen.py` | 228 | Function parameter | 32 | VAE training (some celltypes) |
| `train_scGen.py` | 263 | Function parameter | 32 | Cross-study training |
| `train_scGen.py` | 283-290 | Function calls | 32 | Multiple dataset training |
| `train_scGen.py` | 291 | Function call | 32 | Cross-study training |
| `train_scGen.py` | 304, 316, 328 | Function calls | 32 | Heldout celltype training |
| `_vae.py` | 440 | Method parameter | 32 | VAEArith.train() default |
| `_cvae.py` | 294 | Method parameter | 32 | CVAE.train() default |
| `train_cvae.py` | 12 | Implicit (uses default) | 32 | CVAE training |
| `mouse_atlas.py` | 25 | Direct assignment | 32 | Mouse atlas training |
| `pancreas.py` | 27 | Direct assignment | 32 | Pancreas training |
| `hyperoptim.py` | 27 | Search space | [32,64,128,256] | Hyperparameter search |
| `hyperoptim.py` | 52 | Default | 256 | Hyperparameter optimization |
| `_vae_keras.py` | 431 | Method parameter | 32 | Keras VAE training |

---

## Recommendations for Increasing Batch Size

To take advantage of more powerful GPUs (A100, V100), consider increasing:

1. **ST-GAN** (`st_gan.py` line 54): Increase from 512 to 1024 or 2048
2. **VAE Training** (`train_scGen.py`): Change all `batch_size=32` to `batch_size=64` or `128`
3. **Mouse Atlas** (`mouse_atlas.py` line 25): Increase from 32 to 64 or 128
4. **Pancreas** (`pancreas.py` line 27): Increase from 32 to 64 or 128
5. **CVAE** (`train_cvae.py` line 12): Add explicit `batch_size=64` or `128` parameter

**Note**: Monitor GPU memory usage when increasing batch sizes. A100 (40GB) can handle much larger batches than L4 (24GB).

