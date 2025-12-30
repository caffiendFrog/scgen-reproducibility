# Maximum Batch Sizes for g5.4xlarge Instance

## Instance Specifications
- **GPU**: 1x NVIDIA A10G with **24GB GPU memory**
- **vCPUs**: 16
- **System RAM**: 64 GiB

## Memory Considerations
- TensorFlow overhead: ~1-2GB
- Model weights: ~100-500MB (depending on input dimension)
- Optimizer states (Adam): ~2x model weights
- Batch normalization statistics: ~10-50MB
- **Available for activations**: ~20-21GB

## Model Architecture Analysis

### Typical Input Dimensions
- Gene expression data: 2000-5000 features (X_dim)
- Latent dimension (z_dim): 50-100
- Typical dataset sizes: 10,000-100,000 cells

---

## 1. ST-GAN (`code/st_gan.py`)

### Architecture
- **Generator (stim_ctrl)**: X_dim → 700 → 100 → 50 → 100 → 700 → X_dim
- **Generator (ctrl_stim)**: X_dim → 700 → 100 → 50 → 100 → 700 → X_dim
- **Discriminator (stimulated)**: X_dim → 700 → 100 → 1
- **Discriminator (control)**: X_dim → 700 → 100 → 1
- **Reconstruction losses**: 2x generators
- **Adversarial losses**: 4x (2 generators × 2 discriminators)

### Memory per Sample (estimated)
- Forward pass: ~(X_dim × 4 + 700 × 4 + 100 × 4 + 50 × 2) × 4 bytes
- For X_dim = 3000: ~50KB per sample
- Backward pass: ~2x forward = ~100KB per sample
- **Total per sample**: ~150KB

### Maximum Batch Size Calculation
- Available memory: ~20GB = 20,000MB
- Memory per sample: ~0.15MB
- **Maximum batch size**: ~133,000 samples
- **Conservative estimate (with safety margin)**: **2,048 - 4,096**

### Current Setting
- **Line 54**: `batch_size = 512`
- **Recommended maximum**: **2,048** (safe), **4,096** (aggressive)

---

## 2. VAE Training (`code/train_scGen.py` & `code/scgen/models/_vae.py`)

### Architecture
- **Encoder**: X_dim → 800 → 800 → z_dim (100)
- **Decoder**: z_dim (100) → 800 → 800 → X_dim
- **KL divergence loss**: z_dim
- **Reconstruction loss**: X_dim

### Memory per Sample (estimated)
- Forward pass: ~(X_dim × 2 + 800 × 4 + 100) × 4 bytes
- For X_dim = 3000: ~35KB per sample
- Backward pass: ~2x forward = ~70KB per sample
- **Total per sample**: ~105KB

### Maximum Batch Size Calculation
- Available memory: ~20GB = 20,000MB
- Memory per sample: ~0.105MB
- **Maximum batch size**: ~190,000 samples
- **Conservative estimate (with safety margin)**: **256 - 512**

### Current Settings
- **train_scGen.py Line 14**: `batch_size=32` (function parameter)
- **train_scGen.py Line 228**: `batch_size=32` (function parameter)
- **train_scGen.py Line 263**: `batch_size=32` (function parameter)
- **_vae.py Line 440**: `batch_size=32` (method default)
- **Recommended maximum**: **256** (safe), **512** (aggressive)

---

## 3. CVAE Training (`code/train_cvae.py` & `code/scgen/models/_cvae.py`)

### Architecture
- **Encoder**: (X_dim + 1) → 700 → 400 → z_dim (20)
- **Decoder**: z_dim (20) → X_dim
- **Conditional input**: 1 (label)
- **KL divergence loss**: z_dim
- **Reconstruction loss**: X_dim

### Memory per Sample (estimated)
- Forward pass: ~((X_dim + 1) × 2 + 700 + 400 + 20) × 4 bytes
- For X_dim = 3000: ~30KB per sample
- Backward pass: ~2x forward = ~60KB per sample
- **Total per sample**: ~90KB

### Maximum Batch Size Calculation
- Available memory: ~20GB = 20,000MB
- Memory per sample: ~0.09MB
- **Maximum batch size**: ~220,000 samples
- **Conservative estimate (with safety margin)**: **256 - 512**

### Current Settings
- **train_cvae.py Line 12**: Uses default `batch_size=32`
- **_cvae.py Line 294**: `batch_size=32` (method default)
- **Recommended maximum**: **256** (safe), **512** (aggressive)

---

## 4. Mouse Atlas Training (`code/mouse_atlas.py`)

### Architecture
- Similar to VAE: X_dim → 800 → 800 → z_dim (100) → 800 → 800 → X_dim
- Uses same VAE architecture

### Memory per Sample
- Same as VAE: ~105KB per sample

### Maximum Batch Size
- **Recommended maximum**: **256** (safe), **512** (aggressive)

### Current Setting
- **Line 25**: `batch_size = 32`
- **Recommended maximum**: **256** (safe), **512** (aggressive)

---

## 5. Pancreas Training (`code/pancreas.py`)

### Architecture
- Similar to VAE: X_dim → 800 → 800 → z_dim (100) → 800 → 800 → X_dim
- Uses same VAE architecture

### Memory per Sample
- Same as VAE: ~105KB per sample

### Maximum Batch Size
- **Recommended maximum**: **256** (safe), **512** (aggressive)

### Current Setting
- **Line 27**: `batch_size = 32`
- **Recommended maximum**: **256** (safe), **512** (aggressive)

---

## Summary Table

| Location | File | Line | Current | Safe Max | Aggressive Max | Notes |
|----------|------|------|---------|----------|----------------|-------|
| ST-GAN | `st_gan.py` | 54 | 512 | **2,048** | **4,096** | Most memory-intensive |
| VAE (one celltype) | `train_scGen.py` | 14 | 32 | **256** | **512** | Function parameter |
| VAE (some celltypes) | `train_scGen.py` | 228 | 32 | **256** | **512** | Function parameter |
| VAE (cross-study) | `train_scGen.py` | 263 | 32 | **256** | **512** | Function parameter |
| VAE method default | `_vae.py` | 440 | 32 | **256** | **512** | Method parameter |
| CVAE | `train_cvae.py` | 12 | 32 | **256** | **512** | Uses default |
| CVAE method default | `_cvae.py` | 294 | 32 | **256** | **512** | Method parameter |
| Mouse Atlas | `mouse_atlas.py` | 25 | 32 | **256** | **512** | Direct assignment |
| Pancreas | `pancreas.py` | 27 | 32 | **256** | **512** | Direct assignment |

---

## Recommendations

### Conservative Approach (Recommended)
1. **ST-GAN**: Increase from 512 to **2,048**
2. **VAE/CVAE**: Increase from 32 to **256**
3. **Mouse Atlas/Pancreas**: Increase from 32 to **256**

### Aggressive Approach (If you have memory issues, reduce)
1. **ST-GAN**: Increase from 512 to **4,096**
2. **VAE/CVAE**: Increase from 32 to **512**
3. **Mouse Atlas/Pancreas**: Increase from 32 to **512**

### Testing Strategy
1. Start with conservative values
2. Monitor GPU memory with `nvidia-smi` during training
3. If memory usage < 20GB, gradually increase batch size
4. If you get OOM errors, reduce by 25% and retry

### Expected Speedup
- **ST-GAN**: 2-4x speedup (512 → 2048)
- **VAE/CVAE**: 4-8x speedup (32 → 256)
- **Overall training time**: Should reduce from 20-40 hours to 5-10 hours

---

## Important Notes

1. **Input Dimension Dependency**: Actual maximum batch size depends on `X_dim` (number of genes). Larger `X_dim` = smaller max batch size.

2. **Dataset Size**: If your dataset has fewer samples than the maximum batch size, use the dataset size.

3. **Memory Fragmentation**: TensorFlow may not use all 24GB efficiently. Conservative estimates account for this.

4. **Mixed Precision**: Consider enabling mixed precision training (FP16) to potentially double batch sizes, but this requires code modifications.

5. **Gradient Accumulation**: If you need larger effective batch sizes, consider gradient accumulation instead of increasing batch size.

---

## How to Test

```bash
# Monitor GPU memory during training
watch -n 1 nvidia-smi

# Look for:
# - Memory-Usage: Should stay below 22GB
# - GPU-Util: Should be high (80-100%)
```

If memory usage approaches 24GB, reduce batch size. If it stays well below 20GB, you can increase batch size.

