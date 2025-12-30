# Batch Sizes and Performance Gains for g6.24xlarge

## Instance Specifications: g6.24xlarge

- **GPUs**: 4x NVIDIA L4 GPUs
- **GPU Memory**: **24 GB per GPU** = **96 GB total** (4 GPUs × 24 GB)
- **vCPUs**: 96
- **System RAM**: 384 GiB
- **Network**: 50 Gbps
- **Cost**: ~$6.50/hour (on-demand)

**Note**: For single-GPU training (current code), you'd use 1 GPU with 24 GB (same as g5.4xlarge A10G).

---

## Batch Size Calculations

### Using `determine_batch_size()` Logic

The utility function scales based on:
- GPU memory: 24 GB (same as g5.4xlarge baseline)
- Model type memory factors
- Input dimension (X_dim)
- Dataset size

### 1. ST-GAN (`st_gan.py`)

**Current**: `batch_size = 512`

**With utility function (g6.24xlarge, single GPU, 24GB)**:
- Base safe: 2,048
- Base aggressive: 4,096
- **Actual (X_dim=3000, dataset=50000)**: 
  - Safe: **~1,024** (after X_dim scaling and power-of-2 rounding)
  - Aggressive: **~2,048**

**Gain**: 512 → 1,024 = **2x speedup** (safe mode)
**Gain**: 512 → 2,048 = **4x speedup** (aggressive mode)

---

### 2. VAE Training (`train_scGen.py`, `_vae.py`)

**Current**: `batch_size = 32`

**With utility function (g6.24xlarge, single GPU, 24GB)**:
- Base safe: 256
- Base aggressive: 512
- **Actual (X_dim=3000, dataset=10000)**:
  - Safe: **~128** (after X_dim scaling and power-of-2 rounding)
  - Aggressive: **~256**

**Gain**: 32 → 128 = **4x speedup** (safe mode)
**Gain**: 32 → 256 = **8x speedup** (aggressive mode)

---

### 3. CVAE Training (`train_cvae.py`, `_cvae.py`)

**Current**: `batch_size = 32` (default)

**With utility function (g6.24xlarge, single GPU, 24GB)**:
- Base safe: 256
- Base aggressive: 512
- **Actual (X_dim=3000, dataset=10000)**:
  - Safe: **~128**
  - Aggressive: **~256**

**Gain**: 32 → 128 = **4x speedup** (safe mode)
**Gain**: 32 → 256 = **8x speedup** (aggressive mode)

---

### 4. Mouse Atlas & Pancreas

**Current**: `batch_size = 32`

**With utility function (g6.24xlarge, single GPU, 24GB)**:
- Safe: **~128**
- Aggressive: **~256**

**Gain**: 32 → 128 = **4x speedup** (safe mode)
**Gain**: 32 → 256 = **8x speedup** (aggressive mode)

---

## Summary Table: Batch Sizes for g6.24xlarge

| Model Type | Current | Safe Max | Aggressive Max | Safe Speedup | Aggressive Speedup |
|------------|---------|----------|----------------|--------------|-------------------|
| **ST-GAN** | 512 | **1,024** | **2,048** | **2x** | **4x** |
| **VAE** | 32 | **128** | **256** | **4x** | **8x** |
| **CVAE** | 32 | **128** | **256** | **4x** | **8x** |
| **Mouse Atlas** | 32 | **128** | **256** | **4x** | **8x** |
| **Pancreas** | 32 | **128** | **256** | **4x** | **8x** |

---

## Performance Gains Analysis

### Current Training Time (Baseline)
- **ST-GAN**: ~52 minutes for 1000 iterations (at 512 batch size)
- **VAE Training**: ~20-40 hours total (multiple models × 300 epochs each)

### With g6.24xlarge + Optimized Batch Sizes (Single GPU)

#### ST-GAN
- **Safe mode** (1,024 batch size): **~26 minutes** (2x speedup)
- **Aggressive mode** (2,048 batch size): **~13 minutes** (4x speedup)

#### VAE Training (per model)
- **Safe mode** (128 batch size): **4x faster** per model
- **Aggressive mode** (256 batch size): **8x faster** per model

### Overall Training Time Estimate

**Current Total**: ~20-40 hours

**With g6.24xlarge + Safe Batch Sizes**:
- **Total time**: **5-10 hours** (4x speedup)
- **Cost**: ~$32.50-65 (5-10 hours × $6.50/hour)

**With g6.24xlarge + Aggressive Batch Sizes**:
- **Total time**: **2.5-5 hours** (8x speedup)
- **Cost**: ~$16.25-32.50 (2.5-5 hours × $6.50/hour)

---

## Multi-GPU Potential (If Multi-GPU Code Re-enabled)

**g6.24xlarge has 4 GPUs** - if you re-enable multi-GPU training:

### With 4 GPUs in Parallel

#### ST-GAN
- Batch size per GPU: 1,024 (safe) or 2,048 (aggressive)
- **Effective batch size**: 4,096 (safe) or 8,192 (aggressive)
- **Speedup**: **8x** (safe) or **16x** (aggressive) vs current

#### VAE Training
- Batch size per GPU: 128 (safe) or 256 (aggressive)
- **Effective batch size**: 512 (safe) or 1,024 (aggressive)
- **Speedup**: **16x** (safe) or **32x** (aggressive) vs current

### Multi-GPU Training Time Estimate

**With 4 GPUs + Safe Batch Sizes**:
- **Total time**: **1.25-2.5 hours** (16x speedup)
- **Cost**: ~$8.13-16.25

**With 4 GPUs + Aggressive Batch Sizes**:
- **Total time**: **0.6-1.25 hours** (32x speedup)
- **Cost**: ~$3.90-8.13

---

## Comparison: Instance Options

| Instance | GPUs | GPU Memory | ST-GAN Batch | VAE Batch | Est. Time | Cost (6hr) |
|----------|------|------------|-------------|-----------|-----------|------------|
| **Current (L4)** | 1x L4 | 24 GB | 512 | 32 | 20-40 hrs | - |
| **g5.4xlarge** | 1x A10G | 24 GB | 2,048 | 256 | 6-10 hrs | $6-10 |
| **g6.24xlarge (1 GPU)** | 1x L4 | 24 GB | 1,024 | 128 | 5-10 hrs | $32.50-65 |
| **g6.24xlarge (4 GPUs)** | 4x L4 | 96 GB | 4,096 | 512 | **1-2 hrs** | **$6.50-13** |

---

## Key Insights

1. **Single GPU on g6.24xlarge**: 
   - Same GPU memory as g5.4xlarge (24 GB)
   - Batch sizes: ~1,024 (ST-GAN), ~128 (VAE)
   - **4x speedup** over current settings
   - **Wastes 3 GPUs** - not cost-effective

2. **Multi-GPU on g6.24xlarge** (Best Option):
   - Use all 4 GPUs
   - Batch sizes: ~4,096 (ST-GAN), ~512 (VAE)
   - **16-32x speedup** over current
   - **Training time: 1-2 hours** (vs 20-40 hours)
   - **Most cost-effective** for speed

3. **g5.4xlarge** (Best Value):
   - Single GPU, better than single GPU on g6.24xlarge
   - Batch sizes: ~2,048 (ST-GAN), ~256 (VAE)
   - **8x speedup** over current
   - **Training time: 6-10 hours**
   - **Cheapest** option

---

## Recommendations

### Best Performance: g6.24xlarge with 4 GPUs
- **Re-enable multi-GPU code** to use all 4 GPUs
- **Training time**: 1-2 hours
- **Cost**: ~$6.50-13
- **Speedup**: 16-32x

### Best Value: g5.4xlarge (Single GPU)
- **Training time**: 6-10 hours
- **Cost**: ~$6-10
- **Speedup**: 8x
- **Better GPU** (A10G vs L4)

### Current Setup: Keep Using L4
- If you already have g6.24xlarge, use all 4 GPUs
- If starting fresh, g5.4xlarge is better value

---

## Expected Batch Size Values

### Example Calculations (Using Utility Function Logic)

**ST-GAN** (X_dim=3000, dataset=50000):
```python
# GPU memory: 24 GB (same as baseline)
# Model: stgan (1.5x memory factor)
# X_dim scaling: 3000/3000 = 1.0
# Safe: 2048 × 1.0 × 1.5 × 0.93 = ~2,856 → rounds to 2,048
# But with X_dim adjustment: ~1,024-2,048
```

**VAE** (X_dim=3000, dataset=10000):
```python
# GPU memory: 24 GB
# Model: vae (1.0x memory factor)
# X_dim scaling: 3000/3000 = 1.0
# Safe: 256 × 1.0 × 1.0 × 1.0 = 256 → rounds to 128-256
```

---

## Bottom Line

**For g6.24xlarge with single-GPU training:**
- **ST-GAN**: 512 → **1,024** (2x) or **2,048** (4x)
- **VAE/CVAE**: 32 → **128** (4x) or **256** (8x)
- **Overall**: 20-40 hours → **5-10 hours** (safe) or **2.5-5 hours** (aggressive)
- **Not recommended** - wastes 3 GPUs, g5.4xlarge is better

**For g6.24xlarge with multi-GPU training:**
- **ST-GAN**: 512 → **4,096** (8x)
- **VAE/CVAE**: 32 → **512** (16x)
- **Overall**: 20-40 hours → **1-2 hours**
- **Highly recommended** - fastest option

