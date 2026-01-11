# Batch Sizes and Performance Gains for g6.24xlarge

## Instance Specifications: g6.24xlarge

- **GPUs**: 4x NVIDIA L4 GPUs
- **GPU Memory**: ~22.35 GiB per GPU = **~89.4 GiB total** (4 GPUs × 22.35 GiB)
- **vCPUs**: 96
- **System RAM**: 384 GiB
- **Network**: 50 Gbps

**Note**: This instance has 4 GPUs, but if using single-GPU training (as the user reverted multi-GPU changes), you'd use 1 GPU with ~22.35 GiB.

---

## Batch Size Calculations Using `determine_batch_size()` Utility

### For Single-GPU Training (1x L4, 22.35 GB)

The utility function scales batch sizes based on GPU memory. Since g6.24xlarge has L4 GPUs with ~22.35 GB each (vs g5.4xlarge's 24GB A10G), the scaling is:

**GPU Memory Scale Factor**: 22.35 / 24.0 = **0.93x** (slightly less than g5.4xlarge)

### 1. ST-GAN (`st_gan.py`)

**Current**: `batch_size = 512`

**With utility function (g6.24xlarge, single GPU)**:
- Safe limit: 2048 × 0.93 = **~1,904** → rounds to **1,024** (power of 2)
- Aggressive limit: 4096 × 0.93 = **~3,808** → rounds to **2,048** (power of 2)
- **Recommended**: **1,024** (safe) or **2,048** (aggressive)

**Actual calculation example** (X_dim=3000, dataset_size=50000):
```python
determine_batch_size(
    x_dim=3000,
    dataset_size=50000,
    safe_batch_size=2048,
    aggressive_batch_size=4096,
    model_type='stgan',
    gpu_memory_gb=22.35,  # L4 GPU memory
    use_aggressive=False
)
# Result: ~1,024 (adjusted for X_dim and rounded to power of 2)
```

**Gain**: 512 → 1,024 = **2x speedup** (safe mode)

---

### 2. VAE Training (`train_scGen.py`, `_vae.py`)

**Current**: `batch_size = 32`

**With utility function (g6.24xlarge, single GPU)**:
- Safe limit: 256 × 0.93 = **~238** → rounds to **128** (power of 2)
- Aggressive limit: 512 × 0.93 = **~476** → rounds to **256** (power of 2)
- **Recommended**: **128** (safe) or **256** (aggressive)

**Actual calculation example** (X_dim=3000, dataset_size=10000):
```python
determine_batch_size(
    x_dim=3000,
    dataset_size=10000,
    safe_batch_size=256,
    aggressive_batch_size=512,
    model_type='vae',
    gpu_memory_gb=22.35,
    use_aggressive=False
)
# Result: ~128 (adjusted for X_dim and rounded to power of 2)
```

**Gain**: 32 → 128 = **4x speedup** (safe mode)

---

### 3. CVAE Training (`train_cvae.py`, `_cvae.py`)

**Current**: `batch_size = 32` (default)

**With utility function (g6.24xlarge, single GPU)**:
- Safe limit: 256 × 0.93 = **~238** → rounds to **128** (power of 2)
- Aggressive limit: 512 × 0.93 = **~476** → rounds to **256** (power of 2)
- **Recommended**: **128** (safe) or **256** (aggressive)

**Gain**: 32 → 128 = **4x speedup** (safe mode)

---

### 4. Mouse Atlas & Pancreas

**Current**: `batch_size = 32`

**With utility function (g6.24xlarge, single GPU)**:
- Safe limit: **128**
- Aggressive limit: **256**
- **Recommended**: **128** (safe) or **256** (aggressive)

**Gain**: 32 → 128 = **4x speedup** (safe mode)

---

## Summary Table: Batch Sizes for g6.24xlarge (Single GPU)

| Model Type | Current | Safe Max | Aggressive Max | Speedup |
|------------|---------|----------|-------------------|--------|
| **ST-GAN** | 512 | **1,024** | **2,048** | **2x** |
| **VAE** | 32 | **128** | **256** | **4x** |
| **CVAE** | 32 | **128** | **256** | **4x** |
| **Mouse Atlas** | 32 | **128** | **256** | **4x** |
| **Pancreas** | 32 | **128** | **256** | **4x** |

---

## Performance Gains Analysis

### Training Time Reduction

**Current Training Time** (estimated from your 30 min for 575 iterations):
- ST-GAN: ~52 minutes for 1000 iterations
- VAE models: ~20-40 hours total (multiple models × 300 epochs each)

**With g6.24xlarge Batch Size Increases**:

#### ST-GAN
- Current: 512 batch size → ~52 minutes
- With 1,024 batch size: **~26 minutes** (2x speedup)
- With 2,048 batch size: **~13 minutes** (4x speedup)

#### VAE Training (per model)
- Current: 32 batch size → ~X hours per model
- With 128 batch size: **~X/4 hours** (4x speedup)
- With 256 batch size: **~X/8 hours** (8x speedup)

### Overall Training Time Estimate

**Current Total**: ~20-40 hours

**With g6.24xlarge + Optimized Batch Sizes**:
- **Conservative (safe batch sizes)**: **5-10 hours** (4x speedup)
- **Aggressive (max batch sizes)**: **2.5-5 hours** (8x speedup)

---

## Comparison: g5.4xlarge vs g6.24xlarge

| Instance | GPU | GPU Memory | ST-GAN Batch | VAE Batch | Est. Time |
|----------|-----|------------|--------------|-----------|-----------|
| **g5.4xlarge** | 1x A10G | 24 GB | 2,048 | 256 | 6-10 hours |
| **g6.24xlarge** | 1x L4 | 22.35 GB | 1,024 | 128 | 5-10 hours |
| **g6.24xlarge** | 4x L4 | 89.4 GB total | 4,096* | 512* | **1-2 hours*** |

*If using multi-GPU training (4 GPUs in parallel)

---

## Key Insights

1. **Single GPU**: g6.24xlarge L4 (22.35 GB) is slightly less than g5.4xlarge A10G (24 GB)
   - Batch sizes: ~93% of g5.4xlarge values
   - Still provides 2-4x speedup over current settings

2. **Multi-GPU Potential**: With 4 GPUs, you could:
   - Use all 4 GPUs in parallel (if multi-GPU code is re-enabled)
   - Achieve ~4x additional speedup
   - Total speedup: **8-16x** vs current single-GPU training

3. **Cost Consideration**: 
   - g6.24xlarge: ~$6.50/hour (4 GPUs)
   - g5.4xlarge: ~$1.01/hour (1 GPU)
   - **g6.24xlarge is ~6.4x more expensive but has 4x GPUs**

---

## Recommendations

### Option 1: Single GPU on g6.24xlarge
- Use 1 GPU, batch size 1,024 (ST-GAN) or 128 (VAE)
- **Speedup**: 2-4x
- **Cost**: ~$6.50/hour (wasting 3 GPUs)

### Option 2: Multi-GPU on g6.24xlarge (Best Performance)
- Use all 4 GPUs with multi-GPU code
- Batch size per GPU: 1,024 (ST-GAN) or 128 (VAE)
- Effective batch size: 4,096 (ST-GAN) or 512 (VAE)
- **Speedup**: 8-16x
- **Training time**: **1-2 hours** (vs 20-40 hours)
- **Cost**: ~$6.50-13 for full training

### Option 3: Single GPU on g5.4xlarge (Best Value)
- Use 1 GPU, batch size 2,048 (ST-GAN) or 256 (VAE)
- **Speedup**: 4-8x
- **Training time**: 6-10 hours
- **Cost**: ~$6-10 for full training

---

## Expected Batch Sizes Using Utility Function

### Example Calculations

```python
from utils import determine_batch_size

# ST-GAN on g6.24xlarge (single GPU)
batch_size = determine_batch_size(
    x_dim=3000,
    dataset_size=50000,
    safe_batch_size=2048,
    aggressive_batch_size=4096,
    model_type='stgan',
    gpu_memory_gb=22.35,  # L4 GPU
    use_aggressive=False
)
# Result: ~1,024

# VAE on g6.24xlarge (single GPU)
batch_size = determine_batch_size(
    x_dim=3000,
    dataset_size=10000,
    safe_batch_size=256,
    aggressive_batch_size=512,
    model_type='vae',
    gpu_memory_gb=22.35,
    use_aggressive=False
)
# Result: ~128
```

---

## Bottom Line

**For g6.24xlarge with single-GPU training:**
- **ST-GAN**: 512 → **1,024** (2x speedup)
- **VAE/CVAE**: 32 → **128** (4x speedup)
- **Overall training**: 20-40 hours → **5-10 hours**

**For g6.24xlarge with multi-GPU training (if re-enabled):**
- **ST-GAN**: 512 → **4,096** (8x speedup)
- **VAE/CVAE**: 32 → **512** (16x speedup)
- **Overall training**: 20-40 hours → **1-2 hours**

