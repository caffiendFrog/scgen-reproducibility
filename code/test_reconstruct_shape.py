"""
Minimal test to validate shape handling in reconstruct_whole_data.

This test verifies that:
1. Model initialization matches data dimensions
2. to_latent() correctly validates input shapes
3. Shape mismatches are caught early with clear error messages
"""

import numpy as np
import anndata
import scgen
from scipy import sparse


def test_reconstruct_shape():
    """Test that reconstruct_whole_data handles shapes correctly"""
    print("=" * 60)
    print("Testing shape validation in scGen reconstruction pipeline")
    print("=" * 60)
    
    # Create minimal test data
    n_cells = 100
    n_genes = 50
    
    np.random.seed(42)
    X = np.random.rand(n_cells, n_genes)
    obs = {
        "cell_type": ["A"] * 50 + ["B"] * 50,
        "condition": (["control"] * 25 + ["stimulated"] * 25) * 2
    }
    var = {"var_names": [f"Gene_{i}" for i in range(n_genes)]}
    
    adata = anndata.AnnData(X, obs=obs, var=var)
    
    print(f"\n1. Created test data: {adata.shape}")
    print(f"   - Cells: {adata.shape[0]}")
    print(f"   - Genes: {adata.shape[1]}")
    
    # Test that model initialization matches data
    print("\n2. Testing model initialization...")
    network = scgen.VAEArith(x_dimension=adata.X.shape[1], z_dimension=10)
    assert network.x_dim == adata.X.shape[1], \
        f"Model x_dim ({network.x_dim}) should match data genes ({adata.X.shape[1]})"
    print(f"   ✓ Model x_dim ({network.x_dim}) matches data genes ({adata.X.shape[1]})")
    
    # Test to_latent with correct shape
    print("\n3. Testing to_latent() with correct shape...")
    test_data = adata.X[:10]  # 10 cells, 50 genes
    latent = network.to_latent(test_data)
    assert latent.shape == (10, 10), f"Expected (10, 10), got {latent.shape}"
    print(f"   ✓ to_latent() correctly processed shape {test_data.shape} -> {latent.shape}")
    
    # Test to_latent with wrong shape (should raise error)
    print("\n4. Testing to_latent() with incorrect shape (should raise ValueError)...")
    try:
        wrong_data = np.random.rand(10, n_genes + 1)  # Wrong number of genes
        network.to_latent(wrong_data)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        print(f"   ✓ Correctly caught shape mismatch: {str(e)[:80]}...")
    except Exception as e:
        print(f"   ✗ Unexpected exception type: {type(e).__name__}: {e}")
        raise
    
    # Test reconstruct with correct shape
    print("\n5. Testing reconstruct() with correct shape...")
    rec_data = network.reconstruct(test_data, use_data=False)
    assert rec_data.shape == test_data.shape, \
        f"Reconstruction shape {rec_data.shape} should match input {test_data.shape}"
    print(f"   ✓ reconstruct() correctly processed shape {test_data.shape} -> {rec_data.shape}")
    
    # Test sparse matrix handling
    print("\n6. Testing sparse matrix handling...")
    sparse_X = sparse.csr_matrix(adata.X)
    sparse_adata = anndata.AnnData(sparse_X, obs=obs, var=var)
    assert sparse_adata.X.shape[1] == n_genes, \
        f"Sparse matrix should preserve gene dimension: {sparse_adata.X.shape[1]} != {n_genes}"
    print(f"   ✓ Sparse matrix preserves shape: {sparse_adata.shape}")
    
    print("\n" + "=" * 60)
    print("All shape tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_reconstruct_shape()

