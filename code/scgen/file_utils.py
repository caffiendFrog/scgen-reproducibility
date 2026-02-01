import os

import numpy as np
from scipy import sparse
import anndata


def to_dense(adata, copy_if_view=True):
    """
    Converts an AnnData object to dense format, handling views and sparse matrices.
    
    In modern anndata (post-2018), subsetting operations create views that reference
    the parent object. This function ensures that:
    1. If the object is a view, it is copied first (if copy_if_view=True)
    2. Sparse matrices are converted to dense numpy arrays
    3. Edge cases with nested views are handled
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to convert to dense format
    copy_if_view : bool, optional
        If True (default), copy the object if it's a view before converting.
        If False, attempt to convert in-place (may fail for views).
    
    Returns
    -------
    anndata.AnnData
        A new AnnData object with dense X matrix. If the input was not a view
        and X was already dense, returns the same object (unless copy was needed).
    
    Example
    -------
    >>> import anndata
    >>> import scgen
    >>> adata = anndata.read("data.h5ad")
    >>> # Create a view (subset)
    >>> view = adata[adata.obs["cell_type"] == "CD4T"]
    >>> # Convert to dense, handling the view properly
    >>> dense_adata = scgen.file_utils.to_dense(view)
    """
    # Check if adata is a view
    is_view = False
    try:
        # Modern anndata (>=0.7.0) has is_view attribute
        if hasattr(adata, 'is_view'):
            is_view = adata.is_view
        # Check for parent reference (alternative way to detect views)
        elif hasattr(adata, '_parent'):
            is_view = adata._parent is not None
    except (AttributeError, RuntimeError):
        # If we can't determine, assume it's not a view
        is_view = False
    
    # If it's a view and we should copy, do so first
    # This ensures we have an independent object before converting
    if is_view and copy_if_view:
        adata = adata.copy()
    
    # Convert sparse matrix to dense if needed
    if sparse.issparse(adata.X):
        # Convert sparse to dense
        dense_X = adata.X.toarray()
        # Create new AnnData with dense X, preserving all metadata
        # Use copy() method which handles all attributes properly
        result = adata.copy()
        result.X = dense_X
        return result
    else:
        # Already dense
        # If we copied because it was a view, return the copy
        # Otherwise, if X might still reference parent data, copy to be safe
        if is_view and copy_if_view:
            return adata
        
        # Check if X itself is a view (e.g., from slicing a numpy array)
        try:
            if hasattr(adata.X, 'base') and adata.X.base is not None:
                # X is a view, create a proper copy
                result = adata.copy()
                result.X = np.array(adata.X, copy=True)
                return result
        except (AttributeError, ValueError, TypeError):
            pass
        
        # If we get here, X is already dense and not a view
        # Return as-is (it's safe)
        return adata


def get_dense_X(adata, copy_if_view=True):
    """
    Extracts the dense X matrix from an AnnData object, handling views and sparse matrices.
    
    This is a convenience function for cases where you only need the X array,
    not the full AnnData object.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object
    copy_if_view : bool, optional
        If True (default), copy the object if it's a view before converting.
    
    Returns
    -------
    numpy.ndarray
        Dense numpy array of the X matrix
    
    Example
    -------
    >>> import anndata
    >>> import scgen
    >>> adata = anndata.read("data.h5ad")
    >>> view = adata[adata.obs["cell_type"] == "CD4T"]
    >>> dense_X = scgen.file_utils.get_dense_X(view)
    """
    dense_adata = to_dense(adata, copy_if_view=copy_if_view)
    return dense_adata.X


def ensure_dir_for_file(file_path):
    """
    Ensures the directory for a file path exists, creating it if necessary.

    Parameters
    ----------
    file_path : str
        Path to the file (can be absolute or relative)

    Returns
    -------
    str
        The original file_path (for chaining)

    Example
    -------
    >>> from scgen.file_utils import ensure_dir_for_file
    >>> file_path = ensure_dir_for_file("../data/reconstructed/VecArithm/file.h5ad")
    >>> adata.write(file_path)
    """
    file_dir = os.path.dirname(file_path)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    return file_path


def ensure_dir(dir_path):
    """
    Ensures a directory exists, creating it if necessary.

    Parameters
    ----------
    dir_path : str
        Path to the directory

    Returns
    -------
    str
        The original dir_path (for chaining)

    Example
    -------
    >>> from scgen.file_utils import ensure_dir
    >>> ensure_dir("../results/Figures/")
    """
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path
