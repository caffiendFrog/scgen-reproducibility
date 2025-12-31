"""
Utility functions for handling sparse and dense arrays.
"""
import scipy.sparse as sparse
import numpy as np


def to_dense_array(X):
    """
    Convert a sparse matrix to a dense numpy array.
    If the input is already dense, return it as-is.
    
    Parameters
    ----------
    X : array-like, sparse matrix, or AnnData.X
        Input array that may be sparse or dense.
        
    Returns
    -------
    numpy.ndarray
        Dense numpy array.
    """
    if sparse.issparse(X):
        result = X.toarray()
    else:
        # Ensure it's a numpy array
        result = np.asarray(X)
    
    return result

