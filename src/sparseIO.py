import numpy as np
import scipy.sparse

def csrSave(array, filename):
    """
    Save a scipy-csr format sparse matrix to a filename
    array: the csr matrix to be saved
    filename: target filename

    """
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def csrLoad(filename):
    """
    Load a scipy-csr format sparse matrix from a filename
    filename: target filename
    return: scipy scr format sparse matrix

    """
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])