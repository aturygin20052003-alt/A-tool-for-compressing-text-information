from scipy.sparse import bsr_matrix
import numpy as np

def to_bsr(matrix: np.ndarray, b:int=1):
    """
    Преобразует плотную матрицу в блочно-разреженный формат
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html
    """
    
    if matrix.shape[0] % b != 0 or matrix.shape[1] % b != 0:
        raise ValueError(f"Размеры матрицы {matrix.shape} должны нацело делиться на размер блока {b}")
    sparse_matrix = bsr_matrix(matrix, blocksize=(b,b))
    return sparse_matrix
