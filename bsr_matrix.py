from scipy.sparse import bsr_matrix

def to_bsr(sparse_matrix, b=1):
    sparse = bsr_matrix(sparse_matrix, blocksize=(b,b))
    return sparse

matrix = [
    [1,0],
    [0,1]]
res = to_bsr(matrix)
print(res)
