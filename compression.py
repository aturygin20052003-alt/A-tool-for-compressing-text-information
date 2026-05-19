import gzip
import lzma
import struct
import io
import numpy as np

def compress_matrix_gzip(matrix, b: int, output_filename: str):
    """
    Склеивает заголовок 12 байт и блочно-разреженную матрицу, затем сжимает всё вместе через gzip.
    """
    height, width = matrix.shape

    # 1. создание заголовка
    header = struct.pack('>iii', height, width, b)

    # 2. Сериализация разреженной матрицы
    buffer = io.BytesIO()
    np.savez(buffer, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, blocksize=matrix.blocksize)
    matrix_bytes = buffer.getvalue()

    # 3. Соединение и сжатие
    full_data = header + matrix_bytes
    compressed_data = gzip.compress(full_data)

    # 4. Запись в файл
    with open(output_filename, 'wb') as f:
        f.write(compressed_data)


def compress_matrix_lzma(matrix, b: int, output_filename: str):
    """
    Склеивает заголовок 12 байт и блочно-разреженную матрицу, затем сжимает всё вместе через lzma.
    """
    height, width = matrix.shape

    # 1. создание заголовка
    header = struct.pack('>iii', height, width, b)

    # 2. Сериализация разреженной матрицы
    buffer = io.BytesIO()
    np.savez(buffer, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, blocksize=matrix.blocksize)
    matrix_bytes = buffer.getvalue()

    # 3. Соединение и сжатие
    full_data = header + matrix_bytes
    compressed_data = lzma.compress(full_data)

    # 4. Запись в файл
    with open(output_filename, 'wb') as f:
        f.write(compressed_data)
