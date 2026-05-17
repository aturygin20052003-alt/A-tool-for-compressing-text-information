import gzip
import lzma
import struct
import pickle

def compress_matrix_gzip(matrix, b: int, output_filename: str):
    """
    Склеивает заголовок 12 байт и матрицу, затем сжимает всё вместе через gzip.
    """
    height, width = matrix.shape

    # 1. создание заголовка
    header = struct.pack('>iii', height, width, b)

    # 2. Сериализаация матрицы
    matrix_bytes = pickle.dumps(matrix)

    # 3. Соединеие и сжатие 
    full_data = header + matrix_bytes
    compressed_data = gzip.compress(full_data)

    # 4. Запись в файл
    with open(output_filename, 'wb') as f:
        f.write(compressed_data)


def compress_matrix_lzma(matrix, b: int, output_filename: str):
    """
    №Склеивает заголовок 12 байт и матрицу, затем сжимает всё вместе через lzma.
    """
    height, width = matrix.shape

    # 1. создание заголовка
    header = struct.pack('>iii', height, width, b)

    # 2. Сериализаация матрицы
    matrix_bytes = pickle.dumps(matrix)

    # 3. Соединеие и сжатие 
    full_data = header + matrix_bytes
    compressed_data = lzma.compress(full_data)

    # 4. Запись в файл
    with open(output_filename, 'wb') as f:
        f.write(compressed_data)
