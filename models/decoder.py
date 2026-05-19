"""
Universal Decoder for Compressed Text Images
Декодирование сжатых изображений (формат Арсения)

Формат файла:
- 4 байта: height (int32, big-endian)
- 4 байта: width (int32, big-endian)
- 4 байта: b (int32, big-endian)
- далее: np.savez (data, indices, indptr, blocksize)
"""

import gzip
import lzma
import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import io
from scipy.sparse import bsr_array

def decode_compressed(filepath, verbose=True):
    """
    Декодирование сжатого файла (новая версия с np.savez)
    """
    # Распаковка
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
    elif filepath.endswith('.xz'):
        with lzma.open(filepath, 'rb') as f:
            data = f.read()
    else:
        with open(filepath, 'rb') as f:
            data = f.read()
    
    height = struct.unpack('>i', data[0:4])[0]
    width = struct.unpack('>i', data[4:8])[0]
    b = struct.unpack('>i', data[8:12])[0]
    
    if verbose:
        print(f"{height}×{width}, b={b}")
    
    npz_data = data[12:]
    buffer = io.BytesIO(npz_data)
    
    with np.load(buffer) as npz:
        matrix_data = npz['data']
        indices = npz['indices']
        indptr = npz['indptr']
        blocksize = tuple(npz['blocksize'])
    
    if verbose:
        print(f"   blocksize={blocksize}, nnz_blocks={len(indices)}")
    
    bsr = bsr_array((matrix_data, indices, indptr), 
                    shape=(height, width), 
                    blocksize=blocksize)
    
    image = bsr.toarray().astype(np.uint8)
    
    return image, b

def visualize(image, title="", save_path=None):
    """
    Визуализация: изображение + spy-график
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Reconstructed Image\n{image.shape[0]}×{image.shape[1]}')
    ax1.axis('off')
    
    binary = image > 0
    non_zero = binary.sum()
    sparsity = 100 * non_zero / image.size
    
    ax2.spy(binary, markersize=0.5, aspect='auto')
    ax2.set_title(f'Spy: {non_zero:,} non-zero ({sparsity:.1f}%)')
    ax2.set_xlabel('Column index')
    ax2.set_ylabel('Row index')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.show()
    
    if save_path:
        plt.imsave(save_path, image, cmap='gray')
        print(f"Сохранено: {save_path}")
    
    print(f"\nСтатистика: {image.shape[0]}×{image.shape[1]}, "
          f"ненулевых: {non_zero:,} ({sparsity:.1f}%)")

def batch_decode(folder_path, verbose=True, show_plots=True):
    """
    Пакетное декодирование всех файлов в папке
    """
    files = glob.glob(os.path.join(folder_path, "*.gz")) + \
            glob.glob(os.path.join(folder_path, "*.xz"))
    
    files = list(set(files))
    files.sort()
    
    print("="*70)
    print(f"Папка: {folder_path}")
    print(f"Найдено файлов: {len(files)}")
    print("="*70)
    
    results = []
    
    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        print(f"\n[{i}/{len(files)}]  {filename}")
        print("-" * 50)
        
        try:
            img, b = decode_compressed(filepath, verbose)
            
            non_zero = np.count_nonzero(img)
            sparsity = 100 * non_zero / img.size
            
            results.append({
                'file': filename,
                'shape': img.shape,
                'b': b,
                'non_zero': non_zero,
                'sparsity': sparsity
            })
            
            output_name = filename.replace('.gz', '').replace('.xz', '') + '_decoded.png'
            output_path = os.path.join(folder_path, output_name)
            plt.imsave(output_path, img, cmap='gray')
            
            print(f"  {img.shape[0]}×{img.shape[1]}, b={b}, ненулевых: {non_zero:,} ({sparsity:.1f}%)")
            print(f"  Сохранено: {output_name}")
            
            if show_plots:
                visualize(img, title=filename, save_path=None)
                
        except Exception as e:
            print(f"  Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*70)
    print(f"Успешно обработано: {len(results)} из {len(files)}")
    
    if results:
        print("\nДЕТАЛИ ПО ФАЙЛАМ:")
        print("-" * 70)
        print(f"{'№':<4} {'Имя файла':<25} {'Размер':<18} {'Разреженность':<12} {'b':<6}")
        print("-" * 70)
        
        for i, r in enumerate(results, 1):
            size_str = f"{r['shape'][0]}×{r['shape'][1]}"
            sparsity_str = f"{r['sparsity']:.1f}%"
            print(f"{i:<4} {r['file']:<25} {size_str:<18} {sparsity_str:<12} {r['b']:<6}")
        
        print("-" * 70)
        
        avg_sparsity = sum(r['sparsity'] for r in results) / len(results)
        print(f"\n Средняя разреженность: {avg_sparsity:.1f}%")
        print(f" Мин. разреженность: {min(r['sparsity'] for r in results):.1f}%")
        print(f" Макс. разреженность: {max(r['sparsity'] for r in results):.1f}%")
    
    print("\n" + "="*70)
    print(" ГОТОВО! Все изображения декодированы и сохранены!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    folder_path = "test_files"
    
    if not os.path.exists(folder_path):
        print(f" Папка '{folder_path}' не найдена!")
        print("   Убедитесь, что папка существует и файлы загружены.")
    else:
        results = batch_decode(folder_path, verbose=True, show_plots=True)
