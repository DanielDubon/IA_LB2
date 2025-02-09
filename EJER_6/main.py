import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

# !DISCLAIMER: para que se corra correctamente, tienen que estar dentro de la carpeta EJER_6. de lo contrario no encontrará el archivo de las imágenes

def crop_to_multiple(image, block_size):
    """
    Recorta la imagen para que sus dimensiones sean múltiplos de block_size.
    """
    H, W = image.shape
    new_H = (H // block_size) * block_size
    new_W = (W // block_size) * block_size
    return image[:new_H, :new_W]

def image_to_blocks(image, block_size):
    """
    Divide la imagen en bloques de tamaño block_size x block_size.
    Retorna un arreglo de bloques de forma (n_bloques, block_size, block_size)
    """
    H, W = image.shape
    blocks = []
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return np.array(blocks)

def blocks_to_image(blocks, image_shape, block_size):
    """
    Reconstruye la imagen a partir de los bloques.
    """
    H, W = image_shape
    n_blocks_row = H // block_size
    n_blocks_col = W // block_size
    image_reconstructed = np.zeros(image_shape, dtype=blocks.dtype)
    index = 0
    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            image_reconstructed[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[index]
            index += 1
    return image_reconstructed

def pca_compression(blocks, k):
    """
    Aplica PCA a la matriz formada por los bloques vectorizados.
    - blocks: arreglo de bloques de forma (n_bloques, C, C)
    - k: número de componentes principales a conservar
    Retorna los bloques reconstruidos (desvectorizados) usando los k componentes.
    """
    n_blocks, C, _ = blocks.shape
    X = blocks.reshape(n_blocks, -1)
    
    pca = PCA(n_components=k)
    X_transformed = pca.fit_transform(X)
    
    X_reconstructed = pca.inverse_transform(X_transformed)
    blocks_reconstructed = X_reconstructed.reshape(n_blocks, C, C)
    
    return blocks_reconstructed

def mse(image1, image2):
    """
    Calcula el error cuadrático medio (MSE) entre dos imágenes.
    """
    return np.mean((image1.astype("float") - image2.astype("float")) ** 2)

def main():
    block_size = 8 
    k_values = [1, 5, 10, 20, 40, 64]
    image_paths = [
        "./imgen1.jpg",
        "./imgen2.jpg",
        "./imgen3.jpg",
    ]
    
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Archivo {path} no encontrado. Saltando.")
            continue
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"No se pudo leer la imagen {path}.")
            continue
        
        image = crop_to_multiple(image, block_size)
        H, W = image.shape
        blocks = image_to_blocks(image, block_size)
        
        mse_values = []
        reconstructions = {}
        
        for k in k_values:
            blocks_reconstructed = pca_compression(blocks, k)
            image_reconstructed = blocks_to_image(blocks_reconstructed, (H, W), block_size)
            error = mse(image, image_reconstructed)
            mse_values.append(error)
            reconstructions[k] = image_reconstructed
            
            print(f"Imagen: {path} | k = {k:2d} | MSE = {error:.2f}")
        

        plt.figure(figsize=(10, 4))
        plt.plot(k_values, mse_values, marker="o")
        plt.xlabel("Número de componentes (k)")
        plt.ylabel("Error de reconstrucción (MSE)")
        plt.title(f"Error de reconstrucción vs k para {os.path.basename(path)}")
        plt.grid(True)
        plt.show()
        

        n_total = len(k_values) + 1  
        n_cols = 4
        n_rows = int(math.ceil(n_total / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        if n_rows == 1 or n_cols == 1:
            axes = np.array(axes).flatten()
        else:
            axes = axes.flatten()
        

        scale_factor = 1
        

        original_zoom = cv2.resize(image, (W * scale_factor, H * scale_factor), interpolation=cv2.INTER_NEAREST)
        axes[0].imshow(original_zoom, cmap="gray", interpolation='nearest')
        axes[0].set_title("Original")
        axes[0].axis("off")
        

        for idx, k in enumerate(k_values, start=1):
            recon_zoom = cv2.resize(reconstructions[k], (W * scale_factor, H * scale_factor), interpolation=cv2.INTER_NEAREST)
            axes[idx].imshow(recon_zoom, cmap="gray", interpolation='nearest')
            axes[idx].set_title(f"k = {k}")
            axes[idx].axis("off")
        
        
        for i in range(n_total, len(axes)):
            axes[i].axis("off")
        
        fig.suptitle(f"Reconstrucción usando PCA - {os.path.basename(path)}", fontsize=16)
        plt.tight_layout()
        plt.show()

main()
