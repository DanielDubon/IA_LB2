import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def solicitar_media(n_dim):
    """Solicita al usuario ingresar el vector de medias"""
    print(f"\nIngrese las {n_dim} medias (una por línea):")
    media = []
    for i in range(n_dim):
        while True:
            try:
                valor = float(input(f"Media {i+1}: "))
                media.append(valor)
                break
            except ValueError:
                print("Por favor, ingrese un número válido")
    return np.array(media)

def solicitar_covarianza(n_dim):
    """Solicita al usuario ingresar la matriz de covarianza"""
    print(f"\nIngrese la matriz de covarianza {n_dim}x{n_dim} (fila por fila, valores separados por espacios):")
    covarianza = []
    for i in range(n_dim):
        while True:
            try:
                fila = input(f"Fila {i+1}: ").split()
                fila = [float(x) for x in fila]
                if len(fila) != n_dim:
                    print(f"Debe ingresar exactamente {n_dim} valores")
                    continue
                covarianza.append(fila)
                break
            except ValueError:
                print("Por favor, ingrese números válidos separados por espacios")
    
    covarianza = np.array(covarianza)
    
    if not np.allclose(covarianza, covarianza.T):
        print("Advertencia: La matriz no es simétrica. Se usará (M + M.T)/2")
        covarianza = (covarianza + covarianza.T) / 2
    
    try:
        np.linalg.cholesky(covarianza)
    except np.linalg.LinAlgError:
        print("Advertencia: La matriz no es definida positiva. Se usará una matriz ajustada")
        eigvals = np.linalg.eigvals(covarianza)
        if min(eigvals) < 0:
            covarianza += (-min(eigvals) + 0.01) * np.eye(n_dim)
    
    return covarianza

def generar_muestra_gaussiana(n_dim, n_muestras, media, covarianza):
    """
    Genera muestras de una distribución gaussiana multivariada
    
    Parámetros:
    n_dim: dimensión de la distribución
    n_muestras: número de muestras a generar
    media: vector de medias
    covarianza: matriz de covarianza
    """
    return np.random.multivariate_normal(media, covarianza, size=n_muestras)

def comparar_parametros(muestra, media_teorica, cov_teorica):
    """
    Compara los parámetros teóricos con los estimados de la muestra
    """
    media_estimada = np.mean(muestra, axis=0)
    cov_estimada = np.cov(muestra.T)
    
    print("\nComparación de parámetros:")
    print("\nMedia teórica:")
    print(media_teorica)
    print("\nMedia estimada:")
    print(media_estimada)
    print("\nDiferencia en medias:")
    print(np.abs(media_teorica - media_estimada))
    
    print("\nCovarianza teórica:")
    print(cov_teorica)
    print("\nCovarianza estimada:")
    print(cov_estimada)
    print("\nDiferencia máxima en covarianzas:")
    print(np.max(np.abs(cov_teorica - cov_estimada)))

def main():
    while True:
        try:
            n_dim = int(input("Ingrese la dimensión (n ≥ 4): "))
            if n_dim >= 4:
                break
            print("La dimensión debe ser mayor o igual a 4")
        except ValueError:
            print("Por favor, ingrese un número entero válido")
    
    media = solicitar_media(n_dim)
    covarianza = solicitar_covarianza(n_dim)
    
    n_muestras = 1000
    
    muestra = np.random.multivariate_normal(media, covarianza, size=n_muestras)
    
    import pandas as pd
    df = pd.DataFrame(muestra, columns=[f'X{i+1}' for i in range(n_dim)])
    
    plt.figure(figsize=(12, 12))
    sns.pairplot(df, diag_kind='kde')
    plt.tight_layout()
    plt.show()
    
    comparar_parametros(muestra, media, covarianza)

if __name__ == "__main__":
    main()
