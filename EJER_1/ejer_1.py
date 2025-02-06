import numpy as np
import matplotlib.pyplot as plt

def simular_lanzamientos(p, N):
    """
    Simula N experimentos de lanzamientos de moneda hasta obtener el primer éxito
    p: probabilidad de éxito en cada lanzamiento
    N: número de experimentos a realizar
    """
    
    intentos_hasta_exito = []
    
    for _ in range(N):
        intentos = 0
        while True:
            intentos += 1
            if np.random.random() < p:
                break
        intentos_hasta_exito.append(intentos)
    
    return intentos_hasta_exito

def main():
    while True:
        try:
            p = float(input("Ingresar la probabilidad de éxito (entre 0 y 1): "))
            if 0 < p < 1:
                break
            else:
                print("La probabilidad debe estar entre 0 y 1")
        except ValueError:
            print("Por favor, ingrese un número válido")
    
    N = 1000
    resultados = simular_lanzamientos(p, N)
    
    plt.figure(figsize=(10, 6))
    
    max_intentos = max(resultados)
    bins = np.arange(1, max_intentos + 2) - 0.5
    
    plt.hist(resultados, bins=bins, density=True, alpha=0.7, color='blue')
    plt.title(f'Densidad de intentos hasta el primer éxito (p={p})')
    plt.xlabel('Número de lanzamientos')
    plt.ylabel('Densidad')
    
    plt.xticks(np.arange(1, max_intentos + 1))
    
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
