import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Cargar los datos
# Reemplaza 'weather.csv' con la ruta a tu archivo
data = pd.read_csv('weather.csv', header=0, delimiter=';')

# Asignar nombres a las columnas
data.columns = ['Temperatura_' + str(i) for i in range(data.shape[1] - 1)] + ['Estacion']

# Asegúrate de ajustar el nombre de la columna si es necesario
stations = data['Estacion']  # Columna de nombres de estaciones
temperatures = data.iloc[:, :-1]  # Las columnas de temperaturas

# 2. Estandarizar los datos
scaler = StandardScaler()

# Verificar si temperatures no está vacío
if temperatures.empty:
    raise ValueError("El DataFrame de temperaturas está vacío. Verifica el archivo CSV.")
    
temperatures_scaled = scaler.fit_transform(temperatures)

# 3. Aplicar PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(temperatures_scaled)

# Extraer los componentes principales
p1 = pca_result[:, 0]
p2 = pca_result[:, 1]

# 4. Gráfica de las curvas de los dos primeros componentes principales
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(p1) + 1), p1, label='p1 (Primer componente principal)', marker='o')
plt.plot(range(1, len(p2) + 1), p2, label='p2 (Segundo componente principal)', marker='o')
plt.xlabel('Estaciones (i)')
plt.ylabel('Valores de los componentes principales')
plt.title('Curvas de los componentes principales')
plt.legend()
plt.grid()
plt.savefig('componentes_principales.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura

# 5. Biplot de las estaciones en el espacio PCA
plt.figure(figsize=(10, 6))
plt.scatter(p1, p2, color='blue', label='Estaciones')
for i, station in enumerate(stations):
    plt.text(p1[i], p2[i], station, fontsize=8, ha='right')
plt.xlabel('Primer Componente Principal (p1)')
plt.ylabel('Segundo Componente Principal (p2)')
plt.title('Biplot: Estaciones en el espacio PCA')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.grid()
plt.legend()
plt.savefig('biplot_estaciones.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura

print(np.mean(temperatures_scaled, axis=0))  # Debería estar cerca de 0
print(np.std(temperatures_scaled, axis=0))   # Debería estar cerca de 1

