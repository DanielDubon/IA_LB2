import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from scipy import stats

# -------------------------------
# Paso 1: Cargar el CSV y seleccionar la columna de área
# -------------------------------
# Cargar el CSV
df = pd.read_csv("areas.csv", sep=";")

print("Columnas en el CSV:", df.columns.tolist())

area_column = "Area in square kilometres"

if area_column not in df.columns:
    raise ValueError(f"La columna '{area_column}' no se encontró en el CSV.")

areas = df[area_column].dropna()

print("Datos de áreas cargados correctamente.")


# -------------------------------
# Paso 2: Extraer el primer dígito no nulo de cada valor
# -------------------------------
def extract_first_digit(value):
    """
    Extrae el primer dígito no nulo de un número.
    Se convierte el número a cadena y se utiliza una expresión regular para encontrar
    el primer dígito distinto de cero.
    """
    s = str(value)
    match = re.search(r'[1-9]', s)
    if match:
        return int(match.group())
    return np.nan

df['first_digit'] = areas.apply(extract_first_digit)
first_digits = df['first_digit'].dropna().astype(int)

# -------------------------------
# Paso 3: Calcular las probabilidades teóricas de la Ley de Benford
# -------------------------------
digits = np.arange(1, 10)
benford_probs = np.log10(1 + 1/digits)
benford_dict = {d: np.log10(1 + 1/d) for d in digits}

# -------------------------------
# Paso 4: Visualizar la densidad (histograma y KDE)
# -------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(first_digits, bins=np.arange(0.5, 10.5, 1), stat="density", color="skyblue", label="Empírica")
plt.plot(digits, benford_probs, 'ro-', label="Benford Teórica")
plt.xlabel("Primer Dígito")
plt.ylabel("Densidad")
plt.title("Comparación de densidades: Empírica vs. Benford")
plt.legend()
plt.show()

# -------------------------------
# Paso 5: Visualizar las funciones de distribución (CDF)
# -------------------------------
sorted_digits = np.sort(first_digits)
ecdf = np.arange(1, len(sorted_digits)+1) / len(sorted_digits)
benford_cdf = np.cumsum(benford_probs)

plt.figure(figsize=(10, 6))
plt.step(digits, benford_cdf, where="post", color="red", label="Benford CDF")
plt.step(sorted_digits, ecdf, where="post", color="blue", label="Empírica CDF")
plt.xlabel("Primer Dígito")
plt.ylabel("Probabilidad Acumulada")
plt.title("Comparación de Funciones de Distribución Acumulada")
plt.legend()
plt.show()

# -------------------------------
# Paso 6: Gráfica PP (Probabilidad vs. Probabilidad)
# -------------------------------
empirical_counts = first_digits.value_counts().sort_index()
empirical_probs = empirical_counts / empirical_counts.sum()
empirical_cum = np.cumsum(empirical_probs)

plt.figure(figsize=(8, 8))
plt.plot(benford_cdf, empirical_cum, 'bo', label="Puntos PP")
plt.plot([0, 1], [0, 1], 'r--', label="Línea de identidad")
plt.xlabel("CDF Teórica")
plt.ylabel("CDF Empírica")
plt.title("Gráfica PP: Comparación de CDFs")
plt.legend()
plt.show()

# -------------------------------
# Paso 7: Gráfica QQ (Cuantil vs. Cuantil)
# -------------------------------
def benford_ppf(p):
    """
    Función de cuantiles (inversa de la CDF) para la distribución de Benford.
    Para un nivel p, se retorna el primer dígito tal que la suma acumulada supere a p.
    """
    cumulative = 0
    for d in digits:
        cumulative += benford_dict[d]
        if p <= cumulative:
            return d
    return 9

p_levels = np.linspace(0, 1, 100)
theoretical_quantiles = np.array([benford_ppf(p) for p in p_levels])
empirical_quantiles = np.percentile(first_digits, p_levels * 100)

plt.figure(figsize=(8, 8))
plt.plot(theoretical_quantiles, empirical_quantiles, 'bo', label="Puntos QQ")
plt.plot([1, 9], [1, 9], 'r--', label="Línea de identidad")
plt.xlabel("Cuantiles Teóricos")
plt.ylabel("Cuantiles Empíricos")
plt.title("Gráfica QQ: Comparación de Cuantiles")
plt.legend()
plt.show()

# -------------------------------
# Paso 8: Prueba de Kolmogorov-Smirnov (KS)
# -------------------------------
cumulative_probs = np.cumsum(benford_probs)

def benford_cdf_func(x):
    """
    Función de distribución acumulada para la Ley de Benford.
    Se asume que x es un número real. Se utiliza la suma acumulada para devolver la probabilidad.
    """
    if x < 1:
        return 0
    elif x >= 9:
        return 1
    else:
        d = int(math.floor(x))
        return cumulative_probs[d - 1]

benford_cdf_vectorized = np.vectorize(benford_cdf_func)

ks_statistic, p_value = stats.kstest(first_digits, benford_cdf_vectorized)

print("Resultados de la Prueba de Kolmogorov-Smirnov:")
print("Estadístico KS:", ks_statistic)
print("p-valor:", p_value)

# -------------------------------
# Paso 9: Conclusión
# -------------------------------
if p_value > 0.05:
    print("No se rechaza la hipótesis nula: los datos siguen la Ley de Benford.")
else:
    print("Se rechaza la hipótesis nula: los datos no siguen la Ley de Benford.")
