import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

# Parámetros iniciales
h = 0.1
x_start = 0
x_end = 1
n = int((x_end - x_start) / h)  # Número de intervalos
x = np.linspace(x_start, x_end, n + 1)  # Valores de x de 0 a 1 con paso h

# Condiciones de frontera
y0 = -1
y1 = 0

# Matriz y vector para el sistema de ecuaciones
A = np.zeros((n + 1, n + 1))
B = np.zeros(n + 1)

# Condiciones de frontera
A[0, 0] = 1
B[0] = y0
A[-1, -1] = 1
B[-1] = y1

# Construcción de la matriz A y el vector B para los puntos internos
for i in range(1, n):
    xi = x[i]
    A[i, i-1] = 1/h**2 - (xi + 1) / (2 * h)
    A[i, i] = -2/h**2 + 2
    A[i, i+1] = 1/h**2 + (xi + 1) / (2 * h)
    B[i] = (1 - xi**2) * np.exp(xi)

# Resolución del sistema lineal Ay = B
y = solve(A, B)

# Imprimir resultados
for i in range(n + 1):
    print(f"x = {x[i]:.1f}, y = {y[i]:.4f}")

# Graficar la solución
plt.plot(x, y, marker='o', color='b', label='Solución $y(x)$')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando diferencias finitas")
plt.legend()
plt.grid(True)
plt.show()
