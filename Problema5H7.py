import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Parámetros del problema
a, b = 1.0, 2.0  # límites de x
h = np.pi / 20  # paso
N = int((b - a) / h)  # número de puntos en la malla

# Discretización de la malla de x
x = np.linspace(a, b, N + 1)

# Sistema de ecuaciones lineales para diferencias finitas
A = np.zeros((N + 1, N + 1))
B = np.zeros(N + 1)

# Condiciones de frontera
A[0, 0] = 1  # y(0) = 1
B[0] = 1
A[-1, -1] = 1  # y(π/4) = 1
B[-1] = 1

# Llenado de la matriz A y el vector B para el sistema de ecuaciones
for i in range(1, N):
    A[i, i - 1] = 1 / h**2
    A[i, i] = -2 / h**2 + 1
    A[i, i + 1] = 1 / h**2
    B[i] = 0  # lado derecho es 0 debido a la ecuación y'' + y = 0

# Resolución del sistema de ecuaciones
y = solve(A, B)

# Imprimir resultados en la consola
for xi, yi in zip(x, y):
    print(f"x = {xi:.5f}, y(x) = {yi:.5f}")

# Gráfica de la solución
plt.plot(x, y, marker='o', color='b', label='Aproximación de $y(x)$')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Solución aproximada de la EDO usando diferencias finitas')
plt.grid(True)
plt.legend()
plt.show()