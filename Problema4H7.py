import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Parámetros
a, b = 1.0, 2.0  # límites de x
h = 0.1  # paso
N = int((b - a) / h)  # número de puntos en la malla

# Discretización de la malla de x
x = np.linspace(a, b, N + 1)

# Sistema de ecuaciones lineales para diferencias finitas
A = np.zeros((N + 1, N + 1))
B = np.zeros(N + 1)

# Condiciones de frontera
A[0, 0] = 1  # y(1) = 0
B[0] = 0
A[-1, -1] = 1  # y(2) = 0
B[-1] = 0

# Llenado de la matriz A y el vector B para el sistema de ecuaciones
for i in range(1, N):
    xi = x[i]
    
    # Ecuaciones de diferencias finitas
    A[i, i - 1] = 1/h**2 - 1/(2*h*xi)
    A[i, i] = -2/h**2 + 3/xi**2
    A[i, i + 1] = 1/h**2 + 1/(2*h*xi)
    
    # Lado derecho de la ecuación
    B[i] = -xi**(-1) * np.log(xi) + 1

# Resolución del sistema de ecuaciones
y = solve(A, B)

# Imprimir resultados en la consola
for xi, yi in zip(x, y):
    print(f"x = {xi:.1f}, y(x) = {yi:.5f}")

# Gráfica de la solución
plt.plot(x, y, marker='o', color='b', label='Aproximación de $y(x)$')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Solución aproximada de la EDO usando diferencias finitas')
plt.grid(True)
plt.legend()
plt.show()