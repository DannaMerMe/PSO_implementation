# benchmarks/functions.py
import numpy as np
from typing import Tuple

def sphere(x: np.ndarray) -> float:
    """
    Función Sphere (convexa, unimodal).
    Args:
        x: Vector de entrada
    Returns:
        Suma de cuadrados de los componentes
    """
    return float(np.sum(x**2))

def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    """
    Función Rastrigin (multimodal, muchos mínimos locales).
    Args:
        x: Vector de entrada
        A: Parámetro de amplitud
    Returns:
        Valor de la función Rastrigin
    """
    n = x.size
    return float(A*n + np.sum(x**2 - A * np.cos(2*np.pi*x)))

def ackley(x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> float:
    """
    Función Ackley (multimodal, óptimo en origen).
    Args:
        x: Vector de entrada
        a: Parámetro de profundidad
        b: Parámetro de escala
        c: Parámetro de frecuencia
    Returns:
        Valor de la función Ackley
    """
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c*x))
    return float(-a * np.exp(-b*np.sqrt(s1/n)) - np.exp(s2/n) + a + np.e)

def rosenbrock(x: np.ndarray) -> float:
    """
    Función Rosenbrock (valle estrecho, difícil de optimizar).
    Args:
        x: Vector de entrada (n-dimensional)
    Returns:
        Valor de la función Rosenbrock
    """
    return float(np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0))

def benchmark_bounds(name: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna los límites estándar para funciones benchmark.
    Args:
        name: Nombre de la función benchmark
        dim: Dimensionalidad del problema
    Returns:
        Tupla con (límites_inferiores, límites_superiores)
    Raises:
        ValueError: Si el nombre del benchmark es desconocido
    """
    if name == "sphere":
        return -5.12*np.ones(dim), 5.12*np.ones(dim)
    if name == "rastrigin":
        return -5.12*np.ones(dim), 5.12*np.ones(dim)
    if name == "ackley":
        return -32.768*np.ones(dim), 32.768*np.ones(dim)
    if name == "rosenbrock":
        return -5*np.ones(dim), 10*np.ones(dim)
    raise ValueError("Unknown benchmark")