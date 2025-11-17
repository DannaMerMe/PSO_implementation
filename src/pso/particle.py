# pso/particle.py
import numpy as np
from typing import Callable

class Particle:
    """
    Partícula individual en el algoritmo PSO.
    Mantiene posición, velocidad y mejor posición histórica.
    """
    
    def __init__(self, dim: int, bounds_low: np.ndarray, bounds_high: np.ndarray, rng: np.random.Generator):
        """
        Inicializa partícula con posición y velocidad aleatorias.
        Args:
            dim: Número de dimensiones del espacio de búsqueda
            bounds_low: Límites inferiores para cada dimensión
            bounds_high: Límites superiores para cada dimensión
            rng: Generador de números aleatorios
        """
        self.dim = dim
        self.rng = rng
        self.position = rng.uniform(bounds_low, bounds_high)
        vmax = (bounds_high - bounds_low)
        self.velocity = rng.uniform(-vmax, vmax) * 0.1
        self.pbest = self.position.copy()
        self.pbest_value = np.inf

    def update_velocity(self, w: float, c1: float, c2: float, gbest: np.ndarray, rng: np.random.Generator):
        """
        Actualiza velocidad usando ecuación PSO.
        Args:
            w: Peso de inercia
            c1: Coeficiente cognitivo
            c2: Coeficiente social
            gbest: Mejor posición global del enjambre
            rng: Generador de números aleatorios
        """
        r1 = rng.random(self.dim)
        r2 = rng.random(self.dim)
        cognitive = c1 * r1 * (self.pbest - self.position)
        social = c2 * r2 * (gbest - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def move(self, bounds_low: np.ndarray, bounds_high: np.ndarray):
        """
        Mueve la partícula según su velocidad.
        Args:
            bounds_low: Límites inferiores del espacio
            bounds_high: Límites superiores del espacio
        """
        self.position = self.position + self.velocity
        self.position = np.minimum(np.maximum(self.position, bounds_low), bounds_high)