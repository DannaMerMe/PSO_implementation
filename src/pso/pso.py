# pso/pso.py
import numpy as np
from typing import Callable, List, Tuple
from .particle import Particle

class PSO:
    """
    Implementación del algoritmo de Optimización por Enjambre de Partículas (PSO).
    Optimiza una función objetivo mediante el movimiento coordinado de partículas.
    """
    
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        bounds_low: np.ndarray,
        bounds_high: np.ndarray,
        n_particles: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        rng_seed: int = 42
    ):
        """
        Inicializa el algoritmo PSO.
        Args:
            func: Función objetivo a minimizar
            dim: Número de dimensiones del problema
            bounds_low: Límites inferiores del espacio de búsqueda
            bounds_high: Límites superiores del espacio de búsqueda
            n_particles: Número de partículas en el enjambre
            w: Peso de inercia
            c1: Coeficiente cognitivo (atracción a mejor personal)
            c2: Coeficiente social (atracción a mejor global)
            rng_seed: Semilla para reproducibilidad
        """
        self.func = func
        self.dim = dim
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(rng_seed)
        self.particles: List[Particle] = [Particle(dim, bounds_low, bounds_high, self.rng) for _ in range(n_particles)]
        self.gbest = np.zeros(dim)
        self.gbest_value = np.inf
        self.history = []  # best value per iteration
    
    def step(self):
        """
        Ejecuta una iteración del algoritmo PSO.
        Evalúa todas las partículas, actualiza mejores posiciones y mueve las partículas.
        """
        # Evaluar y actualizar mejores posiciones
        for p in self.particles:
            val = self.func(p.position)
            if val < p.pbest_value:
                p.pbest_value = val
                p.pbest = p.position.copy()
            if val < self.gbest_value:
                self.gbest_value = val
                self.gbest = p.position.copy()
        
        # Actualizar velocidades y mover partículas
        for p in self.particles:
            p.update_velocity(self.w, self.c1, self.c2, self.gbest, self.rng)
            p.move(self.bounds_low, self.bounds_high)
    
    def run(self, iterations: int = 100, store_history: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        """
        Ejecuta el algoritmo PSO por un número determinado de iteraciones.
        Args:
            iterations: Número de iteraciones a ejecutar
            store_history: Si se debe guardar el historial de mejores valores
        Returns:
            Tupla con (mejor_posicion, mejor_valor, historial)
        """
        self.history = []
        for it in range(iterations):
            self.step()
            if store_history:
                self.history.append(self.gbest_value)
        return self.gbest.copy(), self.gbest_value, self.history