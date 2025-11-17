# comparative/sa.py
import numpy as np
from typing import Callable

class SimpleSA:
    """
    Implementación simple del algoritmo Simulated Annealing (SA).
    Optimiza mediante enfriamiento gradual y aceptación probabilística de soluciones.
    """
    
    def __init__(self, func: Callable[[np.ndarray], float], dim: int, low: np.ndarray, high: np.ndarray,
                 initial_temp: float = 1.0, final_temp: float = 1e-3, alpha: float = 0.95, rng_seed: int = 42):
        """
        Inicializa el algoritmo Simulated Annealing.
        Args:
            func: Función objetivo a minimizar
            dim: Número de dimensiones del problema
            low: Límites inferiores del espacio de búsqueda
            high: Límites superiores del espacio de búsqueda
            initial_temp: Temperatura inicial
            final_temp: Temperatura final (criterio de parada)
            alpha: Factor de enfriamiento (0 < alpha < 1)
            rng_seed: Semilla para reproducibilidad
        """
        self.func = func
        self.dim = dim
        self.low = low
        self.high = high
        self.temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.rng = np.random.default_rng(rng_seed)
        self.current = self.rng.uniform(low, high)
        self.current_value = func(self.current)
    
    def neighbor(self):
        """
        Genera una solución vecina mediante perturbación gaussiana.
        Returns:
            Vector con la solución vecina dentro de los límites
        """
        # Perturbación gaussiana
        step = self.rng.normal(0, 0.1, size=self.dim)
        cand = self.current + step * (self.high - self.low) * 0.1
        cand = np.minimum(np.maximum(cand, self.low), self.high)
        return cand
    
    def run(self):
        """
        Ejecuta el algoritmo SA hasta alcanzar la temperatura final.
        Returns:
            Tupla con (mejor_posicion, mejor_valor, historial)
        """
        history = []
        while self.temp > self.final_temp:
            cand = self.neighbor()
            val = self.func(cand)
            # Aceptar si mejora o con probabilidad exponencial
            if val < self.current_value or self.rng.random() < np.exp((self.current_value - val)/self.temp):
                self.current = cand
                self.current_value = val
            history.append(self.current_value)
            self.temp *= self.alpha
        return self.current.copy(), self.current_value, history