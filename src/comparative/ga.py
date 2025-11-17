# comparative/ga.py
import numpy as np
from typing import Callable

class SimpleGA:
    """
    Implementación simple del Algoritmo Genético (GA).
    Optimiza mediante selección, cruce y mutación de una población de soluciones.
    """
    
    def __init__(self, func: Callable[[np.ndarray], float], dim: int, low: np.ndarray, high: np.ndarray,
                 pop_size: int = 50, crossover_rate: float = 0.7, mutation_rate: float = 0.1, rng_seed: int = 42):
        """
        Inicializa el algoritmo genético.
        Args:
            func: Función objetivo a minimizar
            dim: Número de dimensiones del problema
            low: Límites inferiores del espacio de búsqueda
            high: Límites superiores del espacio de búsqueda
            pop_size: Tamaño de la población
            crossover_rate: Probabilidad de cruce entre individuos
            mutation_rate: Probabilidad de mutación por gen
            rng_seed: Semilla para reproducibilidad
        """
        self.func = func
        self.dim = dim
        self.low = low
        self.high = high
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(rng_seed)
        self.population = self.rng.uniform(low, high, size=(pop_size, dim))
        self.fitness = np.array([func(ind) for ind in self.population])
    
    def select(self):
        """
        Selecciona individuos mediante torneo binario.
        Returns:
            Nueva población seleccionada
        """
        i1 = self.rng.integers(0, self.pop_size, size=self.pop_size)
        i2 = self.rng.integers(0, self.pop_size, size=self.pop_size)
        chosen = np.where(self.fitness[i1] < self.fitness[i2], i1, i2)
        return self.population[chosen]
    
    def crossover(self, pop):
        """
        Aplica cruce aritmético entre pares de individuos.
        Args:
            pop: Población actual
        Returns:
            Población después del cruce
        """
        for i in range(0, self.pop_size, 2):
            if self.rng.random() < self.crossover_rate and i+1 < self.pop_size:
                alpha = self.rng.random(self.dim)
                a, b = pop[i].copy(), pop[i+1].copy()
                pop[i] = alpha*a + (1-alpha)*b
                pop[i+1] = alpha*b + (1-alpha)*a
        return pop
    
    def mutate(self, pop):
        """
        Aplica mutación uniforme a la población.
        Args:
            pop: Población actual
        Returns:
            Población después de la mutación
        """
        for i in range(self.pop_size):
            for d in range(self.dim):
                if self.rng.random() < self.mutation_rate:
                    pop[i, d] = self.rng.uniform(self.low[d], self.high[d])
        return pop
    
    def run(self, generations: int = 100):
        """
        Ejecuta el algoritmo genético por un número de generaciones.
        Args:
            generations: Número de generaciones a evolucionar
        Returns:
            Tupla con (mejor_individuo, mejor_fitness, historial)
        """
        history = []
        for g in range(generations):
            pop = self.select()
            pop = self.crossover(pop)
            pop = self.mutate(pop)
            self.population = pop
            self.fitness = np.array([self.func(ind) for ind in self.population])
            history.append(self.fitness.min())
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx], history