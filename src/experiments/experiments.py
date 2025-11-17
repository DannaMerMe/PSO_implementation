# experiments/experiments.py
import numpy as np
from benchmarks.functions import rastrigin, sphere, benchmark_bounds
from pso.pso import PSO
from comparative.ga import SimpleGA
from comparative.sa import SimpleSA
from utils.viz import plot_convergence, plot_trajectories_2d
import matplotlib.pyplot as plt
import os

def run_benchmark(name='rastrigin', dim=2, iters=200, n_particles=50, seed=123):
    """
    Ejecuta experimento de optimización con PSO en función benchmark.
    Args:
        name: Nombre de la función benchmark ('rastrigin', 'sphere')
        dim: Dimensionalidad del problema
        iters: Número de iteraciones
        n_particles: Número de partículas en el enjambre
        seed: Semilla para reproducibilidad
    Returns:
        Tupla con (mejor_posicion, mejor_valor, historial)
    """
    # Seleccionar función benchmark
    if name == 'rastrigin':
        func = rastrigin
    elif name == 'sphere':
        func = sphere
    else:
        raise ValueError(name)
    
    # Obtener límites del problema
    low, high = benchmark_bounds(name, dim)
    
    # Inicializar PSO
    pso = PSO(func=func, dim=dim, bounds_low=low, bounds_high=high, n_particles=n_particles, rng_seed=seed)
    
    # Almacenar trayectorias para visualización 2D
    trajectories = []
    gbest_traj = []
    
    for it in range(iters):
        # Guardar posiciones actuales
        positions = np.array([p.position.copy() for p in pso.particles])
        trajectories.append(positions)
        pso.step()
        gbest_traj.append(pso.gbest.copy())
    
    best, best_val, history = pso.gbest, pso.gbest_value, pso.history
    
    # Crear directorio de resultados
    os.makedirs("results", exist_ok=True)
    
    # Graficar y guardar convergencia
    plt = plot_convergence(history, title=f"PSO {name} convergence")
    plt.savefig(f"results/pso_{name}_conv.png")
    plt.close()
    
    # Graficar trayectorias si es 2D
    if dim == 2:
        plt2 = plot_trajectories_2d(np.array(trajectories), gbest_traj, bounds=(low, high), title=f"PSO {name} trajectories")
        plt2.savefig(f"results/pso_{name}_traj.png")
        plt2.close()
    
    return best, best_val, history