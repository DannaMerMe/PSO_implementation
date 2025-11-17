# utils/viz.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_convergence(history: List[float], title: str = "Convergencia"):
    """
    Grafica la convergencia del algoritmo de optimización.
    Args:
        history: Lista con los mejores valores en cada iteración
        title: Título del gráfico
    Returns:
        Objeto pyplot para mostrar o guardar
    """
    plt.figure()
    plt.plot(history)
    plt.yscale('log' if min(history) > 0 else 'linear')
    plt.xlabel("Iteración")
    plt.ylabel("Mejor valor (fitness)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_trajectories_2d(particles_positions: List[np.ndarray], gbest_traj: List[np.ndarray], bounds=None, title="Trayectorias 2D"):
    """
    Visualiza las trayectorias de las partículas en 2D.
    Args:
        particles_positions: Array con posiciones (iteraciones, n_particulas, 2)
        gbest_traj: Lista con posiciones del mejor global en cada iteración
        bounds: Tupla opcional con (límites_inferiores, límites_superiores)
        title: Título del gráfico
    Returns:
        Objeto pyplot para mostrar o guardar
    """
    arr = np.array(particles_positions)
    iters, n, _ = arr.shape
    plt.figure()
    
    # Graficar trayectoria de cada partícula
    for j in range(n):
        traj = arr[:, j, :]
        plt.plot(traj[:,0], traj[:,1], alpha=0.6)
        plt.scatter(traj[0,0], traj[0,1], marker='x')  # Posición inicial
    
    # Graficar trayectoria del mejor global
    g = np.array(gbest_traj)
    plt.plot(g[:,0], g[:,1], 'k--', linewidth=2, label='gbest')
    
    # Aplicar límites si se proporcionan
    if bounds is not None:
        low, high = bounds
        plt.xlim(low[0], high[0])
        plt.ylim(low[1], high[1])
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt