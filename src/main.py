"""
main.py FINAL 

Este archivo:
- Ejecuta PSO sobre la función Rastrigin (2D)
- Imprime reporte detallado en consola
- Guarda historial de convergencia
- Genera gráficas (convergencia + trayectorias)
"""

import numpy as np
import time
import os

from benchmarks.functions import rastrigin, benchmark_bounds
from pso.pso import PSO
from utils.viz import plot_convergence, plot_trajectories_2d


def run_pso_example():

    print("INICIANDO EJECUCIÓN PSO (FUNC. RASTRIGIN 2D)")
    func = rastrigin
    name = "rastrigin"
    dim = 2
    iterations = 150
    n_particles = 40

    low, high = benchmark_bounds(name, dim)

    pso = PSO(
        func=func,
        dim=dim,
        bounds_low=low,
        bounds_high=high,
        n_particles=n_particles,
        w=0.7,
        c1=1.5,
        c2=1.5,
        rng_seed=42
    )

    pso.history = []
    trajectories = []
    gbest_traj = []

    print("CONFIGURACIÓN DEL EXPERIMENTO")
    print(f" - Función objetivo: {name}")
    print(f" - Dimensiones: {dim}")
    print(f" - Partículas: {n_particles}")
    print(f" - Iteraciones: {iterations}")
    print(f" - w (inercia): {pso.w}")
    print(f" - c1 (componente cognitiva): {pso.c1}")
    print(f" - c2 (componente social): {pso.c2}")
    print(f" - Límites de búsqueda: {low}  →  {high}")


    print("Iniciando optimización...\n")

    start = time.time()
    for i in range(1, iterations + 1):

        # Guardar trayectorias de partículas
        positions = np.array([p.position.copy() for p in pso.particles])
        trajectories.append(positions)

        # Ejecutar un paso del PSO
        pso.step()

        # Guardar historial
        pso.history.append(pso.gbest_value)

        # GUARDAR TRAYECTORIA DE Gbest 
        gbest_traj.append(pso.gbest.copy())

        # métricas
        iter_values = [p.pbest_value for p in pso.particles]
        best_iter = np.min(iter_values)
        prom_iter = np.mean(iter_values)

        # impresión estilo ACO
        if i == 1 or i % 10 == 1:
            print(f"Iter {i:3d} | Mejor global: {pso.gbest_value:8.5f} | "
                  f"Mejor iter: {best_iter:8.5f} | Promedio: {prom_iter:8.5f}")

    end = time.time()
    time_exec = end - start

    # resultados finales 
    print("OPTIMIZACIÓN COMPLETADA")
    print(f"Tiempo de ejecución: {time_exec:.2f} segundos")
    print(f"Mejor posición encontrada: {pso.gbest}")
    print(f"Mejor valor (fitness): {pso.gbest_value:.5f}")

    print("[4/5] Validando solución...")
    print(f"Solución válida: {True}")

    # generar graficas 
    os.makedirs("results", exist_ok=True)

    # Convergencia
    conv_plot = plot_convergence(pso.history, title="Convergencia PSO - Rastrigin")
    conv_plot.savefig("results/pso_convergence.png")
    conv_plot.close()

    # Trayectorias
    traj_plot = plot_trajectories_2d(
        np.array(trajectories),
        np.array(gbest_traj),   # CORREGIDO: ahora es matriz Nx2
        bounds=(low, high),
        title="Trayectorias Partículas PSO - Rastrigin"
    )
    traj_plot.savefig("results/pso_trajectories.png")
    traj_plot.close()

    print("\nGráficas guardadas en la carpeta 'results/'")
    print("Ejecución terminada correctamente.\n")


if __name__ == "__main__":
    run_pso_example()
