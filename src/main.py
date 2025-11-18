"""
main.py FINAL 

Este archivo:
- Ejecuta PSO sobre Rastrigin en 2D
- Guarda trayectorias
- Guarda historial de convergencia
- Genera gráficas sin errores
"""

import numpy as np
import os
from benchmarks.functions import rastrigin, benchmark_bounds
from pso.pso import PSO
from utils.viz import plot_convergence, plot_trajectories_2d
import matplotlib.pyplot as plt


def run_pso_example():

    print("Probando PSO con la función Rastrigin (2 dimensiones)...\n")

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

    # limpiar historia manualmente
    pso.history = []

    # listas para trayectorias
    trajectories = []
    gbest_traj = []

    print("Ejecutando iteraciones...\n")

    for _ in range(iterations):

        # guardar posiciones de las partículas
        positions = np.array([p.position.copy() for p in pso.particles])
        trajectories.append(positions)

        # guardar posición global actual
        gbest_traj.append(pso.gbest.copy())

        # ejecutar un paso del PSO
        pso.step()

        # guardar historial (esta ES LA PARTE IMPORTANTE)
        pso.history.append(pso.gbest_value)

    print("RESULTADOS FINALES")
    print("Mejor posición encontrada:", pso.gbest)
    print("Mejor valor (fitness):    ", pso.gbest_value)

    # asegurarse de que hay historial
    if len(pso.history) == 0:
        print("\nERROR: pso.history ESTÁ VACÍO.")
        print("Esto significa que el ciclo no se ejecutó o no se guardó el historial.\n")
        return

    # crear carpeta si no existe
    os.makedirs("results", exist_ok=True)

    # graficar convergencia
    conv_plot = plot_convergence(
        pso.history,
        title=f"Convergencia PSO - {name}"
    )
    conv_plot.savefig("results/pso_convergence.png")
    conv_plot.close()

    # graficar trayectorias (solo para 2D)
    traj_plot = plot_trajectories_2d(
        np.array(trajectories),
        gbest_traj,
        bounds=(low, high),
        title=f"Trayectorias partículas PSO - {name}"
    )
    traj_plot.savefig("results/pso_trajectories.png")
    traj_plot.close()

    print("\nGráficas guardadas en results/")
    print("Ejecución finalizada sin errores.")


if __name__ == "__main__":
    print("MAIN: Probador general del proyecto PSO\n")
    run_pso_example()
