import numpy as np
import time
import os

# Benchmarks
from benchmarks.functions import (
    rastrigin, sphere, ackley, rosenbrock, benchmark_bounds
)

# PSO
from pso.pso import PSO

# Algoritmos comparativos
from comparative.ga import SimpleGA
from comparative.sa import SimpleSA

# Sistema M/M/c
from queueing.mmc_sim import objective_mmcc

# Gráficas
from utils.viz import plot_convergence, plot_trajectories_2d
import matplotlib.pyplot as plt

def run_pso_custom(func_name, func, dim):
    """
    Ejecuta PSO con parámetros personalizados ingresados por el usuario.
    Args:
        func_name: Nombre de la función benchmark
        func: Función objetivo a optimizar
        dim: Dimensionalidad del problema
    """
    print("\n=== CONFIGURACIÓN PSO ===")
    n_particles = int(input("Número de partículas: "))
    iterations = int(input("Número de iteraciones: "))
    w = float(input("Ingrese w (inercia): "))
    c1 = float(input("Ingrese c1 (componente cognitiva): "))
    c2 = float(input("Ingrese c2 (componente social): "))

    low, high = benchmark_bounds(func_name, dim)

    pso = PSO(
        func=func,
        dim=dim,
        bounds_low=low,
        bounds_high=high,
        n_particles=n_particles,
        w=w,
        c1=c1,
        c2=c2,
        rng_seed=42
    )

    trajectories = []
    gbest_traj = []
    pso.history = []

    print("\n=== INICIANDO OPTIMIZACIÓN PSO ===")
    print(f"Función objetivo: {func_name}")
    print(f"Dimensiones: {dim}")
    print(f"Partículas: {n_particles}")
    print(f"Iteraciones: {iterations}")

    start = time.time()

    for it in range(1, iterations + 1):
        # Guardar posiciones para visualización
        positions = np.array([p.position.copy() for p in pso.particles])
        trajectories.append(positions)

        pso.step()
        pso.history.append(pso.gbest_value)
        gbest_traj.append(pso.gbest.copy())

        iter_vals = [p.pbest_value for p in pso.particles]
        best_iter = np.min(iter_vals)

        # Mostrar progreso cada 10 iteraciones
        if it == 1 or it % 10 == 0:
            print(f"Iter {it:3d} | gbest: {pso.gbest_value:.6f} | best iter: {best_iter:.6f}")

    end = time.time()

    print("\n== RESULTADOS FINALES ===")
    print(f"Tiempo: {end-start:.3f} s")
    print(f"Mejor posición: {pso.gbest}")
    print(f"Mejor valor : {pso.gbest_value:.6f}")

    os.makedirs("results", exist_ok=True)

    # Guardar gráfica de convergencia
    conv = plot_convergence(pso.history, f"Convergencia {func_name}")
    conv.savefig(f"results/PSO_{func_name}_convergence.png")
    conv.close()

    # Guardar trayectorias si es 2D
    if dim == 2:
        traj = plot_trajectories_2d(
            np.array(trajectories),
            np.array(gbest_traj),
            bounds=(low, high),
            title=f"Trayectorias PSO - {func_name}"
        )
        traj.savefig(f"results/PSO_{func_name}_traj.png")
        traj.close()

    print("\nGráficas almacenadas correctamente.\n")
    
def run_comparison(func_name, func):
    """
    Compara el desempeño de PSO, GA y SA en una función benchmark.
    Args:
        func_name: Nombre de la función benchmark
        func: Función objetivo a optimizar
    """
    print(f"   COMPARACIÓN PSO vs GA vs SA — {func_name.upper()}")

    dim = 10
    low, high = benchmark_bounds(func_name, dim)
    iters = 150
    
    print("\n=== EJECUTANDO PSO ===")
    print(f"Dimensión: {dim}")
    print(f"Rango permitido: low={low}, high={high}")
    print(f"Partículas: 50")
    print(f"Iteraciones: {iters}\n")

    pso = PSO(
        func, dim,
        bounds_low=low, bounds_high=high,
        n_particles=50, w=0.7, c1=1.5, c2=1.5
    )

    pso_history = []
    for _ in range(iters):
        pso.step()
        pso_history.append(pso.gbest_value)

    print(f"Mejor posición PSO: {np.round(pso.gbest, 6)}")
    print(f"Mejor valor PSO: {pso.gbest_value:.6f}\n")
    
    print("\n=== EJECUTANDO GA ===")
    print(f"Población inicial dentro del rango: [{low}, {high}]")
    print(f"Generaciones: {iters}\n")

    ga = SimpleGA(func, dim, low, high, pop_size=60)
    ga_best_pos, ga_best, ga_history = ga.run(generations=iters)

    print(f"Mejor cromosoma GA final: {np.round(ga_best_pos, 6)}")
    print(f"Mejor valor GA: {ga_best:.6f}\n")
    
    print("\n=== EJECUTANDO SA ===")
    print(f"Posición inicial aleatoria dentro de [{low}, {high}]")

    sa = SimpleSA(func, dim, low, high)
    sa_best_pos, sa_best, sa_history = sa.run()

    print(f"Mejor posición SA final: {np.round(sa_best_pos, 6)}")
    print(f"Mejor valor SA: {sa_best:.6f}\n")
    
    print("\n=========== RESULTADOS FINALES ===========")
    print(f"Función comparada: {func_name.upper()}")
    print(f"PSO → {pso.gbest_value:.6f}")
    print(f"GA  → {ga_best:.6f}")
    print(f"SA  → {sa_best:.6f}")
    
    # Gráfica comparativa
    plt.figure(figsize=(8, 5))
    plt.plot(pso_history, label="PSO")
    plt.plot(ga_history, label="GA")
    plt.plot(sa_history, label="SA")
    plt.title(f"Comparación PSO / GA / SA — {func_name.capitalize()}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Mejor valor encontrado")
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/comparacion_{func_name}.png")
    plt.close()

    print(f"\nGráfica guardada en results/comparacion_{func_name}.png\n")

def run_mmc_optimization():
    """
    Optimiza los parámetros de un sistema de colas M/M/c usando PSO.
    Encuentra la tasa de servicio (mu) y número de servidores (c) óptimos.
    """
    print("\n=== OPTIMIZACIÓN SISTEMA M/M/c ===")

    dim = 2
    low = np.array([0.1, 1])   # [mu_min, c_min]
    high = np.array([20, 20])  # [mu_max, c_max]

    pso = PSO(
        func=lambda x: objective_mmcc(x, arrival_rate=5, t_max=500),
        dim=dim,
        bounds_low=low,
        bounds_high=high,
        n_particles=30,
        w=0.7,
        c1=1.5,
        c2=1.5
    )

    for _ in range(80):
        pso.step()
        pso.history.append(pso.gbest_value)

    print(f"\nMejor μ = {pso.gbest[0]:.3f}")
    print(f"Mejor c = {int(round(pso.gbest[1]))}")
    print(f"Tiempo espera promedio = {pso.gbest_value:.3f}")

    conv = plot_convergence(pso.history, "Convergencia M/M/c")
    os.makedirs("results", exist_ok=True)
    conv.savefig("results/MMC_convergence.png")
    conv.close()

    print("\nGráfica almacenada en results/MMC_convergence.png\n")

def main_menu():
    """
    Menú principal interactivo del sistema de optimización.
    Permite ejecutar diferentes experimentos con PSO y algoritmos comparativos.
    """
    while True:
        print("MENÚ PRINCIPAL")
        print("="*30)
        print("1. Optimizar función (PSO)")
        print("2. Comparar PSO vs GA vs SA")
        print("3. Optimizar sistema M/M/c")
        print("4. Salir")

        op = input("Seleccione una opción: ")

        # Optimizar función con PSO
        if op == "1":
            print("\nSeleccione la función:")
            print("1. Sphere")
            print("2. Rastrigin")
            print("3. Ackley")
            print("4. Rosenbrock")
            f = input("Opción: ")

            if f == "1": run_pso_custom("sphere", sphere, dim=2)
            elif f == "2": run_pso_custom("rastrigin", rastrigin, dim=2)
            elif f == "3": run_pso_custom("ackley", ackley, dim=2)
            elif f == "4": run_pso_custom("rosenbrock", rosenbrock, dim=2)
            else:
                print("Opción inválida.")

        # Comparación de algoritmos
        elif op == "2":
            print("\nSeleccione la función para comparar:")
            print("1. Sphere")
            print("2. Rastrigin")
            print("3. Ackley")
            print("4. Rosenbrock")
            cmp_opt = input("Opción: ")

            if cmp_opt == "1":
                run_comparison("sphere", sphere)
            elif cmp_opt == "2":
                run_comparison("rastrigin", rastrigin)
            elif cmp_opt == "3":
                run_comparison("ackley", ackley)
            elif cmp_opt == "4":
                run_comparison("rosenbrock", rosenbrock)
            else:
                print("Opción no válida.")
                
        # Optimización M/M/c
        elif op == "3":
            run_mmc_optimization()

        elif op == "4":
            print("Saliendo.")
            break

        else:
            print("Opción inválida, intente nuevamente.")

# Ejecución principal
if __name__ == "__main__":
    main_menu()