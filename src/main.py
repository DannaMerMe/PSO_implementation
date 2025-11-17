# run_experiments.py
from experiments.experiments import run_benchmark

if __name__ == "__main__":
    best, val, history = run_benchmark(name='rastrigin', dim=2, iters=200, n_particles=50, seed=42)
    print("Best:", best, "Value:", val)
    print("Gráficas guardadas en carpeta results/")
