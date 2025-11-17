# queueing/mmc_sim.py
import heapq
import numpy as np
from typing import Tuple

class MMC:
    """
    Simulador de sistema de colas M/M/c.
    Modela llegadas Poisson, tiempos de servicio exponenciales y c servidores.
    """
    
    def __init__(self, arrival_rate: float, service_rate: float, servers: int, rng_seed: int = 42):
        """
        Inicializa el simulador M/M/c.
        Args:
            arrival_rate: Tasa de llegadas (lambda)
            service_rate: Tasa de servicio por servidor (mu)
            servers: Número de servidores (c)
            rng_seed: Semilla para reproducibilidad
        """
        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.c = servers
        self.rng = np.random.default_rng(rng_seed)
    
    def simulate(self, t_max: float = 10000.0) -> Tuple[float, float]:
        """
        Simula el sistema hasta el tiempo t_max.
        Args:
            t_max: Horizonte de tiempo de simulación
        Returns:
            Tupla con (tiempo_espera_promedio, utilización_servidores)
        """
        # Cola de eventos: (tiempo, tipo, id)
        t = 0.0
        event_queue = []
        
        # Programar primera llegada
        first_arrival = t + self.rng.exponential(1/self.lambda_)
        heapq.heappush(event_queue, (first_arrival, 'arrival', None))
        
        n_in_system = 0
        busy_servers = 0
        queue = []  # Tiempos de llegada esperando
        total_wait = 0.0
        n_served = 0
        area_busy = 0.0
        last_t = 0.0
        
        while event_queue:
            event_time, ev_type, _ = heapq.heappop(event_queue)
            if event_time > t_max:
                break
            
            # Actualizar estadísticas de tiempo promedio
            area_busy += busy_servers * (event_time - last_t)
            last_t = event_time
            t = event_time
            
            if ev_type == 'arrival':
                # Programar siguiente llegada
                next_arrival = t + self.rng.exponential(1/self.lambda_)
                heapq.heappush(event_queue, (next_arrival, 'arrival', None))
                n_in_system += 1
                
                if busy_servers < self.c:
                    # Comenzar servicio inmediatamente
                    busy_servers += 1
                    service_time = self.rng.exponential(1/self.mu)
                    departure = t + service_time
                    heapq.heappush(event_queue, (departure, 'departure', None))
                else:
                    # Unirse a la cola
                    queue.append(t)
                    
            elif ev_type == 'departure':
                n_in_system -= 1
                n_served += 1
                
                if queue:
                    arrival_time = queue.pop(0)
                    wait = t - arrival_time
                    total_wait += wait
                    # Comenzar servicio para cliente en cola
                    service_time = self.rng.exponential(1/self.mu)
                    departure = t + service_time
                    heapq.heappush(event_queue, (departure, 'departure', None))
                else:
                    busy_servers -= 1
        
        avg_wait = total_wait / max(1, n_served)
        utilization = area_busy / t_max
        return avg_wait, utilization

def objective_mmcc(params: np.ndarray, arrival_rate: float = 5.0, t_max: float = 2000.0, rng_seed: int = 42) -> float:
    """
    Función objetivo para optimizar parámetros M/M/c con PSO.
    Args:
        params: Array con [mu, c] donde c puede ser no entero (se redondeará)
        arrival_rate: Tasa de llegadas al sistema
        t_max: Horizonte de simulación
        rng_seed: Semilla para reproducibilidad
    Returns:
        Tiempo de espera promedio más penalización por número de servidores
    """
    mu = max(1e-6, float(params[0]))
    c = int(max(1, round(float(params[1]))))
    
    sim = MMC(arrival_rate=arrival_rate, service_rate=mu, servers=c, rng_seed=rng_seed)
    avg_wait, util = sim.simulate(t_max=t_max)
    
    # Penalización para desincentivar muchos servidores (trade-off)
    penalty = 0.01 * c
    
    return avg_wait + penalty