# Optimización por Enjambre de Partículas


## Descripción general

Este proyecto implementa un sistema de optimización basado en:

- **PSO (Particle Swarm Optimization)**
- **GA (Genetic Algorithm)**
- **SA (Simulated Annealing)**
- **Optimización de sistemas de colas M/M/c**

Incluye un menú interactivo para ejecutar benchmarks, comparar algoritmos y optimizar un sistema de colas.

También incorpora un módulo de validación y manejo de excepciones, así como generación automática de gráficas de convergencia y trayectorias generadas por PSO.

---

# 1. Especificación de requisitos

## Requisitos funcionales

- Ejecutar PSO sobre funciones estándar: **Sphere**, **Rastrigin**, **Ackley**, **Rosenbrock**.
- Comparación de PSO vs GA vs SA.
- Optimización de sistemas de colas **M/M/c**.
- Solicitud interactiva de parámetros:  
  - número de partículas  
  - número de iteraciones  
  - w (inercia)  
  - c1 (coeficiente cognitivo)  
  - c2 (coeficiente social)
- Validación de parámetros mediante excepciones personalizadas.
- Generación automática de gráficas:
  - Convergencia del algoritmo  
  - Trayectorias de partículas en 2D  
  - Comparación PSO/GA/SA
- Guardado automático de resultados en la carpeta `/results`.

## Requisitos no funcionales

- Diseño modular y escalable.
- Uso de excepciones y validaciones robustas.
- Resultados reproducibles mediante semillas aleatorias.
- Gráficas generadas con Matplotlib.

## Requisitos de entorno

- Python **3.8+**
- Paquetes necesarios:
  - `numpy`
  - `matplotlib`
- Se recomienda entorno virtual (`venv`).

---

# 2. Arquitectura de la solución
```
project/
│
├── benchmarks/
│   ├── functions.py  # Funciones de benchmark y límites
│
├── pso/
│   ├── pso.py        # Implementación del algoritmo PSO
│
├── comparative/
│   ├── ga.py         # Algoritmo Genético (GA)
│   ├── sa.py         # Simulated Annealing (SA)
│
├── queueing/
│   ├── mmc_sim.py    # Modelo M/M/c y función objetivo para PSO
│
├── utils/
│   ├── validation.py # Validación de parámetros + excepciones personalizadas
│   ├── viz.py        # Gráficas de convergencia y trayectorias
│
├── results/          # Carpeta donde se guardan las imágenes generadas
│
├── main.py           # Menú interactivo principal del programa
└── README.md         # Documentación general del proyecto
```
---

# 3. Descripción de módulos

## 3.1. `benchmarks/functions.py`
Contiene las funciones clásicas de optimización:

- `Sphere(x)`
- `Rastrigin(x)`
- `Ackley(x)`
- `Rosenbrock(x)`

Incluye también:

- `benchmark_bounds(func_name, dim)`

---

## 3.2. `pso/pso.py`

Implementa el algoritmo **Particle Swarm Optimization**:

- Inicialización de partículas.
- Cálculo de best personal y global.
- Actualización de velocidad y posición.
- Parámetros:
  - `w` – inercia
  - `c1` – componente cognitiva
  - `c2` – componente social
- Historial de convergencia.

---

## 3.3. `comparative/ga.py`

Implementación simplificada de un **Algoritmo Genético**:

- Población inicial.
- Cruce.
- Mutación.
- Selección del mejor.
- Registro de convergencia.

---

## 3.4. `comparative/sa.py`

Implementación de **Simulated Annealing**:

- Generación de vecino.
- Enfriamiento gradual.
- Aceptación probabilística.
- Historial del mejor valor.

---

## 3.5. `queueing/mmc_sim.py`

Contiene la simulación del sistema de colas **M/M/c**:

- Cálculo de tiempos promedio en cola.
- Costo asociado al sistema.
- Función objetivo para ser optimizada mediante PSO.

---

## 3.6. `utils/viz.py`

Funciones para graficar:

- Convergencia del algoritmo.
- Trayectorias de partículas en 2D.

Los archivos se guardan automáticamente en `/results`.

---

## 3.7. `utils/validation.py`

Módulo para validación y manejo de excepciones:

```python
class ValidationError(Exception):
    pass
```
#  4. Manejo de excepciones



## Validaciones aplicadas en PSO

- Número de partículas → entero positivo  
- Número de iteraciones → entero positivo  
- Parámetros `w`, `c1`, `c2` → solo valores numéricos (float)

### Ejemplo:

```python
try:
    n_particles = validate_integer_parameter(
        input("Número de partículas: "), 
        "Número de partículas"
    )
except ValidationError as e:
    print(e)
    return
```
### En caso de error:

- El programa **NO** se cierra.  
- Vuelve al menú principal.

---
# Ejemplo de uso 

A continuación se muestra un ejemplo real de cómo funciona el sistema una vez ejecutado.

---

## 1. Ejecutar el programa

```bash
python main.py
```

Al iniciar aparece el menu principal

MENÚ PRINCIPAL
==============================
1. Optimizar función (PSO)
2. Comparar PSO vs GA vs SA
3. Optimizar sistema M/M/c
4. Salir

## 2. Seleccionar la opción de PSO
 - Seleccione una opción: 1
- Se muestra el menu de funciones
Seleccione la función:
1. Sphere
2. Rastrigin
3. Ackley
4. Rosenbrock

- Elegir función a optimizar: opción 2
## 3. Ingresar parametros personalizados
=== CONFIGURACIÓN PSO ===
Número de partículas: 40
Número de iteraciones: 100
Ingrese w (inercia): 0.7
Ingrese c1 (componente cognitiva): 1.5
Ingrese c2 (componente social): 1.5


##  Funciones evaluadas
- Sphere  
- Rastrigin  
- Ackley  
- Rosenbrock  

---

## Resultados típicos de ejecución

| Función     | Dim | Partículas | Iteraciones | Mejor valor |
|-------------|-----|------------|-------------|-------------|
| Sphere      | 2   | 80         | 300         | 0.0         |
| Rastrigin   | 2   | 80         | 100         | 0.0         |
| Ackley      | 2   | 40         | 150         | 0.0         |
| Rosenbrock  | 2   | 60         | 200         | 0.0         |

---

## Archivos generados (carpeta `/results`)

- `PSO_<func>_convergence.png`  
- `PSO_<func>_traj.png` (si `dim == 2`)  
- `comparacion_<func>.png`  
- `MMC_convergence.png`  

---

## Análisis del comportamiento

- PSO converge muy rápido en funciones suaves como **Sphere**.  
- En **Rastrigin**, la gran cantidad de mínimos locales dificulta la convergencia.  
- En **Ackley**, la convergencia es estable y progresiva.  
- **Rosenbrock** presenta un valle estrecho, pero PSO logra aproximarse a valores muy bajos.  

### Comparación de algoritmos:

- **PSO** → converge más rápido y con mayor estabilidad.  
- **GA** → más estable pero lento.  
- **SA** → mejor exploración pero converge tarde.

## Ejemplo de gráficas generadas 
<img width="744" height="558" alt="image" src="https://github.com/user-attachments/assets/d27a45f2-b5e4-48f2-8db8-d45ff8029dd7" />

<img width="694" height="521" alt="image" src="https://github.com/user-attachments/assets/0425ca76-7b1e-46d9-8b5e-a6095ebc0752" />

---
# 5. Guía de ejecución

## 1. Crear entorno virtual

```bash
python -m venv venv
```

## 2. Activar entorno
### Windows
```bash
.\venv\Scripts\activate
```

### Linux/Mac
```
source venv/bin/activate
```

## 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
