import math
import os
import numpy as np
from tqdm import tqdm
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import concurrent.futures # LIBRERÍA MAGICA PARA PARALELISMO
import multiprocessing

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_PATH = "TransformerTSP/data/Validation"
os.makedirs(BASE_PATH, exist_ok=True)

# Detectamos cuántos núcleos tiene tu PC 
NUM_CORES = multiprocessing.cpu_count()

# ==========================================
# 1. FUNCIÓN DE RESOLUCIÓN AISLADA
# ==========================================
# Esta función debe estar "suelta" (no dentro de una clase) para que
# Python pueda enviarla a los otros núcleos fácilmente.

def solve_instance_worker(args):
    """
    Recibe: (n_nodes, time_limit, scaling_factor)
    Retorna: (points, tour)
    """
    n_nodes, time_limit, scale = args
    
    # Generar puntos aleatorios (Cada núcleo genera los suyos)
    points = np.random.rand(n_nodes, 2)
    
    # --- LÓGICA OR-TOOLS ---
    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        x1, y1 = points[from_node]
        x2, y2 = points[to_node]
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return int(dist * scale)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit

    # Resolver
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return points, tour
    else:
        return None, None

# ==========================================
# 2. GENERADOR PARALELO
# ==========================================

def generate_dataset_parallel(config_list, chunk_size=1000):
    
    print(f"🔥 INICIANDO MODO TURBO CON {NUM_CORES} NÚCLEOS 🔥")

    for config in config_list:
        base_filename = config['name'].replace('.npz', '')
        n_nodes = config['n_nodes']
        total_samples = config['samples']
        time_limit = config.get('time_limit', 1)

        print(f"\n🚀 Tarea: {base_filename} (N={n_nodes}) | Meta: {total_samples}")

        # Buffers
        x_chunk = []
        y_chunk = []
        part_counter = 0
        samples_collected = 0

        pbar = tqdm(total=total_samples, desc=f"Generando {base_filename}")

        # Creamos el Pool de procesos
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            
            while samples_collected < total_samples:
                # 1. Verificar si ya existe el archivo
                part_filename = f"{base_filename}_part_{part_counter}.npz"
                full_path = os.path.join(BASE_PATH, part_filename)

                if os.path.exists(full_path):
                    samples_collected += chunk_size
                    part_counter += 1
                    pbar.update(chunk_size)
                    continue

                # 2. Preparar un lote de tareas para enviar a los núcleos
                # Enviamos 'chunk_size' tareas al pool
                tasks = [(n_nodes, time_limit, 10000) for _ in range(chunk_size)]
                
                # 3. Ejecutar en paralelo y esperar resultados
                # executor.map reparte las tareas entre los hilos
                results = list(executor.map(solve_instance_worker, tasks))

                # 4. Procesar resultados
                current_batch_points = []
                current_batch_sols = []
                
                for pts, tour in results:
                    if tour is not None:
                        current_batch_points.append(pts)
                        current_batch_sols.append(tour)

                # 5. Guardar en disco (El hilo principal hace esto)
                if len(current_batch_points) > 0:
                    x_np = np.array(current_batch_points)
                    y_np = np.array(current_batch_sols, dtype=object)
                    
                    np.savez_compressed(full_path, points=x_np, solutions=y_np)
                    
                    samples_collected += len(current_batch_points)
                    part_counter += 1
                    pbar.update(len(current_batch_points))

        pbar.close()

# ==========================================
# 3. EJECUCIÓN
# ==========================================

validation_tasks = [
    {"name": "tsp_easy_val.npz", "n_nodes": 20, "samples": 1000, "time_limit": 1},
    {"name": "tsp_medium_val.npz", "n_nodes": 50, "samples": 1000, "time_limit": 2},
    {"name": "tsp_hard_val.npz", "n_nodes": 100, "samples": 500, "time_limit": 4}
]

if __name__ == "__main__":
    # Windows necesita esta protección en el main
    generate_dataset_parallel(validation_tasks, chunk_size=100)