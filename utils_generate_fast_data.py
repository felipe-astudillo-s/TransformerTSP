import numpy as np
import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from multiprocessing import Pool, cpu_count
import time

# --- CONFIGURACIÓN MASIVA (Estrategia Volumen) ---
DATA_DIR = "Data/Massive_NoLS" 
N_CORES = cpu_count()

# ¡Vamos a lo grande! 100k para Hard
CONFIGS = [
    {'name': 'Easy',   'nodes': 20,  'samples': 50000}, 
    {'name': 'Medium', 'nodes': 50,  'samples': 50000},
    {'name': 'Hard',   'nodes': 100, 'samples': 50000} 
]

def solve_fast(points):
    """Resuelve RAPIDÍSIMO usando solo Cheapest Arc"""
    num_nodes = len(points)
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_fn(from_idx, to_idx):
        p1, p2 = points[manager.IndexToNode(from_idx)], points[manager.IndexToNode(to_idx)]
        return int(np.linalg.norm(p1 - p2) * 100000)

    callback_idx = routing.RegisterTransitCallback(dist_fn)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    # LA CLAVE: Solo constructivo, sin metaheurística
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(params)
    
    if not solution: return None

    tour = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return np.array(tour)

def worker(args):
    idx, n_nodes, save_folder = args
    # Semilla única
    np.random.seed((os.getpid() * int(time.time())) % 123456789 + idx)
    
    points = np.random.rand(n_nodes, 2).astype(np.float32)
    solution = solve_fast(points)
    
    if solution is not None:
        filepath = os.path.join(save_folder, f"tsp_{n_nodes}_{idx}.npz")
        np.savez_compressed(filepath, points=points, solutions=solution)

def main():
    print(f"=== GENERACIÓN MASIVA (NO-LS) CON {N_CORES} NÚCLEOS ===")
    start_global = time.time()
    
    for config in CONFIGS:
        name = config['name']
        nodes = config['nodes']
        samples = config['samples']
        folder = os.path.join(DATA_DIR, name)
        os.makedirs(folder, exist_ok=True)
        
        print(f"\n--> Generando {samples} muestras para {name}...")
        tasks = [(i, nodes, folder) for i in range(samples)]
        
        with Pool(N_CORES) as p:
            p.map(worker, tasks)
            
    print(f"\n¡Listo! Tiempo total: {time.time() - start_global:.2f}s")

if __name__ == "__main__":
    main()