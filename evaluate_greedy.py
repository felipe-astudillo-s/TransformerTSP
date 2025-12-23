import torch
import numpy as np
from tqdm import tqdm
import os
import sys
from scipy.spatial import distance_matrix

# --- ARREGLO DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Agregamos la carpeta padre por si acaso
sys.path.append(os.path.dirname(current_dir))

# Importamos TU modelo
from Models.model import TSPModel

# Importamos librerías de OR-Tools para comparar
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Usamos el modelo Ultimate
MODEL_PATH = os.path.join(current_dir, "Saved_Models", "tsp_transformer_Ultimate.pth") 

# --- 1. JUEZ OR-TOOLS (Referencia) ---
def solve_ortools_internal(points):
    """Resuelve un TSP usando OR-Tools (Solver Estándar)."""
    scale = 10000
    dist_matrix = (distance_matrix(points, points) * scale).astype(int)
    num_cities = len(points)

    manager = pywrapcp.RoutingIndexManager(num_cities, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.seconds = 2

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        return solution.ObjectiveValue() / scale
    return None

# --- 2. SOLVER GREEDY (El Modelo Básico) ---
def solve_greedy(model, points):
    """
    Resuelve el TSP eligiendo siempre la ciudad con mayor probabilidad (Argmax).
    Es rápido, pero no explora alternativas.
    """
    model.eval()
    points = points.to(DEVICE).unsqueeze(0) # [1, N, 2]
    B, N, _ = points.size()
    
    with torch.no_grad():
        enc_out = model.encoder(points)
        graph_emb = enc_out.mean(dim=1)
        
        # Empezamos en nodo 0
        curr_idx = torch.zeros(B, dtype=torch.long).to(DEVICE)
        mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
        mask[0, curr_idx] = True
        
        tour = [curr_idx]
        
        last_emb = enc_out[torch.arange(B), curr_idx, :]
        first_emb = last_emb
        
        for _ in range(N - 1):
            probs, logits = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
            
            # --- AQUÍ ESTÁ LA DIFERENCIA: GREEDY ---
            # Elegimos el máximo absoluto. No hay azar.
            next_idx = torch.argmax(probs, dim=1)
            
            tour.append(next_idx)
            mask[torch.arange(B), next_idx] = True
            last_emb = enc_out[torch.arange(B), next_idx, :]
            
        return torch.stack(tour, dim=1).squeeze(0).cpu().numpy()

def calculate_distance(points, tour):
    ordered = points[tour]
    dist = np.linalg.norm(ordered[1:] - ordered[:-1], axis=1).sum()
    dist += np.linalg.norm(ordered[0] - ordered[-1])
    return dist

# --- 3. EVALUACIÓN POR NIVELES ---
def run_greedy_benchmark():
    print("\n🐢 EVALUANDO MODO GREEDY (Línea Base)")
    print(f"   Cargando modelo: {os.path.basename(MODEL_PATH)}...")
    
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("   ✅ Modelo cargado correctamente.\n")
    except:
        print("   ❌ Error cargando modelo.")
        return

    # LISTA DE NIVELES A PROBAR
    LEVELS = [20, 50, 100]
    N_SAMPLES = 1000 # Muestras por nivel
    
    results = {}

    for N_NODES in LEVELS:
        print(f"--- Evaluando TSP-{N_NODES} ---")
        print(f"   Generando {N_SAMPLES} mapas aleatorios...")
        data = np.random.rand(N_SAMPLES, N_NODES, 2).astype(np.float32)
        
        gaps = []
        
        for i in tqdm(range(N_SAMPLES), desc=f"   TSP-{N_NODES}"):
            points = data[i]
            
            # 1. OR-Tools
            ort_dist = solve_ortools_internal(points)
            if ort_dist is None: ort_dist = 10.0
            
            # 2. Greedy Model
            tour_greedy = solve_greedy(model, torch.tensor(points))
            model_dist = calculate_distance(points, tour_greedy)
            
            # 3. Gap
            gap = ((model_dist - ort_dist) / ort_dist) * 100
            gaps.append(gap)
            
        avg_gap = np.mean(gaps)
        results[N_NODES] = avg_gap
        print(f"   Resultados Parciales TSP-{N_NODES}: Gap Promedio = {avg_gap:.2f}%\n")
    
    print("="*50)
    print("RESUMEN FINAL (MODO GREEDY)")
    print("-" * 50)
    for n_nodes, gap in results.items():
        print(f"TSP-{n_nodes:<3}: Gap Promedio = {gap:.2f}%")
    print("="*50)
    print("Nota: Estos Gaps altos son normales en Greedy.")
    print("Usando Sampling/Beam Search deberían bajar drásticamente.")

if __name__ == "__main__":
    run_greedy_benchmark()