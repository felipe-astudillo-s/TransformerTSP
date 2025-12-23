import torch
import numpy as np
from tqdm import tqdm
import os
import sys
from scipy.spatial import distance_matrix

# --- ARREGLO DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from Models.model import TSPModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# USA TU MEJOR MODELO (El "Ultimate" original con GLS)
MODEL_PATH = os.path.join(current_dir, "Saved_Models", "tsp_transformer_Ultimate.pth") 

# PARÁMETROS DE INFERENCIA
# 500 Samples dio el resultado de 0.57% (SOTA).
N_SAMPLES_PER_INSTANCE = 500  

# --- 1. JUEZ OR-TOOLS (STANDARD) ---
def solve_ortools_standard(points):
    """
    Solver Estándar (Sin GLS) para comparar en igualdad de condiciones.
    """
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

# --- 2. LÓGICA DE SAMPLING ---
def get_tour_distance_batch(points, tour):
    """Calcula distancias de un batch de tours en paralelo."""
    batch, n, _ = points.shape
    idx = tour.unsqueeze(2).expand(-1, -1, 2)
    ordered = torch.gather(points, 1, idx) # [B, N, 2]
    
    diff = ordered[:, 1:] - ordered[:, :-1]
    dist = torch.norm(diff, dim=2).sum(dim=1)
    
    return_dist = torch.norm(ordered[:, 0] - ordered[:, -1], dim=1)
    return dist + return_dist

def solve_with_sampling(model, points_single, n_samples=100):
    """
    Genera 'n_samples' soluciones probabilísticas para UN mapa 
    y devuelve la mejor distancia encontrada.
    """
    model.eval()
    
    # Replicamos el MISMO mapa N veces para procesarlo en paralelo en la GPU
    # shape: [N_Samples, N_Cities, 2]
    points = points_single.unsqueeze(0).repeat(n_samples, 1, 1).to(DEVICE)
    
    B, N, _ = points.size()
    
    with torch.no_grad():
        enc_out = model.encoder(points)
        graph_emb = enc_out.mean(dim=1)
        
        # Empezamos todos en 0
        curr_idx = torch.zeros(B, dtype=torch.long).to(DEVICE)
        mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
        mask[torch.arange(B), curr_idx] = True
        
        tour = [curr_idx]
        
        last_emb = enc_out[torch.arange(B), curr_idx, :]
        first_emb = last_emb
        
        for _ in range(N - 1):
            probs, logits = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
            
            # SAMPLING: Tirar los dados
            m = torch.distributions.Categorical(probs)
            next_idx = m.sample()
            
            tour.append(next_idx)
            mask[torch.arange(B), next_idx] = True
            last_emb = enc_out[torch.arange(B), next_idx, :]
            
        tour_tensor = torch.stack(tour, dim=1) # [B, N]
        
        # Calculamos distancias de los N intentos
        dists = get_tour_distance_batch(points, tour_tensor)
        
        # Nos quedamos con la MEJOR (mínima distancia)
        best_dist, best_idx = torch.min(dists, dim=0)
        
        return best_dist.item()

# --- 3. AUDITORÍA MASIVA POR NIVELES ---
def run_sampling_benchmark_massive():
    print(f"\n🎲 INICIANDO BENCHMARK SAMPLING (Samples={N_SAMPLES_PER_INSTANCE})")
    print(f"   Modelo: {os.path.basename(MODEL_PATH)}")
    print("-" * 60)
    
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("   ✅ Modelo cargado correctamente.")
    except:
        print("   ❌ Error cargando modelo.")
        return

    # CONFIGURACIÓN
    LEVELS = [20, 50, 100]
    N_TEST_CASES = 1000 # 1000 Mapas por nivel
    
    results = {}
    
    for n_nodes in LEVELS:
        print(f"\n📊 Evaluando Nivel: TSP-{n_nodes} ({N_TEST_CASES} mapas)...")
        
        # Generamos datos frescos
        data = np.random.rand(N_TEST_CASES, n_nodes, 2).astype(np.float32)
        gaps = []
        
        # Barra de progreso
        for i in tqdm(range(N_TEST_CASES), desc=f"   Procesando TSP-{n_nodes}"):
            points_np = data[i]
            points_tensor = torch.tensor(points_np)
            
            # 1. OR-Tools (Standard)
            ort_val = solve_ortools_standard(points_np)
            if ort_val is None: continue
            
            # 2. Modelo (Sampling)
            # Pasamos tensor a la función
            model_val = solve_with_sampling(model, points_tensor, n_samples=N_SAMPLES_PER_INSTANCE)
            
            # 3. Gap
            gap = ((model_val - ort_val) / ort_val) * 100
            gaps.append(gap)
            
        avg_gap = np.mean(gaps)
        results[n_nodes] = avg_gap
        print(f"   ↳ Gap Promedio TSP-{n_nodes}: {avg_gap:.2f}%")
    
    print("\n" + "="*60)
    print(f"🏁 RESUMEN FINAL: SAMPLING ({N_SAMPLES_PER_INSTANCE} intents)")
    print("="*60)
    print(f"{'DIFICULTAD':<15} | {'MUESTRAS':<10} | {'GAP PROMEDIO':<15}")
    print("-" * 60)
    
    for n_nodes, gap in results.items():
        print(f"TSP-{n_nodes:<11} | {N_TEST_CASES:<10} | {gap:.2f}%")
        
    print("-" * 60)
    print("Interpretación Esperada:")
    print(" - TSP-20/50: Debería ser increíblemente bajo (< 1%).")
    print(" - TSP-100: Aquí Sampling suele sufrir (Gap alto) por la explosión combinatoria.")
    print("   (Para TSP-100, confía más en los resultados del Beam Search).")
    print("="*60)

if __name__ == "__main__":
    run_sampling_benchmark_massive()