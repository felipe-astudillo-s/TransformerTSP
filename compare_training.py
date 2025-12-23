import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix

# --- 1. ARREGLO DE IMPORTACIONES ---
# Esto asegura que Python encuentre tu carpeta Models sin importar donde estés
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Agregamos la carpeta padre también por seguridad
sys.path.append(os.path.dirname(current_dir))

# Importamos TU modelo
from Models.model import TSPModel

# Importamos OR-Tools (El Juez)
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ¡¡¡ IMPORTANTE: VERIFICA ESTOS NOMBRES !!!
# Asegúrate de que renombraste tu modelo viejo a ..._GLS.pth
PATH_MODEL_GLS  = os.path.join(current_dir, "Saved_Models", "tsp_transformer_Ultimate.pth")
PATH_MODEL_NOLS = os.path.join(current_dir, "Saved_Models", "tsp_transformer_UltimateNoLocalSearch.pth")

# --- 2. EL JUEZ (OR-TOOLS INTEGRADO) ---
def solve_ortools_internal(points):
    """Resuelve un TSP usando OR-Tools (Sin depender de archivos externos)."""
    # Escala para precisión entera
    scale = 10000
    dist_matrix = (distance_matrix(points, points) * scale).astype(int)
    num_cities = len(points)

    # Configuración del Routing Model
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
    # Límite de tiempo breve por muestra para no eternizarnos
    search_parameters.time_limit.seconds = 2 

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return solution.ObjectiveValue() / scale
    return None

# --- 3. FUNCIONES DE EVALUACIÓN ---
def get_tour_length(points, tour):
    """Calcula la distancia total del tour predicho por el modelo."""
    # points: [Batch, N, 2]
    # tour: [Batch, N]
    
    # Reordenamos los puntos según el tour predicho
    # Truco de PyTorch para reordenar usando indices
    batch_size, n_nodes, _ = points.shape
    idx = tour.unsqueeze(2).expand(-1, -1, 2)
    ordered_points = torch.gather(points, 1, idx)
    
    # Distancia paso a paso
    diff = ordered_points[:, 1:] - ordered_points[:, :-1]
    dist = torch.norm(diff, dim=2).sum(dim=1)
    
    # + Regreso al inicio
    return_dist = torch.norm(ordered_points[:, 0] - ordered_points[:, -1], dim=1)
    
    return dist + return_dist

def evaluate_model_greedy(model, points):
    """Ejecuta el modelo Transformer en modo Greedy."""
    model.eval()
    points = points.to(DEVICE)
    B, N, _ = points.size()
    
    with torch.no_grad():
        enc_out = model.encoder(points)
        graph_emb = enc_out.mean(dim=1)
        
        # Estado Inicial
        mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
        curr_idx = torch.zeros(B, dtype=torch.long).to(DEVICE)
        
        tour = [curr_idx]
        mask[torch.arange(B), curr_idx] = True
        
        last_emb = enc_out[torch.arange(B), curr_idx, :]
        first_emb = last_emb
        
        for _ in range(N - 1):
            probs, _ = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
            
            # GREEDY: Elige la probabilidad más alta
            next_idx = torch.argmax(probs, dim=1)
            
            tour.append(next_idx)
            mask[torch.arange(B), next_idx] = True
            last_emb = enc_out[torch.arange(B), next_idx, :]
            
        return torch.stack(tour, dim=1)

# --- 4. LA BATALLA ---
def run_battle(n_nodes, n_samples=100):
    print(f"\n⚔️  COMBATE: TSP-{n_nodes} ({n_samples} muestras) ⚔️")
    
    # A. Generar Datos Frescos
    points_np = np.random.rand(n_samples, n_nodes, 2).astype(np.float32)
    points_tensor = torch.tensor(points_np)
    
    # B. Cargar Modelos
    model_gls = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    model_nols = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    
    models_ready = True
    try:
        model_gls.load_state_dict(torch.load(PATH_MODEL_GLS, map_location=DEVICE))
    except Exception as e:
        print(f"⚠️ Error cargando GLS: {e}")
        models_ready = False

    try:
        model_nols.load_state_dict(torch.load(PATH_MODEL_NOLS, map_location=DEVICE))
    except Exception as e:
        print(f"⚠️ Error cargando NoLS: {e}")
        models_ready = False
        
    if not models_ready:
        print("❌ No se pudieron cargar ambos modelos. Abortando.")
        return

    # C. Turno: Modelo GLS (Viejo)
    print("   🤖 Evaluando Modelo GLS (Viejo)...")
    tour_gls = evaluate_model_greedy(model_gls, points_tensor)
    len_gls = get_tour_length(points_tensor.to(DEVICE), tour_gls).cpu().numpy()
    
    # D. Turno: Modelo NoLS (Nuevo)
    print("   🚀 Evaluando Modelo NoLS (Nuevo)...")
    tour_nols = evaluate_model_greedy(model_nols, points_tensor)
    len_nols = get_tour_length(points_tensor.to(DEVICE), tour_nols).cpu().numpy()
    
    # E. Turno: OR-Tools (Juez)
    print("   👨‍⚖️ Evaluando OR-Tools (Juez)...")
    len_ortools = []
    for i in tqdm(range(n_samples), leave=False):
        dist = solve_ortools_internal(points_np[i])
        if dist is None: dist = 100.0 # Castigo por fallo
        len_ortools.append(dist)
    len_ortools = np.array(len_ortools)

    # --- RESULTADOS ---
    mean_ort = np.mean(len_ortools)
    mean_gls = np.mean(len_gls)
    mean_nols = np.mean(len_nols)
    
    gap_gls = ((mean_gls - mean_ort) / mean_ort) * 100
    gap_nols = ((mean_nols - mean_ort) / mean_ort) * 100
    
    print("-" * 75)
    print(f"{'MÉTODO':<25} | {'COSTO PROM.':<12} | {'GAP vs OR-TOOLS':<15}")
    print("-" * 75)
    print(f"{'OR-Tools (Base)':<25} | {mean_ort:.4f}       | 0.00%")
    print(f"{'Modelo GLS (Viejo)':<25} | {mean_gls:.4f}       | {gap_gls:+.2f}%")
    print(f"{'Modelo NoLS (Nuevo)':<25} | {mean_nols:.4f}       | {gap_nols:+.2f}%")
    print("-" * 75)
    
    diff = gap_gls - gap_nols
    if diff > 0:
        print(f"🏆 GANADOR: Modelo NUEVO (NoLS) por {diff:.2f}% de mejora")
    else:
        print(f"🐢 GANADOR: Modelo VIEJO (GLS) por {abs(diff):.2f}% de ventaja")

if __name__ == "__main__":
    # Prueba rápida
    run_battle(n_nodes=20, n_samples=200)
    run_battle(n_nodes=50, n_samples=100)
    # run_battle(n_nodes=100, n_samples=50) # Descomenta para Hard