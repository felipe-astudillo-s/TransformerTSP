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

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RUTA DEL MODELO (Asegúrate de que sea el "Ultimate" entrenado con GLS)
MODEL_PATH = os.path.join(current_dir, "Saved_Models", "tsp_transformer_Ultimate.pth")

# PARÁMETROS
N_NODES = 50            # Tamaño del problema (donde vimos el éxito)
N_TEST_CASES = 1000     # ¡1000 casos de prueba!
N_SAMPLES = 500         # Intentos de Sampling por mapa (La creatividad del alumno)

# --- 1. JUEZ OR-TOOLS (ESTÁNDAR) ---
def solve_ortools_standard(points):
    """
    Solver 'Standard' de OR-Tools.
    Estrategia: Path Cheapest Arc + Local Search Básico (Default).
    NO usamos Guided Local Search aquí para simular el solver estándar rápido.
    """
    scale = 10000
    dist_matrix = (distance_matrix(points, points) * scale).astype(int)
    num_cities = len(points)

    manager = pywrapcp.RoutingIndexManager(num_cities, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(
        lambda i, j: dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Estrategia constructiva inicial rápida
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # Límite de tiempo estricto (Solver Rápido)
    search_parameters.time_limit.seconds = 2

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        return solution.ObjectiveValue() / scale
    return None

# --- 2. MODELO (SAMPLING) ---
def get_tour_distance_batch(points, tour):
    # Cálculo vectorizado de distancias para N samples
    idx = tour.unsqueeze(2).expand(-1, -1, 2)
    ordered = torch.gather(points, 1, idx)
    diff = ordered[:, 1:] - ordered[:, :-1]
    dist = torch.norm(diff, dim=2).sum(dim=1)
    return_dist = torch.norm(ordered[:, 0] - ordered[:, -1], dim=1)
    return dist + return_dist

def solve_with_sampling(model, points_single, n_samples=100):
    model.eval()
    # Replicamos el mapa N veces
    points = points_single.unsqueeze(0).repeat(n_samples, 1, 1).to(DEVICE)
    B, N, _ = points.size()
    
    with torch.no_grad():
        enc_out = model.encoder(points)
        graph_emb = enc_out.mean(dim=1)
        
        curr_idx = torch.zeros(B, dtype=torch.long).to(DEVICE)
        mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
        mask[torch.arange(B), curr_idx] = True
        tour = [curr_idx]
        
        last_emb = enc_out[torch.arange(B), curr_idx, :]
        first_emb = last_emb
        
        for _ in range(N - 1):
            probs, _ = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
            # Sampling: Tirar los dados basado en probabilidad
            m = torch.distributions.Categorical(probs)
            next_idx = m.sample()
            
            tour.append(next_idx)
            mask[torch.arange(B), next_idx] = True
            last_emb = enc_out[torch.arange(B), next_idx, :]
            
        tour_tensor = torch.stack(tour, dim=1)
        dists = get_tour_distance_batch(points, tour_tensor)
        
        # Nos quedamos con la mejor de las 500
        best_dist, _ = torch.min(dists, dim=0)
        return best_dist.item()

# --- 3. AUDITORÍA MASIVA ---
def run_massive_validation():
    print(f"\n📊 INICIANDO AUDITORÍA MASIVA ({N_TEST_CASES} Casos)")
    print(f"   Modo: TSP-{N_NODES} | Sampling: {N_SAMPLES} intentos")
    print("-" * 60)
    
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("   ✅ Modelo cargado.")
    except:
        print("   ❌ Error cargando modelo.")
        return

    # Estadísticas
    wins = 0      # Gap < 0
    draws = 0     # Gap == 0 (aprox)
    losses = 0    # Gap > 0
    
    gaps = []
    
    # Generar datos
    print("   Generando datos aleatorios...")
    data = np.random.rand(N_TEST_CASES, N_NODES, 2).astype(np.float32)
    
    print("   🚀 Ejecutando benchmark...")
    for i in tqdm(range(N_TEST_CASES)):
        points_np = data[i]
        points_tensor = torch.tensor(points_np)
        
        # 1. OR-Tools (Juez)
        ort_val = solve_ortools_standard(points_np)
        if ort_val is None: continue # Skip si falla (raro)
        
        # 2. Modelo (Aspirante)
        model_val = solve_with_sampling(model, points_tensor, n_samples=N_SAMPLES)
        
        # 3. Calcular Gap
        gap = ((model_val - ort_val) / ort_val) * 100
        gaps.append(gap)
        
        # 4. Clasificar resultado (con pequeña tolerancia flotante)
        if gap < -0.001:
            wins += 1
        elif gap > 0.001:
            losses += 1
        else:
            draws += 1

    gaps = np.array(gaps)
    mean_gap = np.mean(gaps)
    min_gap = np.min(gaps)  # La mayor paliza que dio el modelo
    max_gap = np.max(gaps)  # La peor derrota del modelo
    
    print("\n" + "="*60)
    print("🏁 RESULTADOS DE LA AUDITORÍA (1000 MAPAS)")
    print("="*60)
    print(f"🏆 VICTORIAS DEL MODELO (Gap < 0): {wins}  ({(wins/N_TEST_CASES)*100:.1f}%)")
    print(f"🤝 EMPATES (Gap ≈ 0):             {draws}  ({(draws/N_TEST_CASES)*100:.1f}%)")
    print(f"🐢 DERROTAS (Gap > 0):            {losses}  ({(losses/N_TEST_CASES)*100:.1f}%)")
    print("-" * 60)
    print(f"📉 GAP PROMEDIO GLOBAL:           {mean_gap:.4f}%")
    print(f"💎 MEJOR GAP (Mayor Victoria):    {min_gap:.4f}%")
    print(f"💩 PEOR GAP (Peor Derrota):       {max_gap:.4f}%")
    print("="*60)

    if mean_gap < 0:
        print("✅ CONCLUSIÓN: El modelo es SUPERIOR en promedio al Solver Estándar.")
    elif mean_gap < 1.0:
        print("⚠️ CONCLUSIÓN: El modelo es COMPETITIVO (Empate técnico).")
    else:
        print("❌ CONCLUSIÓN: OR-Tools sigue siendo el rey en promedio.")

if __name__ == "__main__":
    run_massive_validation()