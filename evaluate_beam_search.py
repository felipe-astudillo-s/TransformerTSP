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

# Importamos librerías de OR-Tools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ruta al modelo Ultimate
MODEL_PATH = os.path.join(current_dir, "Saved_Models", "tsp_transformer_Ultimate.pth") 

# ANCHO DEL HAZ (BEAM WIDTH)
# 3 es rápido, 5 es balanceado, 10 es muy preciso pero lento.
BEAM_WIDTH = 3  

# --- 1. EL JUEZ INTEGRADO (OR-TOOLS STANDARD) ---
def solve_ortools_standard(points):
    """
    Resuelve un TSP usando OR-Tools (Solver Estándar, sin GLS).
    Sirve como referencia rápida.
    """
    scale = 10000
    # Usamos scipy para la matriz de distancias rápida
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
    search_parameters.time_limit.seconds = 2 # Tiempo límite por mapa

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return solution.ObjectiveValue() / scale
    return None

# --- 2. LÓGICA DE BEAM SEARCH ---
def get_tour_distance_np(points, tour_idx):
    """Calcula distancia de un tour (numpy)."""
    ordered = points[tour_idx]
    dist = np.linalg.norm(ordered[1:] - ordered[:-1], axis=1).sum()
    dist += np.linalg.norm(ordered[0] - ordered[-1]) # Cerrar ciclo
    return dist

def beam_search_tour(model, points, beam_width=3):
    """Inferencia inteligente explorando 'beam_width' caminos simultáneamente."""
    model.eval()
    points = points.to(DEVICE).unsqueeze(0) # [1, N, 2]
    B, N, _ = points.size()
    
    with torch.no_grad():
        enc_out = model.encoder(points) # [1, N, Emb]
        graph_emb = enc_out.mean(dim=1)
        
        # Inicio: Nodo 0
        start_node = 0
        mask = torch.zeros(1, N, dtype=torch.bool).to(DEVICE)
        mask[0, start_node] = True
        
        # Candidatos: Lista de tuplas (log_prob_acumulada, lista_tour, mask_tensor)
        beams = [(0.0, [start_node], mask)]
        
        for _ in range(N - 1):
            new_candidates = []
            
            for score, tour, msk in beams:
                curr_last_node = tour[-1]
                
                # Preparar inputs para el decoder
                last_emb = enc_out[0, curr_last_node, :].unsqueeze(0) # [1, Emb]
                first_emb = enc_out[0, tour[0], :].unsqueeze(0)
                
                # Predecir siguiente paso
                probs, logits = model.decoder(enc_out, graph_emb, last_emb, first_emb, msk)
                
                # Convertir a Log Probs para sumar score
                log_probs = torch.log(probs + 1e-9).squeeze(0) # [N]
                
                # Tomar los mejores 'k' locales para este beam
                # (Optimizamos tomando un poco más que beam_width para tener variedad)
                top_vals, top_idxs = torch.topk(log_probs, k=beam_width)
                
                for v, i in zip(top_vals, top_idxs):
                    idx = i.item()
                    if msk[0, idx]: continue 
                    
                    new_score = score + v.item()
                    new_tour = tour + [idx]
                    new_mask = msk.clone()
                    new_mask[0, idx] = True
                    
                    new_candidates.append((new_score, new_tour, new_mask))
            
            # Ordenar TODOS los candidatos globales y podar
            # Mayor score es mejor (log prob cercana a 0)
            new_candidates.sort(key=lambda x: x[0], reverse=True) 
            beams = new_candidates[:beam_width]
            
        # Al final, elegimos el mejor tour basado en DISTANCIA REAL (Euclidiana)
        # no solo en probabilidad acumulada
        best_dist = float('inf')
        points_np = points.squeeze(0).cpu().numpy()
        
        for _, tour, _ in beams:
            d = get_tour_distance_np(points_np, np.array(tour))
            if d < best_dist:
                best_dist = d
        
        return best_dist

# --- 3. AUDITORÍA MASIVA POR NIVELES ---
def run_beam_benchmark_massive():
    print(f"\n🔦 INICIANDO BENCHMARK BEAM SEARCH (Width={BEAM_WIDTH})")
    print(f"   Modelo: {os.path.basename(MODEL_PATH)}")
    print("-" * 60)
    
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("   ✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"   ❌ Error cargando modelo: {e}")
        return

    # CONFIGURACIÓN DE LA PRUEBA
    LEVELS = [20, 50, 100]
    N_SAMPLES = 1000  # 1000 muestras por nivel
    
    results = {}
    
    for n_nodes in LEVELS:
        print(f"\n📊 Evaluando Nivel: TSP-{n_nodes} ({N_SAMPLES} casos)...")
        # Generar datos frescos
        data = np.random.rand(N_SAMPLES, n_nodes, 2).astype(np.float32)
        gaps = []
        
        # Barra de progreso
        for i in tqdm(range(N_SAMPLES), desc=f"   Procesando TSP-{n_nodes}"):
            points = data[i]
            
            # A. OR-Tools (Referencia)
            dist_ort = solve_ortools_standard(points)
            if dist_ort is None: continue
            
            # B. Beam Search (Modelo)
            # Pasamos tensor a la GPU dentro de la función
            dist_model = beam_search_tour(model, torch.tensor(points), beam_width=BEAM_WIDTH)
            
            # C. Gap
            gap = ((dist_model - dist_ort) / dist_ort) * 100
            gaps.append(gap)
        
        avg_gap = np.mean(gaps)
        results[n_nodes] = avg_gap
        print(f"   ↳ Gap Promedio TSP-{n_nodes}: {avg_gap:.2f}%")
        
    print("\n" + "="*60)
    print(f"🏁 RESUMEN FINAL: BEAM SEARCH (Width={BEAM_WIDTH})")
    print("="*60)
    print(f"{'DIFICULTAD':<15} | {'MUESTRAS':<10} | {'GAP PROMEDIO':<15}")
    print("-" * 60)
    
    for n_nodes, gap in results.items():
        print(f"TSP-{n_nodes:<11} | {N_SAMPLES:<10} | {gap:.2f}%")
        
    print("-" * 60)
    print("Interpretación:")
    print(" - < 5.00%: Resultado Profesional (Mejor que heurísticas simples).")
    print(" - < 1.00%: Estado del Arte (Competitivo con solvers avanzados).")
    print("="*60)

if __name__ == "__main__":
    run_beam_benchmark_massive()