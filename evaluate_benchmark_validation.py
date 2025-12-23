import torch
import numpy as np
import os
import sys
import glob
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from Models.model import TSPModel

# --- PARÁMETROS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(current_dir, "Saved_Models", "tsp_transformer_Ultimate.pth")
VALIDATION_PATH = os.path.join(current_dir, "Data", "Validation")

# LÍMITE DE PRUEBAS
MAX_SAMPLES = 1000 

# CONFIGURACIÓN DE MÉTODOS
BEAM_WIDTH = 5
SAMPLING_SAMPLES = 500

# --- UTILS DE CARGA DE DATOS ---
def load_val_data(level_name, limit=None):
    """Carga datos reales de validación (Points + Soluciones Óptimas)."""
    search_path = os.path.join(VALIDATION_PATH, level_name, "*.npz")
    files = glob.glob(search_path)
    
    if not files:
        print(f"⚠️ No se encontraron archivos en {search_path}")
        return None, None

    all_points = []
    all_gt_sols = []
    total_loaded = 0
    
    print(f"   📂 Cargando datos de {level_name}...")
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            pts = data['points']
            sols = data['solutions']
            
            # --- CORRECCIÓN AQUÍ: Forzamos float32 siempre ---
            pts = pts.astype(np.float32)
            if sols.dtype == object: sols = sols.astype(np.int64)
            else: sols = sols.astype(np.int64) # Asegurar int64
            
            all_points.append(pts)
            all_gt_sols.append(sols)
            total_loaded += len(pts)
            
            if limit and total_loaded >= limit:
                break
        except Exception as e:
            print(f"Error leyendo {f}: {e}")

    if not all_points: return None, None
    
    # Concatenar y recortar al límite exacto
    X = np.concatenate(all_points, axis=0)
    Y = np.concatenate(all_gt_sols, axis=0)
    
    if limit:
        X = X[:limit]
        Y = Y[:limit]
        
    return X, Y

def calc_dist(points, tour):
    """Calcula distancia euclidiana de un tour."""
    ordered = points[tour]
    d = np.linalg.norm(ordered[1:] - ordered[:-1], axis=1).sum()
    d += np.linalg.norm(ordered[0] - ordered[-1])
    return d

# --- MÉTODOS DE INFERENCIA ---

def run_greedy(model, points):
    # Greedy: Argmax determinista
    model.eval()
    # points ya viene como float32, lo movemos a device
    pts = points.unsqueeze(0).to(DEVICE)
    B, N, _ = pts.size()
    
    with torch.no_grad():
        enc_out = model.encoder(pts)
        graph_emb = enc_out.mean(dim=1)
        
        curr = torch.zeros(B, dtype=torch.long).to(DEVICE)
        mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
        mask[0, curr] = True
        
        tour = [curr]
        last_emb = enc_out[0, curr, :]
        first_emb = last_emb
        
        for _ in range(N-1):
            probs, _ = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
            next_idx = torch.argmax(probs, dim=1)
            tour.append(next_idx)
            mask[0, next_idx] = True
            last_emb = enc_out[0, next_idx, :]
            
    return torch.stack(tour, dim=1).squeeze().cpu().numpy()

def run_beam(model, points, width=3):
    model.eval()
    pts = points.unsqueeze(0).to(DEVICE)
    B, N, _ = pts.size()
    
    with torch.no_grad():
        enc_out = model.encoder(pts)
        graph_emb = enc_out.mean(dim=1)
        
        start_node = 0
        mask = torch.zeros(1, N, dtype=torch.bool).to(DEVICE)
        mask[0, start_node] = True
        beams = [(0.0, [start_node], mask)]
        
        for _ in range(N-1):
            candidates = []
            for score, tour, msk in beams:
                curr = tour[-1]
                # Unsqueeze necesarios para mantener dimensiones [1, Emb]
                last_emb = enc_out[0, curr, :].unsqueeze(0)
                first_emb = enc_out[0, tour[0], :].unsqueeze(0)
                
                probs, _ = model.decoder(enc_out, graph_emb, last_emb, first_emb, msk)
                log_probs = torch.log(probs + 1e-9).squeeze()
                
                # Topk retorna valores y indices
                top_v, top_i = torch.topk(log_probs, k=width)
                
                for v, i in zip(top_v, top_i):
                    idx = i.item()
                    if msk[0, idx]: continue
                    
                    new_mask = msk.clone()
                    new_mask[0, idx] = True
                    candidates.append((score + v.item(), tour + [idx], new_mask))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:width]
            
        best_d = float('inf')
        best_t = None
        pts_np = points.cpu().numpy()
        
        for _, tour, _ in beams:
            d = calc_dist(pts_np, np.array(tour))
            if d < best_d:
                best_d = d
                best_t = tour
                
        return np.array(best_t)

def run_sampling(model, points, samples=500):
    model.eval()
    # Replicar mapa N veces
    pts = points.unsqueeze(0).repeat(samples, 1, 1).to(DEVICE)
    B, N, _ = pts.size()
    
    with torch.no_grad():
        enc_out = model.encoder(pts)
        graph_emb = enc_out.mean(dim=1)
        
        curr = torch.zeros(B, dtype=torch.long).to(DEVICE)
        mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
        mask[torch.arange(B), curr] = True
        tour = [curr]
        
        last_emb = enc_out[torch.arange(B), curr, :]
        first_emb = last_emb
        
        for _ in range(N-1):
            probs, _ = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
            dist = torch.distributions.Categorical(probs)
            next_idx = dist.sample()
            
            tour.append(next_idx)
            mask[torch.arange(B), next_idx] = True
            last_emb = enc_out[torch.arange(B), next_idx, :]
            
        tour_tensor = torch.stack(tour, dim=1) 
        
        # Calcular distancias
        idx = tour_tensor.unsqueeze(2).expand(-1, -1, 2)
        ordered = torch.gather(pts, 1, idx)
        diff = ordered[:, 1:] - ordered[:, :-1]
        lens = torch.norm(diff, dim=2).sum(dim=1)
        lens += torch.norm(ordered[:, 0] - ordered[:, -1], dim=1)
        
        best_val, best_idx = torch.min(lens, dim=0)
        return tour_tensor[best_idx].cpu().numpy()

# --- MAIN BENCHMARK ---
def run_final_benchmark():
    print(f"\n🏆 INICIANDO TRIATLÓN DE VALIDACIÓN (Greedy vs Beam vs Sampling)")
    print(f"   Modelo: {os.path.basename(MODEL_PATH)}")
    print(f"   Data: {VALIDATION_PATH}")
    print("-" * 70)
    
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("   ✅ Modelo cargado.")
    except Exception as e:
        print(f"   ❌ Error modelo: {e}")
        return

    LEVELS = ['Easy', 'Medium', 'Hard'] 
    
    final_report = {}

    for level in LEVELS:
        print(f"\n📊 Procesando Nivel: {level}")
        
        X, Y_gt = load_val_data(level, limit=MAX_SAMPLES)
        
        if X is None:
            print(f"   ⚠️ Saltando {level} (sin datos).")
            continue
            
        n_samples = len(X)
        print(f"   ✅ Evaluando sobre {n_samples} mapas reales.")
        
        gap_greedy = []
        gap_beam = []
        gap_sampling = []
        
        for i in tqdm(range(n_samples), desc=f"   Corriendo tests"):
            points_np = X[i] 
            gt_tour = Y_gt[i]
            
            # --- CORRECCIÓN CRÍTICA AQUÍ ---
            # Aseguramos que el tensor sea Float32 explícitamente
            points_tensor = torch.tensor(points_np, dtype=torch.float32)
            
            dist_gt = calc_dist(points_np, gt_tour)
            
            # A. GREEDY
            tour_g = run_greedy(model, points_tensor)
            dist_g = calc_dist(points_np, tour_g)
            gap_greedy.append( (dist_g - dist_gt) / dist_gt * 100 )
            
            # B. BEAM SEARCH
            tour_b = run_beam(model, points_tensor, width=BEAM_WIDTH)
            dist_b = calc_dist(points_np, tour_b)
            gap_beam.append( (dist_b - dist_gt) / dist_gt * 100 )
            
            # C. SAMPLING
            tour_s = run_sampling(model, points_tensor, samples=SAMPLING_SAMPLES)
            dist_s = calc_dist(points_np, tour_s)
            gap_sampling.append( (dist_s - dist_gt) / dist_gt * 100 )
            
        final_report[level] = {
            'Greedy': np.mean(gap_greedy),
            'Beam': np.mean(gap_beam),
            'Sampling': np.mean(gap_sampling)
        }

    print("\n" + "="*80)
    print(f"{'NIVEL':<10} | {'GREEDY GAP':<15} | {'BEAM (W=3) GAP':<18} | {'SAMPLING (500) GAP':<20}")
    print("="*80)
    
    for level in LEVELS:
        if level in final_report:
            res = final_report[level]
            print(f"{level:<10} | {res['Greedy']:6.2f}%         | {res['Beam']:6.2f}%           | {res['Sampling']:6.2f}%")
        else:
            print(f"{level:<10} | ---               | ---                  | ---")
            
    print("-" * 80)
    print("Nota: El Gap es relativo a la solución óptima (GLS) guardada en los archivos .npz.")
    print("      Un Gap negativo significa que superaste al dato de validación original.")
    print("="*80)

if __name__ == "__main__":
    run_final_benchmark()