import torch
import time
import numpy as np
import sys
import os

# --- 1. ARREGLO DE IMPORTS ---
sys.path.append(os.getcwd())

try:
    from Models.model import TSPModel
except ImportError:
    try:
        from model import TSPModel
    except ImportError:
        print("❌ Error crítico: No se encuentra la clase TSPModel.")
        sys.exit(1)

# --- CONFIGURACIÓN ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join('Saved_Models', 'tsp_transformer_Ultimate.pth') 

BATCH_SIZE_TEST = 500   # Cantidad de mapas para promediar
BEAM_WIDTH = 5         # Ancho del Beam Search

# --- 2. FUNCIÓN GREEDY / SAMPLING MANUAL (Para evitar error de target_tour) ---
def run_inference_loop(model, inputs, strategy='greedy'):
    """
    Realiza la inferencia paso a paso manualmente.
    strategy: 'greedy' (argmax) o 'sampling' (multinomial - aunque para tiempo argmax sirve igual)
    """
    batch_size, num_nodes, _ = inputs.size()
    
    # 1. Encoder
    encoder_output = model.encoder(inputs)
    graph_embedding = encoder_output.mean(dim=1)
    
    # Estructuras de Estado
    mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool).to(DEVICE)
    last_node_embedding = None
    first_node_embedding = None
    
    # Variable para guardar el nodo actual (empezamos asumiendo nodo 0 o placeholder)
    # En inferencia pura, el modelo suele elegir el primero libre o el 0.
    # Para ser consistentes con tu arquitectura, elegiremos el nodo 0 como arranque.
    current_idx = torch.zeros(batch_size, dtype=torch.long).to(DEVICE)
    
    # Guardamos primer nodo para contexto
    first_node_embedding = encoder_output[torch.arange(batch_size), current_idx, :]
    
    # Marcamos el inicial como visitado
    # Ojo: Tu decoder maneja el paso t=0 internamente con placeholders si last_node es None.
    # Vamos a seguir la lógica de tu decoder: paso t=0
    
    last_node_embedding = None # Para t=0
    
    for t in range(num_nodes):
        # Decoder Step
        probs, _ = model.decoder(
            encoder_output=encoder_output,
            graph_embedding=graph_embedding,
            last_node_embedding=last_node_embedding,
            first_node_embedding=first_node_embedding,
            mask=mask
        )
        
        # Selección del siguiente nodo
        if strategy == 'greedy':
            next_node = probs.argmax(dim=1)
        else:
            # Para benchmark de tiempo de sampling, el coste computacional es casi identico
            # Usamos argmax por estabilidad en el test de velocidad
            next_node = probs.argmax(dim=1) 
        
        # Actualizar First Node si estamos en el paso 0 (ya que t=0 define el inicio)
        if t == 0:
            first_node_embedding = encoder_output[torch.arange(batch_size), next_node, :]
            
        # Actualizar Last Node para el siguiente paso
        last_node_embedding = encoder_output[torch.arange(batch_size), next_node, :]
        
        # Actualizar Máscara
        mask[torch.arange(batch_size), next_node] = True
        
    return

# --- 3. FUNCIÓN BEAM SEARCH ---
def run_beam_search(model, inputs, beam_width=3):
    batch_size, num_nodes, _ = inputs.size()
    encoder_output = model.encoder(inputs)
    graph_embedding = encoder_output.mean(dim=1)
    
    sequences = torch.zeros(batch_size, beam_width, 0, dtype=torch.long).to(DEVICE)
    log_probs = torch.zeros(batch_size, beam_width).to(DEVICE)
    log_probs[:, 1:] = float('-inf') 
    
    mask = torch.zeros(batch_size, beam_width, num_nodes, dtype=torch.bool).to(DEVICE)
    last_node_embedding = None
    first_node_embedding = None

    encoder_output = encoder_output.repeat_interleave(beam_width, dim=0)
    graph_embedding = graph_embedding.repeat_interleave(beam_width, dim=0)

    for t in range(num_nodes):
        current_batch_beam = batch_size * beam_width
        
        if t == 0:
            last_node_embedding = None
            first_node_embedding = None
        else:
            last_token = sequences[:, :, -1].view(-1)
            last_node_embedding = encoder_output[torch.arange(current_batch_beam), last_token, :]
            first_token = sequences[:, :, 0].view(-1)
            first_node_embedding = encoder_output[torch.arange(current_batch_beam), first_token, :]

        mask_flat = mask.view(current_batch_beam, num_nodes)

        probs, _ = model.decoder(
            encoder_output=encoder_output,
            graph_embedding=graph_embedding,
            last_node_embedding=last_node_embedding,
            first_node_embedding=first_node_embedding,
            mask=mask_flat
        )
        
        curr_log_probs = torch.log(probs + 1e-10) 
        prev_log_probs = log_probs.view(batch_size, beam_width, 1)
        candidate_log_probs = prev_log_probs + curr_log_probs.view(batch_size, beam_width, num_nodes)
        candidate_log_probs_flat = candidate_log_probs.view(batch_size, -1)
        
        topk_log_probs, topk_indices = candidate_log_probs_flat.topk(beam_width, dim=1)
        
        beam_parents = topk_indices // num_nodes
        real_next_nodes = topk_indices % num_nodes
        
        new_sequences = torch.zeros(batch_size, beam_width, t + 1, dtype=torch.long).to(DEVICE)
        for b in range(batch_size):
            new_sequences[b] = torch.cat([
                sequences[b][beam_parents[b]],
                real_next_nodes[b].unsqueeze(1)
            ], dim=1)
            
        sequences = new_sequences
        log_probs = topk_log_probs
        
        new_mask = torch.zeros(batch_size, beam_width, num_nodes, dtype=torch.bool).to(DEVICE)
        for b in range(batch_size):
            new_mask[b] = mask[b][beam_parents[b]]
            new_mask[b, torch.arange(beam_width), real_next_nodes[b]] = True
        mask = new_mask

    return sequences


# --- 4. BENCHMARK PRINCIPAL ---
def measure_inference_time():
    print(f"🔄 Cargando modelo desde: {MODEL_PATH}")
    
    # Ajusta n_layers=6 si ese es tu modelo final
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ Modelo cargado correctamente.")
    else:
        print(f"❌ ERROR: No se encuentra el archivo {MODEL_PATH}")
        return

    model.eval()
    
    problem_sizes = [20, 50, 100]
    
    print("\n⚡ RESULTADOS DE LATENCIA (Promedio por Instancia) ⚡")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("-" * 80)
    print(f"{'Nivel':<10} | {'Greedy (ms)':<15} | {'Sampling-50 (ms)':<18} | {'Beam-5 (ms)':<15}")
    print("-" * 80)

    for N in problem_sizes:
        inputs = torch.rand(BATCH_SIZE_TEST, N, 2).to(DEVICE)
        
        # --- 1. WARM-UP (Usando run_inference_loop) ---
        with torch.no_grad():
            run_inference_loop(model, inputs[0:2])
            if torch.cuda.is_available(): torch.cuda.synchronize()

        # --- 2. GREEDY ---
        start = time.perf_counter()
        with torch.no_grad():
            run_inference_loop(model, inputs, strategy='greedy')
            if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_greedy = (end - start) / BATCH_SIZE_TEST * 1000 

        # --- 3. SAMPLING (50) ---
        SAMPLES = 50
        inputs_expanded = inputs.repeat_interleave(SAMPLES, dim=0)
        
        # Ajuste de batch para TSP-100 si falta memoria
        real_batch_sampling = max(1, BATCH_SIZE_TEST // 5) if N == 100 else BATCH_SIZE_TEST
        inputs_subset = inputs_expanded[:real_batch_sampling * SAMPLES]

        start = time.perf_counter()
        with torch.no_grad():
            run_inference_loop(model, inputs_subset, strategy='greedy') # Greedy sobre inputs expandidos = Sampling base
            if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_sampling = (end - start) / real_batch_sampling * 1000

        # --- 4. BEAM SEARCH (5) ---
        start = time.perf_counter()
        with torch.no_grad():
            run_beam_search(model, inputs, beam_width=BEAM_WIDTH)
            if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_beam = (end - start) / BATCH_SIZE_TEST * 1000

        print(f"TSP-{N:<6} | {time_greedy:.4f} ms      | {time_sampling:.4f} ms         | {time_beam:.4f} ms")

if __name__ == "__main__":
    measure_inference_time()