import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import numpy as np
import glob
import random
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Models.model import TSPModel

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(current_dir, 'Saved_Models', 'tsp_transformer_UltimateNoLocalSearch.pth')

# Configuración del Entrenamiento
EPOCHS = 50           
BATCH_SIZE = 64       # Subí un poco el batch size ya que los datos cargan rápido
LEARNING_RATE = 1e-4  

# --- CLASE DATASET (Simple, en memoria) ---
class TSPDataset(Dataset):
    def __init__(self, data, solutions):
        self.data = torch.tensor(data, dtype=torch.float32)
        # Aseguramos que las soluciones sean Long (enteros)
        self.solutions = torch.tensor(solutions, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.solutions[idx]

# --- CARGA EFICIENTE (PACKED) ---
def load_data_from_folder(folder_path, max_samples=None):
    """Carga los archivos packed_*.npz concatenados."""
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
    if not file_list: return None, None

    all_points, all_solutions = [], []
    total_loaded = 0
    print(f"📂 Cargando desde: {folder_path}...")

    for f in file_list:
        try:
            data = np.load(f)
            pts, sols = data['points'], data['solutions']
            
            if max_samples and total_loaded >= max_samples: break
            
            all_points.append(pts)
            all_solutions.append(sols)
            total_loaded += len(pts)
            
            # Recorte final si nos pasamos
            if max_samples and total_loaded > max_samples:
                excess = total_loaded - max_samples
                all_points[-1] = all_points[-1][:-excess]
                all_solutions[-1] = all_solutions[-1][:-excess]
                break
        except: pass

    if not all_points: return None, None
    return np.concatenate(all_points, axis=0), np.concatenate(all_solutions, axis=0)

# --- DATA AUGMENTATION (EL SECRETO) ---
def random_augment_batch(x):
    """8 Simetrías del cuadrado para multiplicar data x8"""
    idx = random.randint(0, 7)
    x_c, y_c = x[:, :, 0], x[:, :, 1]
    
    if idx == 0: return x # Original
    if idx == 1: return torch.stack((1-x_c, y_c), dim=2)   # Flip X
    if idx == 2: return torch.stack((x_c, 1-y_c), dim=2)   # Flip Y
    if idx == 3: return torch.stack((1-x_c, 1-y_c), dim=2) # Flip XY
    if idx == 4: return torch.stack((y_c, x_c), dim=2)     # Transpose
    if idx == 5: return torch.stack((1-y_c, 1-x_c), dim=2) # ...
    if idx == 6: return torch.stack((1-y_c, x_c), dim=2)
    if idx == 7: return torch.stack((y_c, 1-x_c), dim=2)
    return x

def train_ultimate():
    print("=== 🚀 INICIANDO ENTRENAMIENTO ULTIMATE (PACKED + AUGMENTED) ===")
    
    # 1. Preparar Loaders por dificultad
    loaders = []
    
    # AJUSTA TUS RUTAS Y CANTIDADES AQUÍ
    configs = [
        ("Data/Packed_NoLS/Easy", 10000),    # 10k Easy
        ("Data/Packed_NoLS/Medium", 10000),  # 10k Medium
        ("Data/Packed_NoLS/Hard", 5000)      # 5k Hard (Experimento A/B)
    ]
    
    for path, limit in configs:
        X, Y = load_data_from_folder(os.path.join(current_dir, path), max_samples=limit)
        if X is not None:
            ds = TSPDataset(X, Y)
            # Batch Size independiente para que no mezcle tamaños en un tensor
            loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))
            print(f"   ✅ {path}: {len(X)} muestras cargadas.")

    # 2. Modelo
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    
    # Si quieres continuar un entrenamiento previo, descomenta:
    # try:
    #     model.load_state_dict(torch.load(SAVE_PATH))
    #     print("   🔄 Modelo previo cargado.")
    # except: print("   ✨ Iniciando desde cero.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. Bucle de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        
        # A. Recolectar todos los batches de todos los niveles
        all_batches = []
        for loader in loaders:
            for batch in loader:
                all_batches.append(batch)
        
        # B. Mezclar el orden (Easy -> Hard -> Medium -> Hard...)
        random.shuffle(all_batches)
        
        epoch_loss = 0
        pbar = tqdm(all_batches, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # C. APLICAR AUGMENTATION (Magia aquí)
            inputs = random_augment_batch(inputs)
            
            optimizer.zero_grad()
            
            # Forward
            enc_out = model.encoder(inputs)
            graph_emb = enc_out.mean(dim=1)
            
            B, N, _ = inputs.size()
            mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
            loss = 0
            
            # Teacher Forcing
            curr_idx = targets[:, 0]
            mask[torch.arange(B), curr_idx] = True
            last_emb = enc_out[torch.arange(B), curr_idx, :]
            first_emb = last_emb
            
            for t in range(N - 1):
                probs, logits = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
                true_next = targets[:, t+1]
                
                loss += criterion(logits, true_next)
                
                # Update estado
                mask = mask.clone()
                mask[torch.arange(B), true_next] = True
                last_emb = enc_out[torch.arange(B), true_next, :]
            
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item() / N
            epoch_loss += loss_val
            pbar.set_postfix(loss=loss_val)
        
        avg_loss = epoch_loss / len(all_batches)
        print(f"   📉 Fin Epoch {epoch+1} | Loss Promedio: {avg_loss:.4f}")
        
        # Guardar checkpoint cada 5 épocas o si es muy buena
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), SAVE_PATH)

    # Guardado final
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"🏁 Entrenamiento finalizado. Modelo guardado en {SAVE_PATH}")

if __name__ == "__main__":
    train_ultimate()