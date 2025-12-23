import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import glob
import random
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Models.model import TSPModel
from Models.dataset import TSPDataset

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(current_dir, 'Saved_Models', 'tsp_transformer_Ultimate.pth')

# Entrenamiento largo y paciente
EPOCHS = 50           
BATCH_SIZE = 32       
LEARNING_RATE = 1e-4  

# --- DATA AUGMENTATION (El remedio contra el Overfitting) ---
def random_augment_batch(x):
    """Rota aleatoriamente el batch para que el modelo no memorice."""
    idx = np.random.randint(0, 8)
    x_c, y_c = x[:, :, 0], x[:, :, 1]
    if idx == 0: return x
    elif idx == 1: return torch.stack((1-y_c, x_c), dim=2)
    elif idx == 2: return torch.stack((1-x_c, 1-y_c), dim=2)
    elif idx == 3: return torch.stack((y_c, 1-x_c), dim=2)
    elif idx == 4: return torch.stack((x_c, 1-y_c), dim=2)
    elif idx == 5: return torch.stack((1-y_c, 1-x_c), dim=2)
    elif idx == 6: return torch.stack((1-x_c, y_c), dim=2)
    elif idx == 7: return torch.stack((y_c, x_c), dim=2)
    return x

# --- CARGAR DATOS ---
def load_data(difficulty):
    folder_path = os.path.join(current_dir, 'Data', difficulty)
    pattern = os.path.join(folder_path, "*.npz")
    files = glob.glob(pattern)
    x_list, y_list = [], []
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                if 'points' in data and 'solutions' in data:
                    pts = data['points']
                    sols = data['solutions']
                    if pts.dtype == object: pts = pts.astype(np.float32)
                    if sols.dtype == object: sols = sols.astype(np.int64)
                    x_list.append(pts)
                    y_list.append(sols)
        except Exception: pass
    if not x_list: return None, None
    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

def train_ultimate():
    print("=== INICIANDO ENTRENAMIENTO ULTIMATE (MIXED + AUGMENTED) ===")
    
    # 1. Cargar Todo
    loaders = []
    difficulties = ['Easy', 'Medium', 'Hard']
    for diff in difficulties:
        X, y = load_data(diff)
        if X is not None:
            ds = TSPDataset(X, y)
            # drop_last=True es vital para mezclar tamaños distintos
            ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            loaders.append(ld)
            print(f"   -> {diff}: {len(X)} muestras.")

    if not loaders: return

    # 2. Modelo Heavy (6 Capas) - Empezamos fresco o cargamos el Heavy base
    # Recomiendo empezar fresco para borrar vicios del overfitting anterior
    model = TSPModel(embedding_dim=128, hidden_dim=512, n_layers=6, n_heads=8).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Recolectar y mezclar batches
        all_batches = []
        for loader in loaders:
            for batch in loader:
                all_batches.append(batch)
        random.shuffle(all_batches) # <--- AQUÍ ESTÁ LA CLAVE DEL MIXTO
        
        pbar = tqdm(all_batches, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in pbar:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # --- APLICAR AUGMENTATION ---
            inputs = random_augment_batch(inputs)
            # ----------------------------
            
            optimizer.zero_grad()
            
            enc_out = model.encoder(inputs)
            graph_emb = enc_out.mean(dim=1)
            
            B, N, _ = inputs.size()
            mask = torch.zeros(B, N, dtype=torch.bool).to(DEVICE)
            loss = 0
            
            curr_idx = targets[:, 0]
            mask[torch.arange(B), curr_idx] = True
            last_emb = enc_out[torch.arange(B), curr_idx, :]
            first_emb = last_emb
            
            for t in range(N - 1):
                probs, logits = model.decoder(enc_out, graph_emb, last_emb, first_emb, mask)
                true_next = targets[:, t+1]
                
                loss += criterion(logits, true_next)
                
                mask = mask.clone()
                mask[torch.arange(B), true_next] = True
                last_emb = enc_out[torch.arange(B), true_next, :]
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += (loss.item() / N)
            pbar.set_postfix(loss=loss.item()/N)
            
        print(f"Fin Epoch {epoch+1} | Loss Global: {epoch_loss/len(all_batches):.4f}")
        
        if (epoch+1) % 5 == 0:
             torch.save(model.state_dict(), SAVE_PATH)

    print(f"Modelo Ultimate guardado en: {SAVE_PATH}")

if __name__ == "__main__":
    train_ultimate()