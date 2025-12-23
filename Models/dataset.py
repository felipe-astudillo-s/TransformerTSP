import torch
import numpy as np  # <--- Agregamos esta importación necesaria
from torch.utils.data import Dataset

class TSPDataset(Dataset):
    def __init__(self, data_samples, solutions=None):
        """
        data_samples: Lista o Tensor de [N, 2] coordenadas
        solutions: Lista o Tensor de [N] índices (orden óptimo)
        """
        self.data = data_samples
        self.solutions = solutions
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 1. Procesar INPUTS (Coordenadas)
        # Extraemos el elemento
        x_sample = self.data[idx]
        # Si por alguna razón viene como objeto, lo forzamos a float32
        if isinstance(x_sample, np.ndarray) and x_sample.dtype == object:
            x_sample = x_sample.astype(np.float32)
        
        x = torch.tensor(x_sample, dtype=torch.float32)

        # 2. Procesar TARGETS (Soluciones)
        if self.solutions is not None:
            y_sample = self.solutions[idx]
            
            # --- CORRECCIÓN CRÍTICA ---
            # Si el array es de tipo 'object' (pasa a menudo con .npz viejos),
            # lo convertimos explícitamente a int64 (enteros)
            if isinstance(y_sample, np.ndarray) and y_sample.dtype == object:
                y_sample = y_sample.astype(np.int64)
            # --------------------------

            y = torch.tensor(y_sample, dtype=torch.long)
            return x, y
            
        return x