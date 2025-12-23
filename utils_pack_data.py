import numpy as np
import glob
import os
from tqdm import tqdm

# --- CONFIGURACIÓN ---
SOURCE_ROOT = "Data/Massive_NoLS"  # Donde están los miles de archivos sueltos
DEST_ROOT = "Data/Packed_NoLS"     # Donde quedarán los archivos grandes

# CONFIGURACIÓN ESPECÍFICA POR NIVEL
PACKING_CONFIG = {
    'Hard': 5000     # Paquetes de 5k (Más seguro para RAM)
}

def pack_folder(level_name):
    samples_per_file = PACKING_CONFIG.get(level_name, 10000) # Default 10k si no encuentra el nivel
    
    src_dir = os.path.join(SOURCE_ROOT, level_name)
    dest_dir = os.path.join(DEST_ROOT, level_name)
    
    # Verificamos si existe la carpeta fuente
    if not os.path.exists(src_dir):
        print(f"⚠️ Saltando {level_name}: No existe la carpeta {src_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Buscar todos los archivos pequeños (.npz)
    # Ordenamos para asegurar consistencia
    files = sorted(glob.glob(os.path.join(src_dir, "*.npz")))
    
    print(f"\n📦 Empaquetando {level_name} (Total: {len(files)} archivos)")
    print(f"   ↳ Tamaño de paquete: {samples_per_file} muestras por archivo")
    
    if not files:
        print("   ⚠️ Carpeta vacía.")
        return

    # 2. Procesar en lotes
    total_batches = len(files) // samples_per_file + (1 if len(files) % samples_per_file != 0 else 0)
    
    for i in range(total_batches):
        start_idx = i * samples_per_file
        end_idx = min((i + 1) * samples_per_file, len(files))
        
        batch_files = files[start_idx:end_idx]
        
        # Si el último lote está vacío (caso borde), saltar
        if not batch_files: continue

        all_points = []
        all_solutions = []
        
        # Leer archivos pequeños con barra de progreso
        for f in tqdm(batch_files, desc=f"   Lote {i+1}/{total_batches}", leave=False):
            try:
                data = np.load(f)
                all_points.append(data['points'])
                all_solutions.append(data['solutions'])
            except Exception as e:
                print(f"❌ Error leyendo {f}: {e}")
        
        # Convertir a arrays gigantes
        # Points Shape: [Batch, N_Nodes, 2]
        packed_points = np.array(all_points)
        # Solutions Shape: [Batch, N_Nodes]
        packed_solutions = np.array(all_solutions)
        
        # Guardar archivo grande
        save_path = os.path.join(dest_dir, f"packed_{level_name}_{i}.npz")
        np.savez_compressed(save_path, points=packed_points, solutions=packed_solutions)
        
    print(f"✅ ¡{level_name} completado! Creados {total_batches} archivos grandes en {dest_dir}")

def main():
    levels = ['Hard']
    print("=== INICIANDO EMPAQUETADO DE DATOS ===")
    for level in levels:
        pack_folder(level)
    print("\n=== PROCESO FINALIZADO ===")

if __name__ == "__main__":
    main()