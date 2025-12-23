import torch
import torch.nn as nn
from .encoder import TSPEncoder
from .decoder import TSPDecoder

class TSPModel(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=128, hidden_dim=512, n_layers=3, n_heads=8):
        super(TSPModel, self).__init__()
        
        self.encoder = TSPEncoder(input_dim, embedding_dim, hidden_dim, n_layers, n_heads)
        self.decoder = TSPDecoder(embedding_dim, n_heads)
        
    def forward(self, inputs, target_tour=None):
        """
        inputs: [batch_size, num_nodes, 2]
        target_tour: [batch_size, num_nodes] (Indices del tour optimo para teacher forcing)
        """
        batch_size, num_nodes, _ = inputs.size()
        
        # 1. ENCODER
        encoder_output = self.encoder(inputs)
        
        # 2. Embedding global del grafo (promedio)
        graph_embedding = encoder_output.mean(dim=1)
        
        # Estructuras para el Decoder
        outputs = []
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool).to(inputs.device)
        last_node_embedding = None
        
        # Preparar el primer nodo
        first_node_idx = target_tour[:, 0]
        first_node_embedding = encoder_output[torch.arange(batch_size), first_node_idx, :]
        
        # 3. DECODER (Bucle paso a paso)
        for t in range(num_nodes):
            probs, logits = self.decoder(
                encoder_output=encoder_output,
                graph_embedding=graph_embedding,
                last_node_embedding=last_node_embedding,
                first_node_embedding=first_node_embedding,
                mask=mask
            )
            
            outputs.append(logits)
            
            # Teacher Forcing
            true_next_node_idx = target_tour[:, t]
            
            # --- CORRECCIÓN AQUÍ ---
            # Clonamos la máscara antes de modificarla. 
            # Esto crea un nuevo tensor y preserva el historial para el gradiente.
            mask = mask.clone() 
            mask[torch.arange(batch_size), true_next_node_idx] = True
            
            # Actualizamos el embedding del nodo anterior
            last_node_embedding = encoder_output[torch.arange(batch_size), true_next_node_idx, :]
            
        # Retornamos los logits apilados
        return torch.stack(outputs, dim=1)