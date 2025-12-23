import torch
import torch.nn as nn

class TSPEncoder(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=128, hidden_dim=512, n_layers=3, n_heads=8):
        """
        Encoder basado en 'Attention, Learn to Solve Routing Problems!' (Kool et al. 2019)
        input_dim: 2 para coordenadas X,Y
        embedding_dim: Dimensión de los vectores latentes (dh) - default 128 
        hidden_dim: Dimensión de la capa oculta en FeedForward - default 512 [cite: 500]
        n_layers: Número de capas de atención (N) - default 3 [cite: 601]
        """
        super(TSPEncoder, self).__init__()
        
        # 1. Proyección Lineal Inicial (Sin Positional Encoding)
        # Transforma (x,y) -> embedding_dim
        self.init_embed = nn.Linear(input_dim, embedding_dim)
        
        # 2. Bloques de Atención
        # Usamos nn.ModuleList para apilar las capas
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, n_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        """
        x: [batch_size, num_nodes, input_dim]
        """
        # Proyección inicial
        h = self.init_embed(x) # [batch, nodes, embed_dim]
        
        # Pasar por las capas de atención
        for layer in self.layers:
            h = layer(h)
            
        return h

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, hidden_dim):
        super(EncoderLayer, self).__init__()
        
        # Multi-Head Attention (MHA)
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)
        
        # Batch Normalization 1 (Preferencia del paper sobre LayerNorm) [cite: 495]
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, x):
        # x shape: [batch, nodes, embed_dim]
        
        # --- Subcapa 1: MHA ---
        # Skip connection + BN
        # Nota: PyTorch MHA retorna (output, weights), solo queremos output
        h_attn, _ = self.mha(x, x, x) 
        
        # Para usar BatchNorm1d en secuencia [batch, nodes, dim], debemos trasponer
        # a [batch, dim, nodes] y luego volver.
        h = x + h_attn
        h = self.bn1(h.transpose(1, 2)).transpose(1, 2)
        
        # --- Subcapa 2: Feed Forward ---
        # Skip connection + BN
        h_ff = self.ff(h)
        h = h + h_ff
        h = self.bn2(h.transpose(1, 2)).transpose(1, 2)
        
        return h