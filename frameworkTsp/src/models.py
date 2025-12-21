import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TSPTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_layers=2, dropout_rate=0.1):
        super(TSPTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.step_embedding = nn.Embedding(101, embed_dim) #para saber que paso vamos, positional encoding en el encoder. 


        
        # --- ENCODER ---
        # Proyección inicial de coordenadas (x, y)
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder (Sin Positional Encoding) 
        # agregue norm first + mas capas
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #normalizar el encoder
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # --- DECODER ---
        # Fusión contexto (grafo + media de visitadas + primera + última)
        self.ctx_fusion = nn.Linear(4 * embed_dim, embed_dim)

        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        #feed forward en el decoder glimpse
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # Pointer Scorer
        self.pointer = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.pointer.weight, gain=0.1)

    def forward(self, x_src, visited):
        """
        x_src: (batch, num_cities, 2) -> Coordenadas
        visited: (batch, num_cities) -> Índice ciudades visitadas (-1 para padding)
        """
        B, num_cities, _ = x_src.shape

        # 1. --- ENCODER ---
        enc_input = self.encoder_input_layer(x_src)
        memory = self.encoder(enc_input) 
        memory = self.encoder_norm(memory) # Normalización final

        # 2. --- PREPARACIÓN DE CONTEXTO ---
        # Máscaras y Step
        current_step = (visited != -1).sum(dim=1)
        step_embed = self.step_embedding(current_step) # (B, D)

        visited_mask_pos = visited != -1
        visited_city_mask = torch.zeros(B, num_cities, dtype=torch.bool, device=visited.device)
        batch_ids, pos_ids = visited_mask_pos.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids].long()] = True

        # Cálculos de contexto
        mask = visited_city_mask.unsqueeze(-1) # (B, N, 1)
        
        # A. Global Graph (Promedio fijo)
        graph_embedding = memory.mean(dim=1) 
        
        # B. Visited Mean (Promedio dinámico) - OJO: Manejar división por cero al inicio
        sum_ctx = (memory * mask).sum(dim=1)
        count_ctx = mask.sum(dim=1).clamp(min=1) # Evitar NaN en el paso 0
        context_mean = sum_ctx / count_ctx 

        # C. Start & Last Node
        # Truco: Si step=0, last_node es start_node (o un placeholder)
        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = (visited_mask_pos.sum(dim=1) - 1).clamp(min=0) # Evitar -1

        batch_idx = torch.arange(B, device=x_src.device)
        start_city_embed = memory[batch_idx, visited[batch_idx, start_idx].long()]
        last_city_embed = memory[batch_idx, visited[batch_idx, last_idx].long()]

        # FUSIÓN (Ahora dimension 4*D es correcta)
        ctx_concat = torch.cat([graph_embedding, context_mean, last_city_embed, start_city_embed], dim=-1)
        
        # Query base + Positional Encoding
        decoder_query = self.ctx_fusion(ctx_concat) + step_embed

        # 3. --- DECODER ATTENTION (GLIMPSE) ---
        query = decoder_query.unsqueeze(1) # (B, 1, D)

        # A. Atención
        attn_out, _ = self.cross_attn(query=query, key=memory, value=memory)
        
        # B. Residual + Norm (Estilo Transformer)
        x = decoder_query.unsqueeze(1) + attn_out
        x = self.decoder_norm(x)
        
        # C. Feed Forward + Residual + Norm (Potencia el razonamiento)
        x2 = self.ff_layer(x)
        output_glimpse = self.final_norm(x + x2).squeeze(1) # (B, D)

        # 4. --- POINTER MECHANISM ---
        # Query final para el puntero
        ptr_query = self.pointer(output_glimpse) 

        # Cálculo de Scores (Compatibilidad)
        scores = torch.matmul(
            ptr_query.unsqueeze(1), 
            memory.transpose(1, 2)
        ).squeeze(1)

        # --- MEJORA 4: Tanh Clipping (Bello et al. / Kool et al.) ---
        # Controla la explosión de gradientes. El factor 10 es estándar.
        #scores = 10 * torch.tanh(scores)

        # Masking
        scores = scores.masked_fill(visited_city_mask, float("-inf"))
        
        # Probabilidades
        probs = F.softmax(scores, dim=-1)

        return probs
    
    def encode(self, x_src):
        """
        x_src: (batch, num_cities, 2) -> Coordenadas
        """
        # Proyectamos las coordenadas y pasamos por el encoder
        enc_input = self.encoder_input_layer(x_src)
        memory = self.encoder(enc_input)  # (batch, n_cities, embed_dim)

        return memory
    
    def decode(self, memory, visited):
        """
        memory:  (batch, num_cities, embed_dim) -> Salida del encoder
        visited: (batch, T) -> Índice ciudades visitadas (-1 para padding)
        """
        B, num_cities, _ = memory.shape
        device = memory.device

        # 2. --- MÁSCARA CIUDADES VISITADAS ---
        # Posiciones válidas del tour
        visited_mask_pos = visited != -1          # (B, T)

        # Máscara por ciudad
        visited_city_mask = torch.zeros(
            B, num_cities, dtype=torch.bool, device=device
        )

        valid = visited != -1
        batch_ids, pos_ids = valid.nonzero(as_tuple=True)

        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        # 3. --- DECODER: Media de ciudades visitadas ---
        mask = visited_city_mask.unsqueeze(-1)    # (B, N, 1)

        sum_ctx = (memory * mask).sum(dim=1)
        count_ctx = mask.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx         # (B, D)

        # 4. --- DECODER: Obtener primera y última ciudad y concatenar a la media ---
        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = visited_mask_pos.sum(dim=1) - 1

        batch_idx = torch.arange(B, device=device)
        start_city_embed = memory[
            batch_idx, visited[batch_idx, start_idx].long()
        ]  # (B, D)
        last_city_embed = memory[
            batch_idx, visited[batch_idx, last_idx].long()
        ]   # (B, D)

        # Concatenación y proyección
        ctx_concat = torch.cat(
            [context_mean, last_city_embed, start_city_embed], dim=-1
        )  # (B, 3D)
        decoder_query = self.ctx_fusion(ctx_concat)  # (B, D)

        # 5. --- DECODER: Cross-Attention (Glimpse) ---
        query = decoder_query.unsqueeze(1)  # (B, 1, D)

        for _ in range(self.num_glimpses):
            attn_out, _ = self.cross_attn(
                query=query,            # (B, 1, D)
                key=memory,             # (B, N, D)
                value=memory,           # (B, N, D)
                key_padding_mask=visited_city_mask  # (B, N)
            )

            query = self.norm1(attn_out + query)   # Residual + Norm
            ff_out = self.ff(query)                # Feed-Forward
            query = self.norm2(ff_out + query)  # Residual + Norm

        attn_out = query.squeeze(1)         # (B, D)

        # 6. --- DECODER: Pointer scoring ---
        ptr_query = self.pointer(attn_out)             # (B, D)

        scores = torch.matmul(
            ptr_query.unsqueeze(1),                    # (B, 1, D)
            memory.transpose(1, 2)                     # (B, D, N)
        ).squeeze(1)                                   # (B, N)

        scores = scores / math.sqrt(self.embed_dim)    # Escalado de scores

        # 7. --- DECODER: Masking y Softmax ---
        scores = scores.masked_fill(visited_city_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)              # (B, N)

        return probs