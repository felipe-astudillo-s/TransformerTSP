import torch
import torch.nn as nn
import math

class TSPDecoder(nn.Module):
    def __init__(self, embedding_dim=128, n_heads=8):
        """
        Decoder basado en 'Attention, Learn to Solve Routing Problems!'
        Calcula la probabilidad de visitar el siguiente nodo dado el contexto actual.
        """
        super(TSPDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        
        # Parámetros aprendibles para el primer paso (placeholder)
        # Cuando t=1, no hay "nodo anterior", usamos este vector v_learnable
        self.v_first = nn.Parameter(torch.Tensor(embedding_dim))
        self.v_last = nn.Parameter(torch.Tensor(embedding_dim))
        
        # W_q: Proyecta el contexto (graph_emb + last_node + first_node) a Query
        # El contexto tiene dimensión 3 * embedding_dim (o 2*emb + graph)
        # Nota: En el paper original concatenan [graph_emb, last_node, first_node]
        self.W_context = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
        
        # Mecanismo de Atención (Glismpse + Pointer)
        # Usamos una capa de atención personalizada para tener control sobre los logits
        self.project_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # Single-Head Attention final para calcular probabilidades (Logits)
        # Se usa tanh para escalar los logits antes del softmax (C * tanh(u))
        self.W_q_final = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k_final = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # Inicialización de pesos
        self._init_parameters()

    def _init_parameters(self):
        # Inicialización uniforme estándar para parámetros
        nn.init.uniform_(self.v_first, -1/math.sqrt(self.embedding_dim), 1/math.sqrt(self.embedding_dim))
        nn.init.uniform_(self.v_last, -1/math.sqrt(self.embedding_dim), 1/math.sqrt(self.embedding_dim))

    def forward(self, encoder_output, graph_embedding, last_node_embedding, first_node_embedding, mask):
        """
        Paso de decodificación individual (un time-step).
        
        Args:
            encoder_output: [batch, num_nodes, embed_dim] (Keys/Values globales)
            graph_embedding: [batch, embed_dim] (Resumen del grafo)
            last_node_embedding: [batch, embed_dim] (Nodo visitado en t-1) o None si t=0
            first_node_embedding: [batch, embed_dim] (Primer nodo del tour) o None
            mask: [batch, num_nodes] (True = nodo ya visitado/inválido, False = disponible)
            
        Returns:
            probs: [batch, num_nodes] Probabilidades sobre todos los nodos para el siguiente paso
            logits: [batch, num_nodes] Logits crudos (útiles para calcular loss con CrossEntropy)
        """
        batch_size, num_nodes, _ = encoder_output.size()
        
        # 1. Construir el Vector de Contexto [cite: 527, 530]
        # Contexto = [Graph Embedding, Last Node, First Node]
        
        if last_node_embedding is None:
            # Paso t=0: Usamos placeholders aprendibles
            # Expandimos los parámetros para coincidir con el batch size
            v_f = self.v_first.expand(batch_size, -1)
            v_l = self.v_last.expand(batch_size, -1)
            context = torch.cat([graph_embedding, v_l, v_f], dim=1)
        else:
            # Pasos t>0: Usamos los embeddings reales pasados como argumento
            context = torch.cat([graph_embedding, last_node_embedding, first_node_embedding], dim=1)
            
        # 2. Calcular Query (Q) desde el Contexto [cite: 534]
        # [batch, embed_dim]
        query = self.W_context(context)
        
        # 3. Calcular Keys (K) y Values (V) desde el output del Encoder
        # Nota: K y V son estáticos por instancia, pero se recalculan aquí por simplicidad.
        # En producción se cachean para eficiencia.
        key = self.project_k(encoder_output) # [batch, nodes, embed_dim]
        # value = self.project_v(encoder_output) # No necesitamos V explícito para el puntero final, solo para glimpses intermedios si los hubiera
        
        # 4. Calcular Compatibilidad (Atención) - Single Head para el puntero final [cite: 550]
        # Q: [batch, 1, dim], K: [batch, nodes, dim] -> scores: [batch, 1, nodes]
        
        # Expandimos query para matmul: [batch, 1, dim]
        query_expanded = query.unsqueeze(1)
        
        # Producto punto escalado (Scaled Dot-Product)
        # u_ij = (Q^T * K) / sqrt(d)
        scores = torch.matmul(query_expanded, key.transpose(1, 2)) / math.sqrt(self.embedding_dim)
        scores = scores.squeeze(1) # [batch, nodes]
        
        # 5. Clipping con Tanh (Truco clave del paper SOTA) [cite: 550]
        # "we clip the result within [-C, C] (C=10) using tanh"
        # Esto evita que los gradientes exploten y ayuda a la convergencia
        scores = 10 * torch.tanh(scores)
        
        # 6. Enmascaramiento (Masking) [cite: 536, 550]
        # Ponemos -infinito a los nodos ya visitados para que su probabilidad sea 0
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        # 7. Softmax para obtener probabilidades [cite: 555]
        probs = torch.softmax(scores, dim=-1)
        
        return probs, scores