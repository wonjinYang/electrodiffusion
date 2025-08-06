class PositionalEncoding(nn.Module):
    """시간 임베딩을 위한 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """시간 t를 고차원 벡터로 인코딩"""
        t_scaled = (t * (self.pe.size(0) - 1)).long().clamp(0, self.pe.size(0) - 1)
        return self.pe[t_scaled]

class AttentionBlock(nn.Module):
    """Self-attention 블록 (분자 상호작용)"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class PhysicsInformedScoreNetwork(nn.Module):
    """물리학 기반 Score function 신경망"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 6, num_heads: int = 8,
                 constants: Optional[PhysicalConstants] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.constants = constants or PhysicalConstants()
        
        # Time embedding
        self.time_encoder = PositionalEncoding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Physics-aware layers
        self.physics_layers = nn.ModuleList([
            self._make_physics_layer(hidden_dim) for _ in range(num_layers // 2)
        ])
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads) for _ in range(num_layers // 2)
        ])
        
        # Output heads
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.force_head = nn.Linear(hidden_dim, input_dim)
        
    def _make_physics_layer(self, dim: int) -> nn.Module:
        """물리학적 inductive bias 레이어"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                external_conditions: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = x.shape[0]
        
        # Time embedding
        t_embed = self.time_encoder(t)
        t_embed = self.time_proj(t_embed)
        
        # Input projection
        h = self.input_proj(x)
        h = h + t_embed
        
        # Physics layers
        for layer in self.physics_layers:
            h = h + layer(h)  # Residual connection
        
        # Attention processing
        h_reshaped = h.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        for attn_layer in self.attention_layers:
            h_reshaped = attn_layer(h_reshaped)
        
        h = h_reshaped.squeeze(1)
        
        # External conditions modulation
        if external_conditions is not None:
            if 'electric_field' in external_conditions:
                E_field = external_conditions['electric_field']
                E_embed = self._embed_electric_field(E_field, h.device)
                h = h + E_embed
            
            if 'concentration_gradient' in external_conditions:
                grad_c = external_conditions['concentration_gradient']
                grad_embed = self._embed_concentration(grad_c, h.device)
                h = h + grad_embed
        
        # Output predictions
        score = self.score_head(h)
        energy = self.energy_head(h)
        force = self.force_head(h)
        
        return {
            'score': score,
            'energy': energy,
            'force': force
        }
    
    def _embed_electric_field(self, E_field: torch.Tensor, device: torch.device) -> torch.Tensor:
        """전기장 임베딩"""
        E_mag = torch.norm(E_field, dim=-1, keepdim=True)
        E_embed = torch.zeros(E_field.shape[0], self.hidden_dim, device=device)
        E_embed[:, :3] = E_field
        E_embed[:, 3] = E_mag.squeeze()
        return E_embed
    
    def _embed_concentration(self, grad_c: torch.Tensor, device: torch.device) -> torch.Tensor:
        """농도 구배 임베딩"""
        grad_embed = torch.zeros(grad_c.shape[0], self.hidden_dim, device=device)
        grad_embed[:, 4:7] = grad_c
        return grad_embed
