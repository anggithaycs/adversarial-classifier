"""
model.py 
=> Bidirectional Transformer encoder with binary classification head.
 
Architecture
------------
- Learned token embeddings + learned positional embeddings
- N stacked Transformer encoder layers (no causal mask → bidirectional)
- [CLS] token pooling → linear classification head → logit for label=1
 
Input format (per example):
    [CLS] source_sv tokens [SEP] translation_en tokens [SEP]
 
The model outputs a single logit (pre-sigmoid). Loss is computed externally
in train.py using focal loss so that sigmoid + threshold can be adjusted
independently at inference time.
"""
 
import math
import torch
import torch.nn as nn
 
 
# ── Multi-head Self-Attention ─────────────────────────────────────────────────
 
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
 
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
 
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
 
        self.attn_dropout = nn.Dropout(dropout)
 
    def forward(
        self,
        x: torch.Tensor,                     # (B, T, d_model)
        key_padding_mask: torch.Tensor = None # (B, T) True = ignore
    ) -> torch.Tensor:
        B, T, _ = x.shape
 
        # Project and reshape to (B, heads, T, d_head)
        def project_and_split(linear, inp):
            out = linear(inp)                           # (B, T, d_model)
            out = out.view(B, T, self.n_heads, self.d_head)
            return out.transpose(1, 2)                  # (B, heads, T, d_head)
 
        Q = project_and_split(self.W_q, x)
        K = project_and_split(self.W_k, x)
        V = project_and_split(self.W_v, x)
 
        # Scaled dot-product attention
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, heads, T, T)
 
        if key_padding_mask is not None:
            # mask: (B, T) → (B, 1, 1, T) so it broadcasts over heads and Q
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))
 
        # NOTE: no causal mask — full bidirectional attention
        attn_weights = torch.softmax(scores, dim=-1)            # (B, heads, T, T)
        attn_weights = self.attn_dropout(attn_weights)
 
        context = torch.matmul(attn_weights, V)                 # (B, heads, T, d_head)
        context = context.transpose(1, 2).contiguous()          # (B, T, heads, d_head)
        context = context.view(B, T, self.d_model)              # (B, T, d_model)
 
        return self.W_o(context)
 
 
# ── Feed-Forward Block ────────────────────────────────────────────────────────
 
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
# ── Encoder Layer ─────────────────────────────────────────────────────────────
 
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff      = FeedForward(d_model, d_ff, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop1   = nn.Dropout(dropout)
        self.drop2   = nn.Dropout(dropout)
 
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-norm residual (more stable than post-norm for small models)
        x = x + self.drop1(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x
 
 
# ── Full Model ────────────────────────────────────────────────────────────────
 
class StyleGuideClassifier(nn.Module):
    """
    Bidirectional Transformer encoder for binary style-guide compliance.
 
    Args:
        vocab_size  : size of the tokeniser vocabulary
        d_model     : embedding / hidden dimension
        n_heads     : number of attention heads
        n_layers    : number of stacked encoder layers
        d_ff        : feed-forward inner dimension (typically 4 * d_model)
        max_len     : maximum sequence length (source + sep + translation + sep + cls)
        dropout     : dropout probability
        pad_idx     : vocabulary index of the [PAD] token
    """
 
    def __init__(
        self,
        vocab_size : int,
        d_model    : int = 256,
        n_heads    : int = 4,
        n_layers   : int = 4,
        d_ff       : int = 1024,
        max_len    : int = 512,
        dropout    : float = 0.1,
        pad_idx    : int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
 
        # Token + positional embeddings (both learned)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)
        self.emb_norm  = nn.LayerNorm(d_model)
 
        # Encoder stack
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
 
        # Classification head: [CLS] hidden state → single logit
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),   # single logit; sigmoid applied at loss time
        )
 
        self._init_weights()
 
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
 
    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)  token ids
        return_embeddings: bool = False,   # set True for gradient saliency
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
 
        # Padding mask: True where token is PAD
        key_padding_mask = (input_ids == self.pad_idx)  # (B, T)
 
        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)  # (B, T, d_model)
        x = self.emb_norm(x)
        x = self.emb_drop(x)
 
        # Store embeddings for gradient saliency (used in analyze.py)
        if return_embeddings:
            x.retain_grad()
            self._last_embeddings = x
 
        # Encoder layers
        for layer in self.layers:
            x = layer(x, key_padding_mask)
 
        x = self.final_norm(x)
 
        # Pool the [CLS] token (position 0)
        cls_hidden = x[:, 0, :]   # (B, d_model)
 
        # Binary logit (no sigmoid here — handled in loss function)
        logit = self.classifier(cls_hidden).squeeze(-1)  # (B,)
        return logit
 
    def get_embedding_output(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedding layer output with gradients enabled.
        Used by analyze.py for saliency computation:
            grad = d(logit) / d(embedding)
        """
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_norm(x)
        return x