# PyTorch LLM Architecture & Data Flow Cheat Sheet

**Advanced techniques for tweaking LLM architectures and optimizing data flow**

---

## 🚀 Custom Transformer Components

### Multi-Head Attention Variants

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Multi-Query Attention (MQA) - Memory efficient
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)  # Multiple query heads
        self.w_k = nn.Linear(d_model, self.d_k)  # Single key head
        self.w_v = nn.Linear(d_model, self.d_k)  # Single value head
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Q: [B, T, d_model] -> [B, num_heads, T, d_k]
        q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # K, V: [B, T, C] -> [B, 1, T, d_k] -> broadcast to [B, num_heads, T, d_k]
        k = self.w_k(x).view(B, T, 1, self.d_k).transpose(1, 2).expand(-1, self.num_heads, -1, -1)
        v = self.w_v(x).view(B, T, 1, self.d_k).transpose(1, 2).expand(-1, self.num_heads, -1, -1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # [B, num_heads, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)

# Grouped Query Attention (GQA) - Balanced approach
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.groups = num_heads // num_kv_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Q: [B, T, d_model] -> [B, num_heads, T, d_k]
        q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # K, V: [B, T, C] -> [B, num_kv_heads, T, d_k]
        k = self.w_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat K, V for each group
        k = k.repeat_interleave(self.groups, dim=1)  # [B, num_heads, T, d_k]
        v = v.repeat_interleave(self.groups, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)

# FlexAttention - PyTorch 2.5+ with custom score modifiers
def causal_mask(score, batch, head, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float('-inf'))

def sliding_window_mask(window_size):
    def mask_fn(score, batch, head, q_idx, kv_idx):
        return torch.where(
            torch.abs(q_idx - kv_idx) <= window_size,
            score, float('-inf')
        )
    return mask_fn

# Usage with FlexAttention (PyTorch 2.5+)
from torch.nn.attention.flex_attention import flex_attention
# output = flex_attention(query, key, value, score_mod=causal_mask)
```

### Advanced Feed-Forward Networks

```python
# SwiGLU Activation (used in LLaMA, PaLM)
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # Up projection
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gate = F.silu(self.w1(x))  # SiLU activation
        up = self.w3(x)
        return self.w2(self.dropout(gate * up))

# Mixture of Experts (MoE) Layer
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Router network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks (batched for efficiency)
        self.experts_w1 = nn.Parameter(torch.randn(num_experts, d_model, d_ff))
        self.experts_w2 = nn.Parameter(torch.randn(num_experts, d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute routing weights
        router_logits = self.gate(x)  # [B, T, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Reshape for batched expert computation
        x_expanded = x.unsqueeze(2).expand(B, T, self.top_k, C)  # [B, T, top_k, C]
        
        # Gather expert weights
        expert_w1 = self.experts_w1[top_k_indices]  # [B, T, top_k, C, d_ff]
        expert_w2 = self.experts_w2[top_k_indices]  # [B, T, top_k, d_ff, C]
        
        # Apply experts
        hidden = torch.matmul(x_expanded.unsqueeze(-2), expert_w1).squeeze(-2)  # [B, T, top_k, d_ff]
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        outputs = torch.matmul(hidden.unsqueeze(-2), expert_w2).squeeze(-2)  # [B, T, top_k, C]
        
        # Weighted combination
        final_output = torch.sum(outputs * top_k_weights.unsqueeze(-1), dim=2)  # [B, T, C]
        
        return final_output
```

## 🎯 Positional Encodings

### Rotary Positional Embedding (RoPE)

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self.register_buffer('cached_freqs', None)
        self.register_buffer('cached_seq_len', torch.tensor(0))
        
    def _update_cache(self, seq_len, device):
        if self.cached_freqs is None or seq_len > self.cached_seq_len:
            self.cached_seq_len = torch.tensor(seq_len)
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
            freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
            self.cached_freqs = torch.stack([freqs.cos(), freqs.sin()], dim=0)
            
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
        
    def forward(self, q, k, seq_len=None, offset=0):
        if seq_len is None:
            seq_len = q.shape[-2]
            
        self._update_cache(seq_len + offset, q.device)
        
        cos_freqs = self.cached_freqs[0, offset:offset+seq_len]  # [seq_len, dim]
        sin_freqs = self.cached_freqs[1, offset:offset+seq_len]
        
        # Apply rotary embedding
        q_rotated = q * cos_freqs + self.rotate_half(q) * sin_freqs
        k_rotated = k * cos_freqs + self.rotate_half(k) * sin_freqs
        
        return q_rotated, k_rotated

# ALiBi (Attention with Linear Biases)
class ALiBiPositionalBias(nn.Module):
    def __init__(self, num_heads, max_seq_len=8192):
        super().__init__()
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Precompute bias matrix
        bias = self._build_bias_matrix(max_seq_len)
        self.register_buffer('bias', bias)
        
    def _get_slopes(self, num_heads):
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Handle non-power-of-2 heads
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(self._get_slopes(2*closest_power_of_2)[:num_heads-closest_power_of_2])
            
        return torch.tensor(slopes).view(-1, 1, 1)
    
    def _build_bias_matrix(self, max_seq_len):
        context_position = torch.arange(max_seq_len)[:, None]
        memory_position = torch.arange(max_seq_len)[None, :]
        relative_position = memory_position - context_position
        return relative_position.unsqueeze(0)  # [1, seq_len, seq_len]
    
    def forward(self, seq_len):
        bias = self.bias[:, :seq_len, :seq_len]
        return self.slopes * bias  # [num_heads, seq_len, seq_len]
```

## 📏 Advanced Normalization

```python
# Root Mean Square Layer Normalization (RMSNorm)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# Layer Normalization with configurable axis
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

# Pre/Post-LayerNorm placement
class TransformerBlockPreNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        # Pre-normalization
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        x = x + self.ff(self.ln2(x))
        return x

# Group Normalization for specific use cases
class GroupNorm1D(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x):
        # x: [B, T, C] -> [B, C, T] for GroupNorm -> [B, T, C]
        B, T, C = x.shape
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        return x.transpose(1, 2)  # [B, T, C]
```

## 🔧 Parameter-Efficient Fine-Tuning (PEFT)

### LoRA (Low-Rank Adaptation)

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Original frozen layer
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False
        
        # LoRA parameters
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        original_output = self.linear(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return original_output + lora_output * self.scaling

# Apply LoRA to existing model
def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """Apply LoRA to specific modules in a transformer model."""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = model.get_submodule(parent_name)
            
            # Replace with LoRA version
            lora_layer = LoRALinear(
                module.in_features, 
                module.out_features, 
                rank=rank, 
                alpha=alpha
            )
            # Copy original weights
            lora_layer.linear.weight.data = module.weight.data.clone()
            setattr(parent_module, child_name, lora_layer)
    
    return model

# AdaLoRA - Adaptive rank allocation
class AdaLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, initial_rank=8, 
                 alpha=16, dropout=0.1, target_rank=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_rank = initial_rank
        self.target_rank = target_rank
        self.alpha = alpha
        
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False
        
        # Adaptive LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(initial_rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, initial_rank))
        
        # Importance scores for rank pruning
        self.importance_scores = nn.Parameter(torch.ones(initial_rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply importance masking
        mask = torch.sigmoid(self.importance_scores)
        lora_A_masked = self.lora_A * mask.unsqueeze(1)
        lora_B_masked = self.lora_B * mask.unsqueeze(0)
        
        original_output = self.linear(x)
        lora_output = F.linear(F.linear(self.dropout(x), lora_A_masked), lora_B_masked.t())
        
        return original_output + lora_output * (self.alpha / self.initial_rank)
```

### Adapters and Prefix Tuning

```python
# Bottleneck Adapters
class BottleneckAdapter(nn.Module):
    def __init__(self, d_model, bottleneck_size=64, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual

# Prefix Tuning
class PrefixTuning(nn.Module):
    def __init__(self, num_layers, num_heads, head_dim, prefix_length=20):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Learnable prefix parameters
        self.prefix_keys = nn.Parameter(
            torch.randn(num_layers, prefix_length, num_heads, head_dim)
        )
        self.prefix_values = nn.Parameter(
            torch.randn(num_layers, prefix_length, num_heads, head_dim)
        )
        
    def get_prefix_states(self, batch_size, layer_idx):
        prefix_keys = self.prefix_keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
        prefix_values = self.prefix_values[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
        return prefix_keys, prefix_values
```

## 💾 Memory Optimization

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# Method 1: Function-based checkpointing
def checkpoint_wrapper(func):
    def wrapper(*args, **kwargs):
        return checkpoint(func, *args, **kwargs)
    return wrapper

# Method 2: Module-based checkpointing
class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Use checkpointing to save memory
        x = x + checkpoint(self._attention_block, self.ln1(x), mask)
        x = x + checkpoint(self.ff, self.ln2(x))
        return x
    
    def _attention_block(self, x, mask):
        return self.attn(x, x, x, attn_mask=mask)[0]

# Method 3: Sequential checkpointing for transformer stack
class CheckpointedTransformerStack(nn.Module):
    def __init__(self, layers, segments=4):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.segments = segments
        
    def forward(self, x, mask=None):
        # Divide layers into segments for checkpointing
        return checkpoint_sequential(
            self.layers, self.segments, x, mask, preserve_rng_state=True
        )

# Advanced: Selective activation checkpointing
def selective_checkpoint(module, inputs, condition_fn):
    """Only checkpoint if condition is met (e.g., layer depth, memory usage)"""
    if condition_fn(module):
        return checkpoint(module, *inputs)
    else:
        return module(*inputs)
```

### Gradient Accumulation and Scaling

```python
class GradientAccumulator:
    def __init__(self, model, accumulation_steps, max_grad_norm=1.0):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        
    def backward_and_step(self, loss, optimizer, scheduler=None):
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.current_step += 1
        
        # Only step optimizer every accumulation_steps
        if self.current_step % self.accumulation_steps == 0:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
    def finalize_step(self, optimizer, scheduler=None):
        """Handle partial accumulation at end of epoch"""
        if self.current_step % self.accumulation_steps != 0:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

# Mixed precision training with automatic scaling
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def training_step(self, batch):
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## 🎛️ Custom Loss Functions & Training Techniques

```python
# Label Smoothing for Language Modeling
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.epsilon) + (1 - one_hot) * self.epsilon / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Contrastive Learning Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features1, features2):
        # Normalize features
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        batch_size = features1.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Create labels (positive pairs are diagonal)
        labels = torch.arange(batch_size, device=features1.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

# Custom learning rate schedulers
class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_i:
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                   for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi)) / 2 for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i *= self.T_mult
        else:
            self.T_cur = epoch
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
```

## 🔍 Advanced PyTorch Utilities

### Custom Data Flow and Hooks

```python
# Forward hooks for layer analysis
class LayerAnalyzer:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    def register_hooks(self, layer_names):
        for name, module in self.model.named_modules():
            if name in layer_names:
                # Forward hook
                fwd_hook = module.register_forward_hook(
                    lambda module, input, output, name=name: 
                    self._save_activation(name, output)
                )
                # Backward hook
                bwd_hook = module.register_backward_hook(
                    lambda module, grad_input, grad_output, name=name: 
                    self._save_gradient(name, grad_output)
                )
                self.hooks.extend([fwd_hook, bwd_hook])
    
    def _save_activation(self, name, activation):
        self.activations[name] = activation.detach().clone()
        
    def _save_gradient(self, name, gradient):
        if gradient[0] is not None:
            self.gradients[name] = gradient[0].detach().clone()
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# Dynamic layer freezing/unfreezing
class DynamicFreezing:
    def __init__(self, model):
        self.model = model
        self.freeze_schedule = {}
        
    def freeze_layers(self, layer_pattern, freeze=True):
        for name, param in self.model.named_parameters():
            if layer_pattern in name:
                param.requires_grad = not freeze
                
    def schedule_freezing(self, schedule):
        """Schedule: {step: [(pattern, freeze), ...]}"""
        self.freeze_schedule = schedule
        
    def apply_schedule(self, current_step):
        if current_step in self.freeze_schedule:
            for pattern, freeze in self.freeze_schedule[current_step]:
                self.freeze_layers(pattern, freeze)

# Custom autograd functions
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def reverse_layer(x, alpha=1.0):
    return ReverseLayerF.apply(x, alpha)

# Gradient clipping variants
def adaptive_gradient_clipping(model, clip_factor=0.01):
    """Adaptive gradient clipping based on parameter norm"""
    total_norm = 0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    if param_count > 0:
        clip_coef = clip_factor / (total_norm / param_count + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

### Model Surgery and Weight Manipulation

```python
# Weight initialization schemes
def init_weights_normal(module, std=0.02):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)

def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()

def init_weights_kaiming(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(module.bias, -bound, bound)

# Model pruning
def magnitude_pruning(model, sparsity=0.2):
    """Remove smallest magnitude weights"""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Global magnitude pruning
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=sparsity,
    )

# Weight averaging (Model soups)
def average_model_weights(models, weights=None):
    """Average weights from multiple models"""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    averaged_dict = {}
    model_dicts = [model.state_dict() for model in models]
    
    for key in model_dicts[0]:
        averaged_dict[key] = sum(
            weight * model_dict[key] for weight, model_dict in zip(weights, model_dicts)
        )
    
    return averaged_dict
```

## 🚀 Performance Optimization Tips

```python
# Compile model for faster execution (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')  # or 'reduce-overhead', 'default'

# Use scaled dot-product attention (PyTorch 2.0+)
# Automatically uses FlashAttention when available
from torch.nn.functional import scaled_dot_product_attention

def efficient_attention(q, k, v, mask=None):
    return scaled_dot_product_attention(
        q, k, v, 
        attn_mask=mask, 
        dropout_p=0.1 if self.training else 0.0,
        is_causal=True if mask is None else False
    )

# Memory-efficient attention for long sequences
def memory_efficient_attention(q, k, v, chunk_size=512):
    B, H, T, D = q.shape
    assert T % chunk_size == 0
    
    chunks = T // chunk_size
    output = torch.zeros_like(q)
    
    for i in range(chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        
        q_chunk = q[:, :, start_idx:end_idx, :]
        output[:, :, start_idx:end_idx, :] = scaled_dot_product_attention(
            q_chunk, k, v, is_causal=True
        )
    
    return output

# Efficient sequence packing for training
def pack_sequences(sequences, max_length):
    """Pack variable-length sequences to reduce padding"""
    packed_sequences = []
    current_length = 0
    current_batch = []
    
    for seq in sequences:
        if current_length + len(seq) <= max_length:
            current_batch.append(seq)
            current_length += len(seq)
        else:
            if current_batch:
                packed_sequences.append(torch.cat(current_batch))
            current_batch = [seq]
            current_length = len(seq)
    
    if current_batch:
        packed_sequences.append(torch.cat(current_batch))
    
    return packed_sequences
```

---

## 🔗 Key Resources & References

- **FlexAttention**: PyTorch 2.5+ flexible attention API
- **torchtune**: Meta's PyTorch library for LLM fine-tuning
- **transformers**: Hugging Face transformers library
- **accelerate**: Hugging Face library for distributed training
- **DeepSpeed**: Microsoft's training acceleration library
- **FairScale**: Meta's training utilities

Remember to profile your model with tools like `torch.profiler` and use `torch.compile()` for production deployments!