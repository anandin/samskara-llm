"""
Samskara-LLM: Vedic Cognitive Architecture as Neural Network

Core model implementing Chitta-Manus-Buddhi-Synthesis in a single forward pass.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class AhamkaraLayer(nn.Module):
    """
    Ahamkara (अहंकार) — the ego/identity layer of the Antahkarana.

    In Vedic philosophy, Ahamkara is the ego-sense: the "I am" that synthesizes
    all experience into a unified self. Here it produces a context-conditioned
    identity vector that biases all Manas processing.

    The identity_prior is a stable baseline "self" (learned across all training).
    The context_encoder adapts the identity to what's currently being processed.
    A blend gate controls how much the moment-to-moment context shifts the self.

    This lets the model maintain a coherent personality while remaining sensitive
    to context — exactly the Ahamkara function in Antahkarana philosophy.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Persistent baseline identity — the stable "I" across all contexts
        self.identity_prior = nn.Parameter(torch.randn(d_model) * 0.02)
        # Context encoder: adapts identity to what's currently being processed
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Blend gate: [0=all context, 1=all prior]
        # Initialise near 0.5 so identity starts half-stable, half-adaptive
        self.gate = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, d_model] — pooled input representation (query_vec)
        Returns:
            identity: [batch, d_model] — context-conditioned identity vector
        """
        context_self = self.norm(self.context_encoder(context))      # [B, D]
        blend        = torch.sigmoid(self.gate(context))             # [B, 1]
        identity     = blend * self.identity_prior + (1 - blend) * context_self
        return identity                                               # [B, D]


class ChittaEncoder(nn.Module):
    """
    Differentiable retrieval from learned memory bank.

    Learns to encode query → ChittaField (memory matrix)
    Replaces: pgvector + BM25 + Neo4j with neural retrieval
    """
    
    def __init__(
        self,
        n_seeds: int = 10000,
        d_model: int = 2048,
        n_vrittis: int = 5,
        top_k: int = 50,
        retrieval_temp: float = 0.5,
    ):
        super().__init__()
        self.n_seeds = n_seeds
        self.d_model = d_model
        self.top_k = top_k
        self.retrieval_temp = retrieval_temp
        
        # Learnable seed memory bank [n_seeds, d_model]
        # These are the "seeds" in Chitta - trainable knowledge embeddings
        self.seed_memory = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        
        # Vritti (thought-wave) type embeddings [n_vrittis, d_model]
        # smriti=0, pramana=1, vikalpa=2, viparyaya=3, nidra=4
        self.vritti_embed = nn.Parameter(torch.randn(n_vrittis, d_model) * 0.02)
        
        # Vritti bias scores (learned trust hierarchy)
        # smriti (+0.15), pramana (+0.10), vikalpa (-0.05), viparyaya (0.0)
        self.vritti_bias = nn.Parameter(torch.tensor([0.15, 0.10, -0.05, 0.0, -1.0]))
        
        # Karma scores for each seed (trainable trustworthiness)
        self.karma_scores = nn.Parameter(torch.zeros(n_seeds))
        
        # Query encoder (transforms input to query vector)
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Scoring head (computes relevance)
        self.scoring_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
    def forward(
        self,
        query_vec: torch.Tensor,  # [batch, d_model]
        vritti_types: Optional[torch.Tensor] = None,  # [n_seeds] int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            chitta_field: [batch, top_k, d_model] - retrieved memory
            attention_weights: [batch, n_seeds] - what was attended to
            indices: [batch, top_k] - which seeds were retrieved
        """
        batch_size = query_vec.size(0)
        
        # 1. Encode query
        query_encoded = self.query_encoder(query_vec)  # [batch, d_model]
        
        # 2. Compute similarity to all seeds (batch matrix multiply)
        # [batch, d_model] @ [d_model, n_seeds] = [batch, n_seeds]
        similarities = torch.matmul(query_encoded, self.seed_memory.t())
        
        # 3. Apply karma gating (seeds with karma < -0.3 are suppressed)
        karma_gate = torch.sigmoid((self.karma_scores + 0.3) * 10)  # [n_seeds]
        similarities = similarities * karma_gate.unsqueeze(0)
        
        # 4. Apply vritti bias
        if vritti_types is not None:
            vritti_bias = self.vritti_bias[vritti_types]  # [n_seeds]
            similarities = similarities + vritti_bias.unsqueeze(0)
        
        # 5. Soft top-k selection (differentiable)
        # Use softmax with temperature to get soft top-k
        attention_weights = F.softmax(similarities / self.retrieval_temp, dim=-1)
        
        # 6. Hard top-k for efficiency (during forward, keep gradients via straight-through)
        topk_values, topk_indices = torch.topk(attention_weights, self.top_k, dim=-1)
        
        # 7. Retrieve seeds
        # Gather: [batch, top_k, d_model]
        chitta_field = torch.stack([
            self.seed_memory[indices] for indices in topk_indices
        ])
        
        return chitta_field, attention_weights, topk_indices


class ManasLayer(nn.Module):
    """
    Reactive mind - fast, associative, emotional.
    
    High temperature, pattern-matching, generates tagged signals.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        temperature: float = 1.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # Transformer layers for associative processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Signal classification heads (what Manas "feels")
        self.signal_heads = nn.ModuleDict({
            'FEAR': nn.Linear(d_model, 1),
            'DESIRE': nn.Linear(d_model, 1),
            'PATTERN': nn.Linear(d_model, d_model),  # Pattern embedding
            'RISK': nn.Linear(d_model, 1),
            'OPPORTUNITY': nn.Linear(d_model, 1),
            'NOISE': nn.Linear(d_model, 1),
        })
        
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, d_model]
        chitta_field: torch.Tensor,  # [batch, top_k, d_model]
        ahamkara_bias: torch.Tensor,  # [d_model] - identity
        dharma_bias: torch.Tensor,  # [d_model] - ethical constraints
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            output: [batch, seq, d_model] - processed representation
            signals: dict of signal tensors
        """
        # Add ChittaField to input (mean pool to single vector, broadcast)
        chitta_context = chitta_field.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        x = x + chitta_context
        
        # Add Ahamkara (identity) as bias
        x = x + ahamkara_bias.view(1, 1, -1)
        
        # Add Dharma as attention mask (handled in transformer)
        x = x + dharma_bias.view(1, 1, -1)
        
        # Process with high-temperature noise (associative)
        x = self.layers(x)
        x = x * self.temperature  # Amplify for more associative mixing
        
        # Add exploration noise during training
        if self.training:
            x = x + torch.randn_like(x) * 0.1 * self.temperature
        
        # Generate tagged signals
        signals = {}
        for name, head in self.signal_heads.items():
            if name == 'PATTERN':
                signals[name] = head(x)  # [batch, seq, d_model]
            else:
                signals[name] = torch.sigmoid(head(x).squeeze(-1))  # [batch, seq]
        
        return x, signals


class BuddhiLayer(nn.Module):
    """
    Discriminating intellect - slow, grounded, reasoned.
    
    Deep processing, low temperature, generates options.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 8,
        temperature: float = 0.3,
        n_options: int = 3,
        n_dharma_rules: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.n_options = n_options
        
        # Deeper transformer for reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Graph expansion (simulates Neo4j traversal)
        self.graph_expander = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Option generation
        self.option_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * n_options),
        )
        
        # Dharma (ethical) scoring per option
        self.dharma_scorer = nn.Linear(d_model, n_dharma_rules)
        
        # Option selector
        self.option_selector = nn.Linear(d_model * n_options, n_options)
        
    def forward(
        self,
        manas_output: torch.Tensor,  # [batch, seq, d_model]
        manas_signals: Dict[str, torch.Tensor],
        chitta_field: torch.Tensor,  # [batch, top_k, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            selected_option: [batch, d_model] - dharmic recommendation
            all_options: [batch, n_options, d_model]
            dharma_scores: [batch, n_options, n_dharma_rules]
        """
        # 1. Graph expansion (multi-hop attention over ChittaField)
        expanded_memory, _ = self.graph_expander(
            query=manas_output,
            key=chitta_field,
            value=chitta_field,
        )
        
        # 2. Combine Manas + expanded memory
        x = manas_output + expanded_memory
        
        # 3. Deep, focused reasoning (low temperature)
        x = self.layers(x)
        x = x * self.temperature  # Suppress noise
        
        # 4. Pool to single decision vector
        decision_vec = x.mean(dim=1)  # [batch, d_model]
        
        # 5. Generate options
        options_flat = self.option_generator(decision_vec)  # [batch, d_model * n_options]
        all_options = options_flat.view(-1, self.n_options, self.d_model)
        
        # 6. Score each option against Dharma rules
        dharma_scores = self.dharma_scorer(all_options)  # [batch, n_options, n_rules]
        
        # 7. Select dharmic option (highest compliance)
        option_scores = self.option_selector(options_flat)  # [batch, n_options]
        # Weight by dharma compliance (fewer violations = better)
        dharma_compliance = (dharma_scores > 0).sum(dim=-1).float()  # [batch, n_options]
        combined_scores = option_scores + dharma_compliance
        
        selected_idx = combined_scores.argmax(dim=-1)  # [batch]
        selected_option = all_options[torch.arange(all_options.size(0)), selected_idx]
        
        return selected_option, all_options, dharma_scores


class ElevationRouter(nn.Module):
    """
    Learns when to escalate from Manas to Buddhi.
    """
    
    def __init__(
        self,
        d_model: int,
        threshold: float = 0.7,
    ):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        
        # Stability scoring from Manas output
        self.stability_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
    def forward(
        self,
        manas_output: torch.Tensor,  # [batch, seq, d_model]
        manas_signals: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            elevation_gate: [batch] - probability of elevating to Buddhi
            stability: [batch] - mental stability score
        """
        # Compute mental stability
        # High FEAR + low pattern coherence = low stability
        fear_level = manas_signals['FEAR'].mean(dim=-1)  # [batch]
        
        # Pattern coherence (std of pattern embeddings)
        patterns = manas_signals['PATTERN']  # [batch, seq, d_model]
        pattern_coherence = 1.0 - patterns.std(dim=1).mean(dim=-1)  # [batch]
        
        # Encode stability from features
        pooled = manas_output.mean(dim=1)  # [batch, d_model]
        learned_stability = torch.sigmoid(self.stability_encoder(pooled)).squeeze(-1)
        
        # Combine heuristics + learned
        stability = (learned_stability + (1.0 - fear_level) + pattern_coherence) / 3.0
        
        # Gate: elevate if stability < threshold
        # Use sigmoid to make differentiable
        elevation_gate = torch.sigmoid((self.threshold - stability) * 10)
        
        return elevation_gate, stability


class SynthesisLayer(nn.Module):
    """
    Resolves Manas/Buddhi divergence.
    """
    
    def __init__(
        self,
        d_model: int,
        threshold: float = 0.6,
    ):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        
        # Divergence scoring
        self.divergence_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
        # Synthesis combination weights
        self.combination_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
    def forward(
        self,
        manas_output: torch.Tensor,  # [batch, d_model]
        buddhi_output: torch.Tensor,  # [batch, d_model]
        manas_signals: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            synthesis: [batch, d_model]
            divergence: [batch]
            manas_weight: [batch] - how much Manas signal was preserved
        """
        # Compute divergence
        combined = torch.cat([manas_output, buddhi_output], dim=-1)
        divergence = torch.sigmoid(self.divergence_scorer(combined)).squeeze(-1)
        
        # Compute combination weight (higher = more Manas preservation)
        manas_weight = torch.sigmoid(self.combination_gate(combined)).squeeze(-1)
        
        # Synthesis: weighted combination
        # If high divergence, preserve more Manas insight even though Buddhi has authority
        synthesis = manas_weight.unsqueeze(-1) * manas_output + \
                   (1 - manas_weight.unsqueeze(-1)) * buddhi_output
        
        return synthesis, divergence, manas_weight


class SamskaraLLM(nn.Module):
    """
    Complete Vedic cognitive architecture as neural network.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        d_model = config['d_model']
        
        # Ahamkara (identity) — context-conditioned identity layer
        self.ahamkara = AhamkaraLayer(d_model)
        
        # Dharma (ethical constraints) - learnable rule embeddings
        self.dharma = nn.Parameter(torch.randn(config['n_dharma_rules'], d_model))
        
        # Chitta (memory)
        self.chitta = ChittaEncoder(
            n_seeds=config['n_seeds'],
            d_model=d_model,
            top_k=config['chitta_top_k'],
        )
        
        # Manas (reactive)
        self.manas = ManasLayer(
            d_model=d_model,
            n_heads=config['n_heads'],
            n_layers=config['manas_layers'],
            temperature=config['manas_temp'],
        )
        
        # Router (elevation)
        self.router = ElevationRouter(
            d_model=d_model,
            threshold=config['elevation_threshold'],
        )
        
        # Buddhi (discriminating)
        self.buddhi = BuddhiLayer(
            d_model=d_model,
            n_heads=config['n_heads'],
            n_layers=config['buddhi_layers'],
            temperature=config['buddhi_temp'],
            n_options=config['n_options'],
            n_dharma_rules=config['n_dharma_rules'],
        )
        
        # Synthesis (conflict resolution)
        self.synthesis = SynthesisLayer(
            d_model=d_model,
            threshold=config['synthesis_threshold'],
        )
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, config['vocab_size'], bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        vritti_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through Antahkarana.
        
        Returns dict with logits and interpretability info.
        """
        # 1. Embed input
        # (In practice, use pretrained base model embeddings)
        x = self._embed(input_ids)  # [batch, seq, d_model]
        
        # 2. Encode ChittaField (shared memory substrate)
        query_vec = x.mean(dim=1)  # [batch, d_model]
        chitta_field, attention_weights, seed_indices = self.chitta(
            query_vec, vritti_types
        )

        # 2.5. Ahamkara — context-conditioned identity
        # Identity adapts to the current query while retaining a stable baseline.
        ahamkara_vec = self.ahamkara(query_vec)  # [batch, d_model]

        # 3. Manas processing (fast, associative)
        manas_out, manas_signals = self.manas(
            x, chitta_field, ahamkara_vec, self.dharma.mean(dim=0)
        )
        
        # 4. Router decides elevation
        manas_pooled = manas_out.mean(dim=1)  # [batch, d_model]
        elevation_gate, stability = self.router(manas_pooled, manas_signals)
        
        # 5. Buddhi processing (if elevated)
        # Use soft gate for training, hard for inference
        if self.training:
            # Soft elevation (differentiable)
            buddhi_out, all_options, dharma_scores = self.buddhi(
                manas_out, manas_signals, chitta_field
            )
            # Weighted by elevation gate
            cognitive_out = elevation_gate.unsqueeze(-1) * buddhi_out + \
                          (1 - elevation_gate.unsqueeze(-1)) * manas_pooled
        else:
            # Hard elevation for inference
            elevate = elevation_gate > 0.5
            if elevate.any():
                buddhi_out, all_options, dharma_scores = self.buddhi(
                    manas_out, manas_signals, chitta_field
                )
                cognitive_out = torch.where(
                    elevate.unsqueeze(-1),
                    buddhi_out,
                    manas_pooled
                )
            else:
                cognitive_out = manas_pooled
                all_options = None
                dharma_scores = None
        
        # 6. Synthesis (if divergence detected)
        if self.training or (not self.training and elevation_gate.mean() > 0.5):
            synthesis_out, divergence, manas_weight = self.synthesis(
                manas_pooled, cognitive_out, manas_signals
            )
        else:
            synthesis_out = cognitive_out
            divergence = torch.zeros_like(elevation_gate)
            manas_weight = torch.zeros_like(elevation_gate)
        
        # 7. Project to vocabulary
        logits = self.lm_head(synthesis_out)  # [batch, vocab_size]
        
        return {
            'logits': logits,
            'chitta_attention': attention_weights,
            'chitta_indices': seed_indices,
            'manas_signals': manas_signals,
            'stability': stability,
            'elevation_gate': elevation_gate,
            'buddhi_options': all_options,
            'dharma_scores': dharma_scores,
            'divergence': divergence,
            'manas_weight': manas_weight,
        }
    
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Placeholder - use pretrained embeddings in practice."""
        # This would be replaced with actual base model embeddings
        # lm_head.weight shape: [vocab_size, d_model] — correct for F.embedding
        return F.embedding(input_ids, self.lm_head.weight)


def create_samskara_llm(
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_seeds: int = 10000,
    **kwargs
) -> SamskaraLLM:
    """
    Create Samskara-LLM from a pretrained base model.
    
    Loads base model, adds Chitta-Manus-Buddhi-Synthesis on top.
    """
    # Load base config
    base_config = AutoConfig.from_pretrained(base_model_name)
    
    config = {
        'vocab_size': base_config.vocab_size,
        'd_model': base_config.hidden_size,
        'n_heads': base_config.num_attention_heads,
        'n_seeds': n_seeds,
        'n_dharma_rules': kwargs.get('n_dharma_rules', 20),
        'n_options': kwargs.get('n_options', 3),
        'chitta_top_k': kwargs.get('chitta_top_k', 50),
        'manas_layers': kwargs.get('manas_layers', 4),
        'buddhi_layers': kwargs.get('buddhi_layers', 8),
        'manas_temp': kwargs.get('manas_temp', 1.5),
        'buddhi_temp': kwargs.get('buddhi_temp', 0.3),
        'elevation_threshold': kwargs.get('elevation_threshold', 0.7),
        'synthesis_threshold': kwargs.get('synthesis_threshold', 0.6),
    }
    
    model = SamskaraLLM(config)
    
    # In practice: load base model weights, freeze some, initialize new layers
    # This is a simplified version
    
    return model
