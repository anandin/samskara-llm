"""
Multi-task training objectives for Samskara-LLM.
"""

import torch
import torch.nn.functional as F


def samskara_loss(
    model_output: dict,
    target_ids: torch.Tensor,
    target_signals: dict = None,
    target_elevation: torch.Tensor = None,
    target_dharma: torch.Tensor = None,
    target_outcome: torch.Tensor = None,
    weights: dict = None,
) -> tuple:
    """
    Multi-task loss for Vedic cognitive training.
    
    Args:
        model_output: Dict from SamskaraLLM forward pass
        target_ids: Target token IDs [batch, seq]
        target_signals: Target Manas signals (dict of tensors)
        target_elevation: Binary - should have elevated to Buddhi [batch]
        target_dharma: Which dharma rules apply [batch, n_rules]
        target_outcome: Quality of outcome [-1, 1] [batch]
        weights: Loss component weights
    
    Returns:
        total_loss: Scalar
        loss_dict: Dict of individual losses
    """
    if weights is None:
        weights = {
            'generation': 1.0,
            'signal': 0.3,
            'elevation': 0.5,
            'dharma': 0.4,
            'karma': 0.2,
        }
    
    losses = {}
    
    # 1. Generation loss (standard next-token prediction)
    logits = model_output['logits']
    losses['generation'] = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        ignore_index=-100,
    )
    
    # 2. Signal accuracy (Manas should correctly tag)
    if target_signals is not None:
        signal_loss = 0.0
        for signal_name, target in target_signals.items():
            pred = model_output['manas_signals'][signal_name]
            if signal_name == 'PATTERN':
                # Pattern is embedding, use MSE
                signal_loss += F.mse_loss(pred, target)
            else:
                # Other signals are probabilities
                signal_loss += F.binary_cross_entropy(pred, target)
        losses['signal'] = signal_loss / len(target_signals)
    else:
        losses['signal'] = torch.tensor(0.0, device=logits.device)
    
    # 3. Elevation accuracy (Router should match human judgment)
    if target_elevation is not None:
        losses['elevation'] = F.binary_cross_entropy(
            model_output['elevation_gate'],
            target_elevation.float(),
        )
    else:
        losses['elevation'] = torch.tensor(0.0, device=logits.device)
    
    # 4. Dharma compliance (Buddhi should respect constraints)
    if target_dharma is not None and model_output['dharma_scores'] is not None:
        # Dharma scores: [batch, n_options, n_rules]
        # For selected option, should match target
        # Average over options
        dharma_pred = model_output['dharma_scores'].mean(dim=1)  # [batch, n_rules]
        losses['dharma'] = F.binary_cross_entropy_with_logits(
            dharma_pred,
            target_dharma.float(),
        )
    else:
        losses['dharma'] = torch.tensor(0.0, device=logits.device)
    
    # 5. Karma calibration (predicted seed usefulness matches outcome)
    if target_outcome is not None:
        # Attention-weighted karma should predict outcome
        # Simplified: mean attention correlates with outcome
        mean_attention = model_output['chitta_attention'].mean(dim=-1)  # [batch]
        losses['karma'] = F.mse_loss(mean_attention, (target_outcome + 1) / 2)
    else:
        losses['karma'] = torch.tensor(0.0, device=logits.device)
    
    # Weighted total
    total_loss = sum(
        weights.get(k, 1.0) * v for k, v in losses.items()
    )
    
    return total_loss, losses


def compute_s3_score(model_output: dict, targets: dict) -> dict:
    """
    Compute Samskara System Score (S3) for evaluation.
    
    Returns 7-dimensional score card.
    """
    scores = {}
    
    # 1. ATMAN Fidelity - did we use right cognitive mode?
    if 'target_elevation' in targets:
        elevation_acc = (
            (model_output['elevation_gate'] > 0.5).float() == targets['target_elevation']
        ).float().mean()
        scores['elevation_accuracy'] = elevation_acc.item()
    else:
        scores['elevation_accuracy'] = 0.0
    
    # 2. Chitta Quality - retrieved relevant seeds?
    # Proxy: high attention weights on seeds matching target domain
    scores['retrieval_precision'] = model_output['chitta_attention'].max(dim=-1)[0].mean().item()
    
    # 3. Divergence - resolved conflicts well?
    if 'target_signals' in targets:
        fear_match = F.cosine_similarity(
            model_output['manas_signals']['FEAR'].unsqueeze(-1),
            targets['target_signals']['FEAR'].unsqueeze(-1),
            dim=0
        ).mean()
        scores['signal_preservation'] = fear_match.item()
    else:
        scores['signal_preservation'] = 0.0
    
    # 4. Karma - predicted outcome quality?
    if 'target_outcome' in targets:
        predicted_quality = model_output['chitta_attention'].sum(dim=-1)
        actual_quality = (targets['target_outcome'] + 1) / 2
        karma_corr = torch.corrcoef(
            torch.stack([predicted_quality, actual_quality])
        )[0, 1]
        scores['karma_correlation'] = karma_corr.item() if not torch.isnan(karma_corr) else 0.0
    else:
        scores['karma_correlation'] = 0.0
    
    # 5. Dharma - compliant?
    if model_output['dharma_scores'] is not None:
        violations = (model_output['dharma_scores'] < 0).sum(dim=-1).float().mean()
        scores['dharma_compliance'] = 1.0 - (violations / model_output['dharma_scores'].size(-1)).item()
    else:
        scores['dharma_compliance'] = 1.0
    
    # 6. Cognitive Health - stability maintained?
    scores['stability'] = model_output['stability'].mean().item()
    
    # 7. Efficiency - not overthinking?
    scores['elevation_rate'] = (model_output['elevation_gate'] > 0.5).float().mean().item()
    # Good if 10-30% elevation (not everything needs Buddhi)
    scores['efficiency'] = 1.0 - abs(scores['elevation_rate'] - 0.2) * 2
    
    # Composite S3 Score
    scores['s3_total'] = (
        0.20 * scores['elevation_accuracy'] +
        0.20 * scores['retrieval_precision'] +
        0.15 * scores['signal_preservation'] +
        0.15 * scores['karma_correlation'] +
        0.10 * scores['dharma_compliance'] +
        0.10 * scores['stability'] +
        0.10 * scores['efficiency']
    )
    
    return scores
