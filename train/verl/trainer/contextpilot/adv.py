"""ContextPilot advantage estimator.

Stage 1 implementation:
    Without partial rollout, the "subtree" of every emitted snapshot is the
    chain of snapshots that share the same trajectory.  In that case the
    paper's bottom-up subtree-mean reduces to broadcasting the trajectory's
    final reward to every snapshot, which is exactly what the existing
    statelm agent loop already does (every snapshot inherits
    ``final_trajectory_reward``).  The remaining piece is therefore the
    GRPO group-relative normalisation across all snapshots that share the
    same query (uid).

We register a dedicated ``contextpilot_grpo`` estimator (alias of GRPO with
group-relative z-score normalisation) so we can wire in the partial-rollout
subtree mean in stage 2 without changing the trainer plumbing.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config.algorithm import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("contextpilot_grpo")
def compute_contextpilot_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ContextPilot GRPO advantage.

    Behaviour: identical to vanilla GRPO group-relative z-score normalisation.
    Each row of ``token_level_rewards`` already corresponds to a snapshot
    (because :class:`StatelmToolAgentLoop` emits one row per snapshot and
    every snapshot inherits the final trajectory reward, which equals the
    subtree-mean in the no-partial-rollout regime).  Grouping by the query
    uid therefore gives us the paper's group-relative advantage over all
    snapshots produced for the same query.
    """
    norm_adv_by_std_in_grpo = True
    if config is not None:
        try:
            norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
        except Exception:
            norm_adv_by_std_in_grpo = True

    scores = token_level_rewards.sum(dim=-1)
    id2score: dict = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(stacked)
                id2std[idx] = torch.std(stacked)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages
