"""
ContextPilot RL components for verl.

This package contains additions to verl that implement (a subset of) the
ContextPilot training scheme described in the paper.

Stage 1 (currently implemented, no partial rollout):
    * snapshot segmentation at all 8 context-editing (ce) tool calls
      (4 offloading + 4 memory writing tools)
    * R_pen: -0.5 penalty on the final trajectory reward when ANY tool call
      failed during the trajectory
    * token-level loss masking via existing snapshot mechanism in
      ``statelm_agent_loop``
    * each snapshot is treated as an independent training sample and gets a
      group-relative GRPO advantage among the same query (default GRPO already
      does this once snapshots are emitted; the ``contextpilot_grpo`` adv
      estimator is registered as a forward-compatible alias so we can wire in
      the partial-rollout / subtree-mean credit assignment in stage 2 without
      touching the trainer wiring again).

The whole stack is gated by the config flag
``actor_rollout_ref.rollout.multi_turn.contextpilot.enable``; when this flag
is False the behaviour is identical to vanilla verl/StateLM.
"""

from . import adv  # noqa: F401
from . import constants  # noqa: F401

__all__ = ["adv", "constants"]
