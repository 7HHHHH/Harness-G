# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""
from __future__ import annotations

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_snc_span_advantage(snc_token_rewards: torch.Tensor,
                               support_mask: torch.Tensor,
                               index: torch.Tensor,
                               epsilon: float = 1e-6,
                               clamp: float = 5.0,
                               scale_floor: float = 5e-4) -> torch.Tensor:
    """Span-local SNC advantage with one batch-global calibrated scale.

    SNC credit is already a local signal with a sign: + = useful step,
    - = harmful step.  We must NOT compress it into a per-sequence scalar
    first: two rollouts crediting *different* action spans with the same total
    (e.g. [1,0,0,0] vs [0,1,0,0]) would both sum to 1 and yield zero advantage
    after group centering — exactly the sequence-level flattening this whole
    estimator exists to fix.

    A per-uid unit-variance normalization makes arbitrarily small non-zero
    deltas as strong as large, reliable IG. Use one scale for the whole batch:
      * pool = every support-token credit in the batch
      * scale = max(std(pool), ``scale_floor``)
      * adv[i,t] = snc_credit[i,t] / scale

    No mean is subtracted, so local signs remain intact. The fixed floor is the
    noise calibration: sub-floor signals remain proportionally small instead
    of being forced to O(1). ``index`` remains in the signature for API
    compatibility but no longer controls scaling.

    Args:
        snc_token_rewards: (bs, response_length) per-token SNC credit.
        support_mask: (bs, response_length) bool, typically
            ``(snc_token_rewards != 0) & loss_mask``.
        index: retained for estimator API compatibility; scaling is global.
        epsilon: retained for estimator API compatibility.
        clamp: symmetric clamp on the final per-token advantage.

    Returns:
        advantages: (bs, response_length) span-local advantage, non-zero only
        on support tokens; zero elsewhere; never nan/inf.
    """
    support = support_mask.bool()
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    pool = snc_token_rewards[support]
    if pool.numel() == 0:
        return torch.zeros_like(snc_token_rewards)
    std = pool.to(torch.float32).std(unbiased=False) if pool.numel() > 1 else pool.abs().mean()
    floor = torch.as_tensor(scale_floor, device=std.device, dtype=std.dtype)
    scale = torch.maximum(std, floor).to(snc_token_rewards.dtype)
    advantages = snc_token_rewards * support / scale
    advantages = advantages.clamp(-clamp, clamp)
    return advantages


def compute_grpo_snc_advantage(token_level_rewards: torch.Tensor,
                               snc_token_rewards: torch.Tensor,
                               eos_mask: torch.Tensor,
                               index: torch.Tensor,
                               loss_mask: torch.Tensor | None = None,
                               epsilon: float = 1e-6,
                               snc_lambda: float = 0.2,
                               snc_clamp: float = 5.0,
                               snc_scale_floor: float = 5e-4,
                               return_snc_advantage: bool = False):
    """GRPO outcome advantage + lambda * SNC span-local advantage.

    Two independent advantage channels, each normalized on its own:
      * outcome path: standard GRPO over ``token_level_rewards`` (outcome only).
      * SNC path: span-local advantage over ``snc_token_rewards`` (see
        :func:`compute_snc_span_advantage`).

    ``returns`` are the GRPO outcome returns (SNC contributes only to the
    policy-gradient advantage, not to a critic target — this estimator does
    not use a critic).
    """
    outcome_adv, outcome_returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=eos_mask,
        index=index,
        epsilon=epsilon,
    )
    mask = loss_mask if loss_mask is not None else eos_mask
    support_mask = (snc_token_rewards != 0) & mask.bool()
    snc_adv = compute_snc_span_advantage(
        snc_token_rewards=snc_token_rewards,
        support_mask=support_mask,
        index=index,
        epsilon=epsilon,
        clamp=snc_clamp,
        scale_floor=snc_scale_floor,
    )
    advantages = outcome_adv + snc_lambda * snc_adv
    if return_snc_advantage:
        return advantages, outcome_returns, snc_adv
    return advantages, outcome_returns


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange,
                        clip_ratio_c=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_c: (float or None)
            Dual-clip PPO ceiling (Ye et al., https://arxiv.org/abs/1912.09729).
            The vanilla PPO surrogate is unbounded for advantages < 0: with a huge
            importance ratio, max(-A*ratio, -A*clip(ratio)) selects the unclipped
            side, so a single off-policy token can dominate the whole update
            (observed as pg_loss spikes of 1e2-1e3 and grad_norm up to 1e7 that
            collapsed the intro menu 3B run around step 85). When set (> 1),
            the per-token loss for A < 0 is capped at c * |A|, which also zeroes
            the gradient through pathological ratios. None keeps old behavior.

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    # Clamp the log-ratio before exp: an extreme off-policy token (often a masked
    # state/tool token) would otherwise overflow to inf, and exp backward then
    # yields 0 * inf = NaN even though the forward loss stays finite.
    ratio = torch.exp(torch.clamp(negative_approx_kl, min=-20.0, max=20.0))
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_losses = torch.max(pg_losses, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, f"dual-clip ratio c must be > 1, got {clip_ratio_c}"
        pg_losses3 = -advantages * clip_ratio_c
        clipped_losses = torch.where(advantages < 0,
                                     torch.min(pg_losses3, clipped_losses),
                                     clipped_losses)

    pg_loss = verl_F.masked_mean(clipped_losses, eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        # Clamp before exp: once the policy drifts far from the reference, a single
        # token with (ref - logprob) > ~88 makes exp() overflow to inf; the output
        # clamp below only protects the forward value while the backward pass gets
        # 0 * inf = NaN and poisons the whole gradient.
        kl = torch.clamp(ref_logprob - logprob, min=-20.0, max=20.0)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
