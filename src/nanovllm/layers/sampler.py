import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        logits = logits.float()
        greedy_mask = temperatures <= 1e-10
        safe_temperatures = torch.where(
            greedy_mask,
            torch.ones_like(temperatures),
            temperatures,
        )
        logits = logits.div_(safe_temperatures.unsqueeze(dim=1))

        if torch.any(top_ks > 0):
            top_ks = torch.clamp(top_ks, max=logits.size(-1))
            kth_values = torch.gather(
                torch.topk(logits, k=int(top_ks.max().item()), dim=-1).values,
                1,
                (top_ks - 1).clamp_min(0).unsqueeze(1),
            )
            apply_top_k = top_ks.unsqueeze(1) > 0
            logits = torch.where(
                apply_top_k & (logits < kth_values),
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if torch.any(top_ps < 1.0):
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_ps.unsqueeze(1)
            sorted_mask[:, 0] = False
            keep_first_above = sorted_mask[:, :-1].clone()
            sorted_mask[:, 1:] = keep_first_above
            sorted_mask[:, 0] = False
            mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
            mask.scatter_(1, sorted_indices, sorted_mask)
            logits = torch.where(mask, torch.full_like(logits, float("-inf")), logits)

        if torch.any(greedy_mask):
            greedy_token_ids = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        if torch.any(greedy_mask):
            sample_tokens = torch.where(greedy_mask, greedy_token_ids, sample_tokens)
        return sample_tokens
