# Copyright (c) 2026 The Scripps Research Institute
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def tokenize_ecg(ecg: torch.Tensor, token_dim: int = 500) -> torch.Tensor:
    """(B, 8, 5000) -> (B, 80, token_dim)"""
    B, leads, T = ecg.shape
    ecg = ecg.reshape(B, leads, T // token_dim, token_dim)
    ecg = ecg.reshape(B, leads * (T // token_dim), token_dim)
    return ecg


def random_obfuscate(tokenized_ecg: torch.Tensor, replace: float = 0.75):
    """
    Randomly mask `replace` fraction of tokens.
    Returns:
        obf_ecg:   (B, n_clean, token_dim) — visible tokens fed to encoder
        clean_idx: (B, n_clean)            — original positions of visible tokens
        obf_idx:   (B, n_obf)             — original positions of masked tokens
    """
    B, n_blocks, token_dim = tokenized_ecg.shape
    n_obf   = round(replace * n_blocks)
    n_clean = n_blocks - n_obf

    perm      = torch.stack([torch.randperm(n_blocks, device=tokenized_ecg.device) for _ in range(B)])
    clean_idx = perm[:, n_obf:].sort(dim=1).values
    obf_idx   = perm[:, :n_obf].sort(dim=1).values

    obf_ecg = tokenized_ecg.gather(1, clean_idx.unsqueeze(-1).expand(-1, -1, token_dim))
    return obf_ecg, clean_idx, obf_idx


def recon_loss(recon: torch.Tensor, tokenized_ecg: torch.Tensor, obf_idx: torch.Tensor) -> torch.Tensor:
    token_dim = tokenized_ecg.shape[-1]
    target = tokenized_ecg.gather(1, obf_idx.unsqueeze(-1).expand(-1, -1, token_dim))
    return F.l1_loss(recon, target)


def clip_loss(ecg_emb: torch.Tensor, text_emb: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    ecg_emb  = F.normalize(ecg_emb,  dim=1)
    text_emb = F.normalize(text_emb, dim=1)
    logits = ecg_emb @ text_emb.T * torch.clamp(torch.exp(log_scale), max=100.)
    labels = torch.arange(ecg_emb.shape[0], device=ecg_emb.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.


def make_optimizer_and_scheduler(model, lr, beta1, beta2, weight_decay, max_epochs, niter_per_epoch, warmup_epochs):
    warmup_iters = warmup_epochs * niter_per_epoch
    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    warmup    = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_iters)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs * niter_per_epoch - warmup_iters))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
    return optimizer, scheduler

