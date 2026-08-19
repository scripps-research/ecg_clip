# Copyright (c) 2026 The Scripps Research Institute
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import math
import torch
import warnings

from torch import nn
from torchvision.ops import MLP


ECG_LEADS = ["L1", "L2", "V1", "V2", "V3", "V4", "V5", "V6"]


class PositionalEncoder(nn.Module):
    """Encode scalar inputs as sinusoidal frequency features."""
    def __init__(self, hidden_dim):
        super().__init__()
        k = torch.arange(0, hidden_dim // 2)
        L = torch.pi / (100. ** (2 * k / hidden_dim))
        L = L.unsqueeze(0).float()
        self.register_buffer('L', L)

    def forward(self, x):
        x = x @ self.L
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class TextEncoder(nn.Module):
    """Bio_ClinicalBERT with most layers frozen; last encoder layer + pooler unfrozen."""
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        from transformers import AutoModel, logging as hf_logging
        _prev = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        self.model = AutoModel.from_pretrained(model_name)
        hf_logging.set_verbosity(_prev)
        self.embedding_dim = self.model.pooler.dense.out_features
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.model.pooler.parameters():
            param.requires_grad = True

    def forward(self, tokens, attention_mask):
        return self.model(tokens, attention_mask=attention_mask).pooler_output


class ECGClip(nn.Module):
    """
    Transformer encoder-decoder for ECG patch tokens w/
    masked autoencoding and mean-pooled embeddings.
    Includes a text encoder and CLIP projection heads for contrastive pretraining.
    """
    def __init__(
        self,
        token_dim: int = 500,
        hidden_dim: int = 512,
        embedding_dim: int = 512,
        enc_depth: int = 12,
        dec_depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        contrastive_dim: int = 512,
        text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        transformer_norm=nn.LayerNorm,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.num_heads = num_heads
        self.dropout = dropout
        self.transformer_norm = transformer_norm(hidden_dim)

        self.linear_projection = nn.Linear(token_dim, hidden_dim)
        self.positional_encoder = PositionalEncoder(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, 4 * hidden_dim, dropout, batch_first=True
        )
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer, enc_depth, self.transformer_norm, enable_nested_tensor=False
        )
        self.encoder_FC = nn.Linear(hidden_dim, embedding_dim)

        self.decoder_FC = nn.Linear(embedding_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, 4 * hidden_dim, dropout, batch_first=True
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_layer, dec_depth, self.transformer_norm, enable_nested_tensor=False
        )

        self.reconstruction_layer = MLP(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim, token_dim],
            norm_layer=None,
            activation_layer=nn.GELU,
        )
        self.reconstruction_layer[-1].p = 0.0

        # CLIP components
        self.text_encoder = TextEncoder(text_model)
        self.ecg_FC  = nn.Linear(embedding_dim, contrastive_dim)
        self.text_FC = nn.Linear(self.text_encoder.embedding_dim, contrastive_dim)
        self.log_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        for name, module in self.named_children():
            if name != 'text_encoder':
                module.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, x, clean_token_idx=None):
        x = self.linear_projection(x)

        if clean_token_idx is None:
            padding_mask = None
            num_seq = x.shape[1]
            pos = torch.arange(num_seq).float().to(x.device)
            pos_embedding = self.positional_encoder(pos[:, None])
            x = x + pos_embedding[None]
        else:
            padding_mask = clean_token_idx == -1
            clean_token_idx = clean_token_idx[..., None].float()
            pos_embedding = self.positional_encoder(clean_token_idx)
            x = x + pos_embedding

        x = self.encoder_transformer(x, src_key_padding_mask=padding_mask)
        x = self.encoder_FC(x)

        embedding = x.mean(dim=1)
        return embedding

    def decode(self, x, obf_token_idx=None):
        x = self.decoder_FC(x)

        if obf_token_idx is None:
            padding_mask = None
            pos = torch.arange(80).float().to(x.device)
            pos_embedding = self.positional_encoder(pos[:, None])
            x = x[:, None] + pos_embedding[None]
        else:
            padding_mask = obf_token_idx == -1
            obf_token_idx = obf_token_idx[..., None].float()
            pos_embedding = self.positional_encoder(obf_token_idx)
            x = x[:, None] + pos_embedding

        x = self.decoder_transformer(x, src_key_padding_mask=padding_mask)
        return x

    def forward(self, x, clean_token_idx=None, obf_token_idx=None):
        embedding = self.encode(x, clean_token_idx=clean_token_idx)
        x = self.decode(embedding, obf_token_idx=obf_token_idx)
        reconstruction = self.reconstruction_layer(x)
        return embedding, reconstruction

    def forward_clip(self, obf_ecg, tokens, attention_mask,
                     clean_token_idx=None, obf_token_idx=None, decode=False):
        x = self.encode(obf_ecg, clean_token_idx)
        ecg_emb  = self.ecg_FC(x)
        text_emb = self.text_FC(self.text_encoder(tokens, attention_mask))
        if decode:
            decoded = self.decode(x, obf_token_idx=obf_token_idx)
            recon   = self.reconstruction_layer(decoded)
            return ecg_emb, text_emb, recon
        return ecg_emb, text_emb


class ECGClipFinetuning(nn.Module):
    """
    Wraps ECGClip encoder with a linear head for downstream
    classification or regression.
    """
    def __init__(self, model, output_dim):
        super().__init__()
        self.model = model
        self.linear_layer = nn.Linear(model.embedding_dim, output_dim)
        trunc_normal_(self.linear_layer.weight, std=0.02)
        nn.init.constant_(self.linear_layer.bias, 0)

    def forward(self, x):
        x = self.model.encode(x, None)
        x = self.linear_layer(x)
        return x
