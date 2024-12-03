# copied from 
# https://github.com/mosaicml/llm-foundry/blob/release/v0.3.0/llmfoundry/models/layers/ffn.py
# modified lines are marked # MOD

# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Any, Optional

import torch
import torch.nn as nn

# MOD
# original
# from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
# modified
from .fc import FC_CLASS_REGISTRY
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers import PretrainedConfig
# END MOD

try:
    import transformer_engine.pytorch as te
except:
    te = None


class MPTMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: int,
        fc_type: str = 'torch',
        device: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        fc_kwargs: dict[str, Any] = {
            'bias': bias,
        }
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.up_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            expansion_ratio * d_model,
            **fc_kwargs,
        )
        self.act = nn.GELU(approximate='none')
        self.down_proj = FC_CLASS_REGISTRY[fc_type](
            expansion_ratio * d_model,
            d_model,
            **fc_kwargs,
        )
        self.down_proj._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


# MOD
# original
# FFN_CLASS_REGISTRY = {
#     'mptmlp': MPTMLP,
# }
FFN_CLASS_REGISTRY = {
    'mptmlp': MPTMLP,
    'llamamlp': LlamaMLP
}

if te is not None:
    te.LayerNormMLP._has_norm = True
    FFN_CLASS_REGISTRY['te_ln_mlp'] = te.LayerNormMLP


def build_ffn(
    d_model: int,
    expansion_ratio: int,
    fc_type: str = 'torch',
    device: Optional[str] = None,
    bias: bool = True,
    **kwargs: Any,
) -> nn.Module:
    ffn_type = kwargs.pop('ffn_type')
    if ffn_type == 'mptmlp':
        if len(kwargs) > 0:
            raise ValueError(
                f'MPTMLP got an unexpected keyword argument: {kwargs}')
        return MPTMLP(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            device=device,
            bias=bias,
        )
        # MOD (original does not have the following lines
    elif ffn_type == 'llamamlp':
        llama_config ={
            "pretraining_tp": 1,
            "hidden_size": d_model,
            "intermediate_size": kwargs.get("intermediate_size", int(d_model * expansion_ratio)),
            "hidden_act": kwargs.get("hidden_act", "silu"),
            "mlp_bias" : kwargs.get('mlp_bias', False),
            }
        llama_config =PretrainedConfig.from_dict(llama_config)
        return LlamaMLP(llama_config)
        # MOD END
    elif ffn_type == 'te_ln_mlp':
        assert te is not None
        return te.LayerNormMLP(
            hidden_size=d_model,
            ffn_hidden_size=d_model * expansion_ratio,
            bias=bias,
            **kwargs,
        )

    raise ValueError(f'{ffn_type=} not recognized.')