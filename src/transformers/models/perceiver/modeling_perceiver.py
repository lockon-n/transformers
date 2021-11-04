# coding=utf-8
# Copyright 2021 Deepmind and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Perceiver model. """


import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_perceiver import PerceiverConfig


ModalitySizeT = Mapping[str, int]
PreprocessorOutputT = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorT = Callable[..., PreprocessorOutputT]
PostprocessorT = Callable[..., Any]

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "deepmind/language-perceiver"
_CONFIG_FOR_DOC = "PerceiverConfig"
_TOKENIZER_FOR_DOC = "PerceiverTokenizer"

PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "deepmind/language-perceiver",
    # See all Perceiver models at https://huggingface.co/models?filter=perceiver
]


@dataclass
class PerceiverModelOutput(ModelOutput):
    """
    Base class for Perceiver base model's outputs, with potential hidden states, attentions and cross-attentions.

    Args:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_labels)`):
            Logits.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
    """

    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PerceiverDecoderOutput(ModelOutput):
    """
    Base class for Perceiver decoder outputs, with potential cross-attentions.

    Args:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_labels)`):
            Output of the basic decoder.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
    """

    logits: torch.FloatTensor = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PerceiverMaskedLMOutput(MaskedLMOutput):
    """
    Base class for Perceiver's masked language model outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, num_latents,
            num_latents)`. Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
    """

    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PerceiverClassifierOutput(SequenceClassifierOutput):
    """
    Base class for Perceiver's outputs of sequence/image classification models, optical flow and multimodal
    autoencoding.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
    """

    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class PerceiverEmbeddings(nn.Module):
    """Construct the latent embeddings."""

    def __init__(self, config):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(config.num_latents, config.d_latents))

    def forward(self, batch_size):
        embeddings = self.latents.expand(batch_size, -1, -1)  # Thanks, Phil Wang

        return embeddings


class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by" f" num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by" f" num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        # if inputs is not None:
        #     print("First few elements of queries before layernorm:", hidden_states[0,:3,:3])
        #     print("Sum of queries before layernorm:", hidden_states.sum())
        #     print("First few elements of keys + values before layernorm:", inputs[0,:3,:3])
        #     print("Sum of keys + values before layernorm:", inputs.sum())

        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        # Project queries, keys and values to a common feature dimension.
        # If this is instantiated as a cross-attention module, the keys
        # and values come from the inputs; the attention mask needs to be
        # such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        # if is_cross_attention:
        #     print("First few elements of queries after layernorm:", hidden_states[0, :3, :3])
        #     print("First few elements of keys + values after layernorm:", inputs[0, :3, :3])

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # if is_cross_attention:
        #     print("First few elements of queries:", queries[0, :3, :3])
        #     print("First few elements of keys:", keys[0, :3, :3])
        #     print("First few elements of values:", values[0, :3, :3])

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        # print("Shape of attention scores:", attention_scores.shape)
        # print("First few elements of attention scores:", attention_scores[0, :3, :3, :3])

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        # print("Shape of attention scores:", attention_scores.shape)
        # print("First few elements of attention scores after scaling:", attention_scores[0, :3, :3, :3])

        # if is_cross_attention:
        #     print("Inputs mask:", inputs_mask)
        #     print("Shape of attention mask:", attention_mask.shape)
        #     print("Sum of attention mask:", attention_mask.sum())

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PerceiverModel forward() function)
            attention_scores = attention_scores + attention_mask

        # print("Attention scores after applying mask:", attention_scores[0, :3, :3, :3])
        # print("Sum of attention scores after applying mask:", attention_scores.sum())

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # print("Attention probs after softmax:", attention_probs[0, :3, :3, :3])

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # print("Result:", context_layer[0, :3, :3])

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PerceiverSelfOutput(nn.Module):
    def __init__(self, config, input_channels, output_channels):
        super().__init__()
        self.dense = nn.Linear(input_channels, output_channels)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states


class PerceiverAttention(nn.Module):
    """Attention module, including a dense block."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
    ):
        super().__init__()
        # MultiHead attention
        if is_cross_attention and qk_channels is None:
            if config.cross_attention_shape_for_attention == "q":
                qk_channels = q_dim
            elif config.cross_attention_shape_for_attention == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {config.cross_attention_shape_for_attention} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.self = PerceiverSelfAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        # dense block
        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        else:
            if output_channels is None:
                output_channels = v_channels
        self.output = PerceiverSelfOutput(config, input_channels=self.self.v_channels, output_channels=output_channels)
        self.use_query_residual = use_query_residual
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        # Output projection
        attention_output = self.output(self_outputs[0])

        # print("Result after conv1d:", attention_output[0, :3, :3])

        # Optionally include a residual to the original queries.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        # print("Result after query residual:", attention_output[0, :3, :3])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, config, input_size, widening_factor):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(input_size, input_size)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerceiverAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # add attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        layer_output = layer_output + attention_output  # residual connection

        # print("Output after MLP:", layer_output[0, :3, :3])

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)

        # print("Output after layernorm before MLP:", layer_output[0, :3, :3])

        layer_output = self.mlp(layer_output)

        return layer_output


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Check that we can use multihead-attention with these shapes.
        if config.d_latents % config.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_self_attend_heads ({config.num_self_attention_heads})."
            )
        if config.d_latents % config.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({config.num_cross_attention_heads})."
            )

        # Construct the cross attention layer.
        self.cross_attention = PerceiverLayer(
            config,
            is_cross_attention=True,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            num_heads=config.num_cross_attention_heads,
            q_dim=config.d_latents,
            kv_dim=config.d_model,
            widening_factor=config.cross_attention_widening_factor,
            use_query_residual=config.use_query_residual,
        )

        # Construct a single block of self-attention layers.
        # We get deeper architectures by applying this block more than once.
        self_attends = []
        for _ in range(config.num_self_attends_per_block):
            layer = PerceiverLayer(
                config,
                is_cross_attention=False,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_self_attention_heads,
                q_dim=config.d_latents,
                kv_dim=config.d_latents,
                widening_factor=config.self_attention_widening_factor,
            )
            self_attends.append(layer)

        self.self_attends = nn.ModuleList(self_attends)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # print("Shape of latents before cross-attention:", hidden_states.shape)
        # print("First few elements of latents:", hidden_states[0, :3, :3])

        # print("Shape of inputs before cross-attention:", inputs.shape)
        # print("First few elements of inputs:", inputs[0,:3,:3])

        # Apply the cross-attention between the latents (hidden_states) and inputs:
        layer_outputs = self.cross_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # Apply the block of self-attention layers more than once:
        for _ in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                # print(f"Block {i} -----------------------------------------------")
                # print(f"Hidden states before block {i}:", hidden_states[0, :3, :3])
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PerceiverPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PerceiverConfig
    base_model_prefix = "perceiver"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


PERCEIVER_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.PerceiverConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

PERCEIVER_INPUTS_DOCSTRING = r"""
    Args:
        inputs
            ...
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    """The Perceiver: a scalable, fully attentional architecture.""",
    PERCEIVER_START_DOCSTRING,
)
class PerceiverModel(PerceiverPreTrainedModel):
    def __init__(self, config, decoder=None, input_preprocessor=None, output_postprocessor=None):
        super().__init__(config)
        self.config = config

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(config)
        self.decoder = decoder

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.latents

    def set_input_embeddings(self, value):
        self.embeddings.latents = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=PerceiverModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        inputs,
        attention_mask=None,
        subsampled_output_points=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.input_preprocessor(inputs)
        else:
            modality_sizes = None
            inputs_without_pos = None

        print("Shape of inputs before going into perceiver:", inputs.shape)
        # print("First elements of inputs:", inputs[0,:3,:3])

        if inputs.size()[-1] != self.config.d_model:
            raise ValueError(
                f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to config.d_model: {self.config.d_model}. "
                "Please update config.d_model appropriately."
            )
        else:
            input_shape = inputs.size()

        batch_size, seq_length, _ = input_shape
        device = inputs.device

        # If no attention mask is provided, make them all ones
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        # Make the attention mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = self.invert_attention_mask(attention_mask)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_blocks x num_heads]
        # and head_mask is converted to shape [num_blocks x batch x num_heads x N x N]
        head_mask = self.get_head_mask(head_mask, self.config.num_blocks * self.config.num_self_attends_per_block)

        embedding_output = self.embeddings(batch_size=batch_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        print("Shape of encoder outputs:", sequence_output.shape)
        # print("Encoder outputs:", sequence_output[0, :3, :3])

        # print("Modality sizes before postprocessing:", modality_sizes)

        logits = None
        if self.decoder:
            _, output_modality_sizes = self.decoder.output_shape(inputs)
            output_modality_sizes = output_modality_sizes or modality_sizes
            decoder_query = self.decoder.decoder_query(
                inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_output_points
            )
            decoder_outputs = self.decoder(
                decoder_query,
                z=sequence_output,
                query_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )
            logits = decoder_outputs.logits

            # add cross-attentions of decoder
            if output_attentions and decoder_outputs.cross_attentions is not None:
                if return_dict:
                    encoder_outputs.cross_attentions = (
                        encoder_outputs.cross_attentions + decoder_outputs.cross_attentions
                    )
                else:
                    encoder_outputs = encoder_outputs + decoder_outputs.cross_attentions

            print("Shape of decoder outputs:", logits.shape)

            if self.output_postprocessor:
                logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)

        if not return_dict:
            return (
                logits,
                sequence_output,
            ) + encoder_outputs[1:]

        return PerceiverModelOutput(
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings("""Example use of Perceiver for masked language modeling. """, PERCEIVER_START_DOCSTRING)
class PerceiverForMaskedLM(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverTextPreprocessor(config),
            decoder=PerceiverBasicDecoder(
                config,
                output_num_channels=config.d_latents,
                output_index_dims=config.max_position_embeddings,  # we need to define the seq_len of the inputs beforehand
                num_channels=config.d_model,
                qk_channels=8 * 32,
                v_channels=config.d_model,
                num_heads=8,
                use_query_residual=False,
                final_project=False,
                trainable_position_encoding_kwargs=dict(
                    num_channels=config.d_model, index_dims=config.max_position_embeddings
                ),
            ),
        )
        self.embedding_decoder = PerceiverEmbeddingDecoder(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=PerceiverMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.embedding_decoder(
            outputs.logits if return_dict else outputs[0], embedding_layer=self.perceiver.input_preprocessor.embeddings
        )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return PerceiverMaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# @add_start_docstrings("""Example use of Perceiver for image classification. """, PERCEIVER_START_DOCSTRING)
class PerceiverForImageClassification(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv1x1",
                spatial_downsample=1,
                out_channels=256,
                position_encoding_type="trainable",
                concat_or_add_pos="concat",
                project_pos_dim=256,
                trainable_position_encoding_kwargs=dict(
                    num_channels=256,
                    index_dims=config.image_size ** 2,
                ),
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
                use_query_residual=True,
            ),
        )

        self.init_weights()

    # @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# @add_start_docstrings("""Example use of Perceiver for image classification. """, PERCEIVER_START_DOCSTRING)
class PerceiverForImageClassificationFourier(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="pixels",
                spatial_downsample=1,
                fourier_position_encoding_kwargs=dict(
                    concat_pos=True, max_resolution=(224, 224), num_bands=64, sine_only=False
                ),
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
                use_query_residual=True,
            ),
        )

        self.init_weights()

    # @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# @add_start_docstrings("""Example use of Perceiver for image classification. """, PERCEIVER_START_DOCSTRING)
class PerceiverForImageClassificationConvProcessing(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv",
                spatial_downsample=1,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=dict(
                    concat_pos=True, max_resolution=(56, 56), num_bands=64, sine_only=False
                ),
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
                use_query_residual=True,
            ),
        )

        self.init_weights()

    # @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# @add_start_docstrings("""Example use of Perceiver for optical flow. """, PERCEIVER_START_DOCSTRING)
class PerceiverForOpticalFlow(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="patches",
                spatial_downsample=1,
                conv_after_patching=True,
                conv_after_patching_in_channels=54,
                temporal_downsample=2,
                position_encoding_type="fourier",
                # position_encoding_kwargs
                fourier_position_encoding_kwargs=dict(
                    num_bands=64,
                    max_resolution=config.train_size,
                    sine_only=False,
                    concat_pos=True,
                ),
            ),
            decoder=PerceiverFlowDecoder(
                config,
                num_channels=config.d_model,
                output_image_shape=config.train_size,
                rescale_factor=100.0,
                # decoder kwargs
                use_query_residual=False,
                output_num_channels=2,
                # We query the decoder using the first frame features
                # rather than a standard decoder position encoding.
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=dict(
                    concat_pos=True, max_resolution=config.train_size, num_bands=64, sine_only=False
                ),
            ),
        )

        self.init_weights()

    # @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            raise NotImplementedError("Optical flow training is not yet supported")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# @add_start_docstrings("""Example use of Perceiver for multimodal autoencoding. """, PERCEIVER_START_DOCSTRING)
class PerceiverForMultimodalAutoencoding(PerceiverPreTrainedModel):
    def __init__(self, config, subsampling):
        super().__init__(config)

        n_audio_samples = config.num_frames * config.audio_samples_per_frame

        subsampled_index_dims = {
            "audio": subsampling["audio"].shape[0],
            "image": subsampling["image"].shape[0],
            "label": 1,
        }

        input_preprocessor = PerceiverMultimodalPreprocessor(
            min_padding_size=4,
            modalities={
                "audio": PerceiverAudioPreprocessor(
                    config,
                    position_encoding_type="fourier",
                    fourier_position_encoding_kwargs=dict(
                        num_bands=192,
                        max_resolution=(n_audio_samples,),
                        sine_only=False,
                        concat_pos=True,
                    ),
                    prep_type="patches",
                    samples_per_patch=config.samples_per_patch,
                ),
                "image": PerceiverImagePreprocessor(
                    config,
                    position_encoding_type="fourier",
                    fourier_position_encoding_kwargs=dict(
                        num_bands=32,
                        max_resolution=(config.num_frames, config.image_size, config.image_size),
                        sine_only=False,
                        concat_pos=True,
                    ),
                    prep_type="patches",
                    spatial_downsample=4,
                    temporal_downsample=1,
                ),
                "label": PerceiverOneHotPreprocessor(config),
            },
            mask_probs={"image": 0.0, "audio": 0.0, "label": 1.0},
        )

        image_decoder = PerceiverBasicVideoAutoencodingDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            subsampled_index_dims=subsampling["image"],
            output_shape=config.output_shape,
            # num_z_channels=1024,
            output_num_channels=512,
            use_query_residual=False,
            position_encoding_only=True,
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs=dict(
                num_bands=32,
                max_resolution=(config.num_frames, config.image_size, config.image_size),
                sine_only=False,
                concat_pos=True,
            ),
        )

        decoder = PerceiverMultimodalDecoder(
            config,
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            subsampled_index_dims=subsampled_index_dims,
            # Modality specific decoders are used ONLY to generate queries.
            # All modalties are decoded together using a unified decoder.
            modalities={
                "audio": PerceiverBasicDecoder(
                    config,
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    subsampled_index_dims=subsampling["audio"],
                    output_index_dims=(n_audio_samples // config.samples_per_patch,),
                    # num_z_channels=1024,
                    output_num_channels=512,
                    use_query_residual=False,
                    position_encoding_only=True,
                    position_encoding_type="fourier",
                    fourier_position_encoding_kwargs=dict(
                        num_bands=192,
                        max_resolution=(n_audio_samples,),
                        sine_only=False,
                        concat_pos=True,
                    ),
                ),
                "image": image_decoder,
                "label": PerceiverClassificationDecoder(
                    config,
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    # num_z_channels=1024,
                    use_query_residual=False,
                    position_encoding_only=True,
                    position_encoding_type="trainable",
                    trainable_position_encoding_kwargs=dict(
                        num_channels=1024,
                        index_dims=1,
                    ),
                ),
            },
            num_outputs=None,
            output_num_channels=512,
            use_query_residual=False,
        )

        output_postprocessor = PerceiverMultimodalPostprocessor(
            modalities={
                "audio": PerceiverAudioPostprocessor(config, in_channels=512),
                "image": PerceiverProjectionPostprocessor(in_channels=512, out_channels=3),
                "label": PerceiverClassificationPostprocessor(config, in_channels=512),
            }
        )

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=input_preprocessor,
            decoder=decoder,
            output_postprocessor=output_postprocessor,
        )

        self.init_weights()

    # @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        inputs=None,
        attention_mask=None,
        subsampled_output_points=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            subsampled_output_points=subsampled_output_points,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            raise NotImplementedError("Multimodal autoencoding training is not yet supported")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# Below: position encodings


def build_position_encoding(
    position_encoding_type,
    out_channels=None,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
):
    """
    Builds the position encoding.

    Args:

    - out_channels: refers to the number of channels of the position encodings.
    - project_pos_dim: if specified, will project the position encodings to this dimension.

    """

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = PerceiverTrainablePositionEncoding(**trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # We don't use the index_dims argument, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = PerceiverFourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")

    # Optionally, project the position encoding to a target dimension:
    positions_projection = nn.Linear(out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()

    return output_pos_enc, positions_projection


# Below: Perceiver decoders


class PerceiverAbstractDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Perceiver abstract decoder."""

    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_query_channels(self):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, query, z, query_mask=None):
        raise NotImplementedError


class PerceiverProjectionDecoder(PerceiverAbstractDecoder):
    """Baseline projection decoder (no cross-attention)."""

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.d_latents, config.num_labels)

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return None

    def output_shape(self, inputs):
        return ((inputs.shape[0], self.num_labels), None)

    def forward(self, query, z, query_mask=None):
        # (batch_size, num_latents, d_latents) -> (batch_size, d_latents)
        z = torch.mean(z, dim=1)
        # (batch_size, d_latents) -> (batch_size, config.num_labels)
        logits = self.classifier(z)
        return logits


class PerceiverBasicDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention-based decoder.

    Here, `output_num_channels` refers to the number of output channels. `num_channels` refers to the number of
    channels of the output queries.

    """

    def __init__(
        self,
        config,
        output_num_channels,
        position_encoding_type="trainable",
        # The following 2 arguments are ignored if position_encoding_type == 'none':
        output_index_dims=None,
        num_channels=128,
        subsampled_index_dims=None,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        widening_factor=1,
        use_query_residual=False,
        concat_preprocessed_input=False,
        final_project=True,
        position_encoding_only=False,
        **position_encoding_kwargs,
    ):
        super().__init__()

        self.output_num_channels = output_num_channels
        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs
        if position_encoding_type != "none":
            self.output_position_encodings, self.positions_projection = build_position_encoding(
                position_encoding_type=position_encoding_type, **position_encoding_kwargs
            )

        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only

        # for multimodal autoencoding, we don't need the decoder cross-attention and final layer
        # so then we will set position_encoding_only to True
        if not self.position_encoding_only:
            self.decoding_cross_attention = PerceiverLayer(
                config,
                is_cross_attention=True,
                qk_channels=qk_channels,
                v_channels=v_channels,
                num_heads=num_heads,
                q_dim=num_channels,
                kv_dim=config.d_latents,
                widening_factor=widening_factor,
                use_query_residual=use_query_residual,
            )
            self.final_layer = nn.Linear(num_channels, output_num_channels) if final_project else nn.Identity()

    @property
    def num_query_channels(self) -> int:
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError(
                "You cannot calculate number of decoder query channels when position_encoding_type is set to none"
            )
        if self.position_encoding_only:
            if "project_pos_dim" in self.position_encoding_kwargs:
                return self.position_encoding_kwargs["project_pos_dim"]
            return self.output_position_encodings.output_size(pos_dim=1)
        if self.final_project:
            return self.output_num_channels
        return self.num_channels

    def output_shape(self, inputs):
        return ((inputs[0], self.subsampled_index_dims, self.output_num_channels), None)

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
        if subsampled_points is not None:
            # subsampled_points are the indices if the inputs would be flattened
            # however, the inputs aren't flattened, that's why we use unravel_index
            # to get the indices for the unflattened array
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            indices = list(torch.from_numpy(x) for x in np.unravel_index(subsampled_points, self.output_index_dims))
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, pos=pos
                )

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]

            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(index_dims, batch_size, device=inputs.device)

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)

        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
            pos_emb = torch.cat([inputs_without_pos, pos_emb], div=-1)

        print("Shape of decoder query:", pos_emb.shape)
        
        return pos_emb

    def forward(self, query, z, query_mask=None, output_attentions=False):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        # print("Shape of query before decoder cross attention:", query.shape)
        # print("First elements of query before decoder cross attention:", query[0,:3,:3])

        # print("Shape of z before decoder cross attention:", z.shape)
        # print("First elements of z before decoder cross attention:", z[0,:3,:3])

        cross_attentions = () if output_attentions else None

        layer_outputs = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
            output_attentions=output_attentions,
        )
        output = layer_outputs[0]

        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)

        # print("Shape of output after decoder cross attention:", output.shape)
        # print("First elements of output after decoder cross attention:", output[0,:3,:3])

        logits = self.final_layer(output)

        return PerceiverDecoderOutput(logits=logits, cross_attentions=cross_attentions)


class PerceiverClassificationDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention based classification decoder. Light-weight wrapper of `BasicDecoder` for logit output. Will turn
    the output of the Perceiver encoder which is of shape (batch_size, num_latents, d_latents) to a tensor of shape
    (batch_size, num_labels). The queries are of shape (batch_size, 1, num_labels).
    """

    def __init__(self, config, **decoder_kwargs):
        super().__init__()

        self.num_labels = config.num_labels
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=self.num_labels,
            output_index_dims=1,  # Predict a single logit array.
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points
        )

    def output_shape(self, inputs):
        return (inputs.shape[0], self.num_labels), None

    def forward(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        # B x 1 x num_classes -> B x num_classes
        logits = decoder_outputs.logits[:, 0, :]

        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)


class PerceiverFlowDecoder(PerceiverAbstractDecoder):
    """Cross-attention based flow decoder."""

    def __init__(self, config, output_image_shape, output_num_channels=2, rescale_factor=100.0, **decoder_kwargs):
        super().__init__()

        self.output_image_shape = output_image_shape
        self.output_num_channels = output_num_channels
        self.rescale_factor = rescale_factor
        self.decoder = PerceiverBasicDecoder(config, output_num_channels=output_num_channels, **decoder_kwargs)

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def output_shape(self, inputs):
        # The channel dimensions of output here don't necessarily correspond to
        # (u, v) of flow: they may contain dims needed for the post-processor.
        return ((inputs.shape[0],) + tuple(self.output_image_shape) + (self.output_num_channels,), None)

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")
        # assumes merged in time
        return inputs

    def forward(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        preds = decoder_outputs.logits
        # Output flow and rescale.
        preds /= self.rescale_factor
        preds = preds.reshape([preds.shape[0]] + list(self.output_image_shape) + [preds.shape[-1]])
        return PerceiverDecoderOutput(logits=preds, cross_attentions=decoder_outputs.cross_attentions)


class PerceiverBasicVideoAutoencodingDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention based video-autoencoding decoder. Light-weight wrapper of `BasicDecoder` with video reshaping
    logic.
    """

    def __init__(self, config, output_shape, position_encoding_type, **decoder_kwargs):
        super().__init__()
        if len(output_shape) != 4:  # B, T, H, W
            raise ValueError(f"Expected rank 4 output_shape, got {output_shape}.")
        # Build the decoder components:
        self.output_shape = output_shape
        self.output_num_channels = decoder_kwargs["output_num_channels"]

        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=self.output_shape[1:4],  # T*H*W
            position_encoding_type=position_encoding_type,
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self) -> int:
        print("Num query channels of image modality:", self.decoder.num_query_channels)
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(
            inputs,
            modality_sizes=modality_sizes,
            inputs_without_pos=inputs_without_pos,
            subsampled_points=subsampled_points,
        )

    def output_shape(self, inputs):
        return ([inputs.shape[0]] + self.output_shape[1:] + [self.output_num_channels], None)

    def forward(self, query, z, query_mask=None):
        decoder_outputs = self.decoder(query, z)
        logits = decoder_outputs.logits

        logits = torch.reshape(logits, self.output_shape + [logits.shape[-1]])
        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)


def restructure(modality_sizes, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """
    Partitions a [B, N, C] tensor into tensors for each modality.

    Args:
        modality_sizes: dict specifying the size of the modality
        inputs: input tensor

    Returns:
        dict mapping name of modality to its associated tensor.
    """
    outputs = {}
    index = 0
    # Apply a predictable ordering to the modalities
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        inp = inputs[:, index : index + size]
        index += size
        outputs[modality] = inp
    return outputs


class PerceiverMultimodalDecoder(PerceiverAbstractDecoder):
    """
    Multimodal decoding by composing uni-modal decoders. The modalities argument of the constructor is a dictionary
    mapping modality name to the decoder of that modality. That decoder will be used to construct queries for that
    modality. However, there is a shared cross attention across all modalities, using the concatenated per-modality
    query vectors.
    """

    def __init__(
        self,
        config,
        modalities,
        num_outputs,
        output_num_channels,
        min_padding_size=2,
        subsampled_index_dims=None,
        **decoder_kwargs
    ):
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.subsampled_index_dims = subsampled_index_dims
        self.min_padding_size = min_padding_size
        self.output_num_channels = output_num_channels
        self.num_outputs = num_outputs
        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=(num_outputs,),
            output_num_channels=output_num_channels,
            position_encoding_type="none",
            num_channels=self.num_query_channels,
            **decoder_kwargs,
        )
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, self.num_query_channels - decoder.num_query_channels))
                for modality, decoder in modalities.items()
            }
        )

    @property
    def num_query_channels(self) -> int:
        max_channel_size = max(decoder.num_query_channels for _, decoder in self.modalities.items())
        common_channel_size = max_channel_size + self.min_padding_size

        print("Max number of channels in decoder:", common_channel_size)
        return common_channel_size

    def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None):
        # Partition the flat inputs among the different modalities
        inputs = restructure(modality_sizes, inputs)

        # Obtain modality-specific decoders' queries
        subsampled_points = subsampled_points or dict()

        decoder_queries = dict()
        for modality, decoder in self.modalities.items():
            print("Creating query for modality:", modality)
            # Get input_without_pos for this modality if it exists.
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            decoder_queries[modality] = decoder.decoder_query(
                inputs=inputs[modality],
                modality_sizes=None,
                inputs_without_pos=input_without_pos,
                subsampled_points=subsampled_points.get(modality, None),
            )

        # Pad all queries with trainable position encodings to make them have the same channels

        def embed(modality, x):
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = self.padding[modality]
            batch_size = x.shape[0]
            pos = torch.broadcast_to(pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]])
            return torch.cat([x, pos], dim=2)

        # Apply a predictable ordering to the modalities
        return torch.cat(
            [embed(modality, decoder_queries[modality]) for modality in sorted(self.modalities.keys())], dim=1
        )

    def output_shape(self, inputs):
        if self.subsampled_index_dims is not None:
            subsampled_index_dims = sum(self.subsampled_index_dims.values())
        else:
            subsampled_index_dims = self.num_outputs
        return ((inputs.shape[0], subsampled_index_dims, self.output_num_channels), self.subsampled_index_dims)

    def forward(self, query, z, query_mask=None, output_attentions=False):
        # B x 1 x num_classes -> B x num_classes
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        return decoder_outputs


## Below: IO pre- and post-processor classes for Perceiver.


def space_to_depth(frames: torch.Tensor, temporal_block_size: int = 1, spatial_block_size: int = 1) -> torch.Tensor:
    """Space to depth transform, using einops."""
    try:
        import einops
    except ImportError:
        raise ImportError("Einops is not installed in your environment, which is required.")

    if len(frames.shape) == 4:
        return einops.rearrange(
            frames, "b c (h dh) (w dw) -> b h w (dh dw c)", dh=spatial_block_size, dw=spatial_block_size
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b (t dt) c (h dh) (w dw) -> b t h w (dt dh dw c)",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, channels, height, width)"
            " or rank 5 (batch, time, channels, height, width)"
        )


#  ------------------------------------------------------------
#  -------------------  Up/down-sampling  ---------------------
#  ------------------------------------------------------------


class Conv2dSamePadding(nn.Conv2d):
    """
    Conv2d layer with padding="same" support. Source:
    https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
    """

    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(
            reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        )

    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class Conv2DDownsample(nn.Module):
    """Downsamples 4x by applying a 2D convolution and doing max pooling."""

    def __init__(
        self,
        num_layers: int = 1,
        in_channels: int = 3,
        out_channels: int = 64,
        use_batchnorm: bool = True,
    ):
        """
        Constructs a Conv2DDownsample model.

        Args:
          in_channels: The number of input channels.
          out_channels: The number of conv output channels.
          use_batchnorm: Whether to use batchnorm.
        """
        super().__init__()

        self.conv = Conv2dSamePadding(
            in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, bias=False
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        return out


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    """
    Generate a Fourier frequency position encoding with linear spacing.

    Args:
      pos: The Tensor containing the position of n points in d dimensional space.
        A Torch tensor of shape [batch_size, n, d].
      num_bands: The number of bands (K) to use.
      max_resolution: The maximum resolution (i.e. the number of pixels per dim).
        A tuple representing resolution for each dimension.
      concat_pos: Whether to concatenate the input position encoding to the Fourier features.
      sine_only: Whether to use a single phase (sin) or two (sin/cos) for each frequency band.

    Returns:
      embedding: A Torch tensor of shape [batch_size, n, n_channels]. If concat_pos is True and sine_only is False,
      output dimensions are ordered as: [dim_1, dim_2, ..., dim_d, sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
      sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d),
      ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the kth frequency band.
    """

    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """
    Generate an array of position indices for an N-D input array.

    Args:
      index_dims: The shape of the index dimensions of the input array.
      output_range: The min and max values taken by each input index dimension.

    Returns:
      A Torch tensor of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
    """

    def _linspace(n_xels_per_dim):
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges)

    return torch.stack(array_index_grid, dim=-1)


class PerceiverAbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    """Perceiver abstract position encoding."""

    @property
    @abc.abstractmethod
    def num_dimensions(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def output_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch_size, pos):
        raise NotImplementedError


class PerceiverTrainablePositionEncoding(PerceiverAbstractPositionEncoding):
    """Trainable position encoding."""

    def __init__(self, index_dims, num_channels=128):
        super().__init__()
        self._num_channels = num_channels
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channels))

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        return self._num_channels

    def forward(self, batch_size):
        position_embeddings = self.position_embeddings

        if batch_size is not None:
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)
        return position_embeddings


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """
    Checks or builds spatial position features (x, y, ...).

    Args:
      pos: None, or an array of position features. If None, position features
        are built. Otherwise, their size is checked.
      index_dims: An iterable giving the spatial/index size of the data to be
        featurized.
      batch_size: The batch size of the data to be featurized.

    Returns:
      An array of position features, of shape [batch_size, prod(index_dims)].
    """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        # Just a warning label: you probably don't want your spatial features to
        # have a different spatial layout than your pos coordinate system.
        # But feel free to override if you think it'll work!
        assert pos.shape[-1] == len(index_dims)

    return pos


class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self, pos_dim: Optional[int] = None):
        """Returns size of positional encodings last dimension.

        Args:
            pos_dim: Size of the original position encoding. If None, will be equal to number of input dimensions.
                Defaults to None.
        """
        num_dims = len(self.max_resolution)
        print("Number of dims:", num_dims)
        encoding_size = self.num_bands * num_dims
        print("Encoding size:", encoding_size)
        if not self.sine_only:
            encoding_size *= 2
        print("Encoding size after sine_only:", encoding_size)
        if self.concat_pos:
            if pos_dim is None:
                print("we are here")
                encoding_size += self.concat_pos * num_dims
            else:
                print("pos_dim:", pos_dim)
                encoding_size += self.concat_pos * pos_dim
        print("Encoding size of Fourier embeddings:", encoding_size)
        
        return encoding_size

    def forward(self, index_dims, batch_size, device, pos=None):
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device)
        return fourier_pos_enc


class AbstractPreprocessor(nn.Module):
    @property
    def num_channels(self) -> int:
        """Returns size of preprocessor output."""
        raise NotImplementedError()


class PerceiverTextPreprocessor(AbstractPreprocessor):
    """Text preprocessing for Perceiver Encoder."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

    @property
    def num_channels(self) -> int:
        return self.config.d_model

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)

        seq_length = inputs.shape[1]
        position_ids = torch.arange(0, seq_length, device=inputs.device)
        embeddings = embeddings + self.position_embeddings(position_ids)

        return embeddings, None, None


class PerceiverEmbeddingDecoder(nn.Module):
    """Module to decode embeddings."""

    def __init__(self, config):
        """Constructs the module."""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))

    def forward(self, hidden_states, embedding_layer):
        batch_size, seq_len, d_model = hidden_states.shape
        output = torch.matmul(hidden_states.reshape([-1, d_model]), embedding_layer.weight.T)  # Flatten batch dim
        output = output + self.bias

        return output.reshape([batch_size, seq_len, self.vocab_size])


class PerceiverMultimodalPostprocessor(nn.Module):
    """Multimodal postprocessing for Perceiver."""

    def __init__(self, modalities: Mapping[str, PostprocessorT], input_is_dict: bool = False):
        """
        Constructor.

        Args:
          modalities: dict mapping modality name to post processor for that modality
          input_is_dict: If True, input is assumed to be dictionary structured,
            and outputs keep the same dictionary shape. If False, input is a tensor which is sliced up during
            postprocessing by `modality_sizes`.
          name: name of the module
        """
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.input_is_dict = input_is_dict

    def forward(
        self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None
    ) -> Mapping[str, torch.Tensor]:
        if not self.input_is_dict:
            # Slice up modalities by their sizes.
            assert modality_sizes is not None
            # print("Modality sizes:", modality_sizes)
            inputs = restructure(modality_sizes=modality_sizes, inputs=inputs)

        # print("Shape of inputs after restructure:")
        # for k,v in inputs.items():
        # print(k, v.shape)

        outputs = {
            modality: postprocessor(inputs[modality], pos=pos, modality_sizes=None)
            for modality, postprocessor in self.modalities.items()
        }
        return outputs


class PerceiverClassificationPostprocessor(nn.Module):
    """Classification postprocessing for Perceiver."""

    def __init__(self, config, in_channels):
        super().__init__()
        self.classifier = nn.Linear(in_channels, config.num_labels)

    def forward(self, inputs, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        logits = self.classifier(inputs)
        return logits[:, 0, :]


class PerceiverAudioPostprocessor(nn.Module):
    """Audio postprocessing for Perceiver."""

    def __init__(self, config, in_channels, postproc_type: str = "patches"):
        super().__init__()

        if postproc_type not in ("patches",):  # to be supported: 'conv', 'patches', 'pixels'
            raise ValueError("Invalid postproc_type!")

        # Architecture parameters:
        self.classifier = nn.Linear(in_channels, config.samples_per_patch)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        logits = self.classifier(inputs)
        return torch.reshape(logits, [inputs.shape[0], -1])


class PerceiverProjectionPostprocessor(nn.Module):
    """Projection postprocessing for Perceiver."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.classifier = nn.Linear(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        logits = self.classifier(inputs)
        return logits


class PerceiverImagePreprocessor(AbstractPreprocessor):
    """
    Image preprocessing for Perceiver Encoder.

    Note: the `out_channels` argument refers to the output channels of a convolutional layer, if `prep_type` is set to
    "conv1x1" or "conv". If one adds absolute position embeddings, one must make sure the `num_channels` of the
    position encoding kwargs are set equal to the `out_channels`.

    """

    def __init__(
        self,
        config,
        prep_type="conv",
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        position_encoding_type: str = "fourier",
        in_channels: int = 3,
        out_channels: int = 64,
        conv_after_patching: bool = False,
        conv_after_patching_in_channels: int = 54,  # only relevant when conv_after_patching = True
        conv2d_use_batchnorm: bool = True,
        concat_or_add_pos: str = "concat",
        project_pos_dim: int = -1,
        **position_encoding_kwargs,
    ):
        super().__init__()
        self.config = config

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError(f"Prep_type {prep_type} is invalid")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.in_channels = in_channels
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        if self.prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial " "and 1 expected for temporal " "downsampling with conv."
                )
            self.convnet = Conv2DDownsample(
                in_channels=in_channels,
                num_layers=int(convnet_num_layers),
                out_channels=out_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )

        elif self.prep_type == "conv1x1":
            if temporal_downsample != 1:
                raise ValueError("Conv1x1 does not downsample in time.")
            self.convnet_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(spatial_downsample, spatial_downsample),
            )

        # Position embeddings
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

        # Optional convolutional layer after patches.
        self.conv_after_patches = (
            nn.Linear(conv_after_patching_in_channels, self.out_channels) if conv_after_patching else nn.Identity()
        )

    @property
    def num_channels(self) -> int:
        # Let's assume that the number of resolutions (in the context of image preprocessing)
        # of the input data is 2 or 3 depending on whether we are processing image or video respectively.
        # In this case, for convenience, we will declare is_temporal variable,
        # which will show whether the data has a temporal dimension or not.
        is_temporal = self.position_embeddings.num_dimensions > 2

        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim

        # inputs
        if self.conv_after_patching or self.prep_type in ("conv1x1", "conv"):
            inp_dim = self.out_channels
        elif self.prep_type == "pixels":
            inp_dim = self.in_channels
            if not is_temporal:
                inp_dim = math.ceil(inp_dim / self.spatial_downsample)
        elif self.prep_type == "patches":
            if self.conv_after_patching:
                inp_dim = self.out_channels
            else:
                inp_dim = self.in_channels * self.spatial_downsample ** 2
                if is_temporal:
                    inp_dim *= self.temporal_downsample

        return inp_dim + pos_dim

    def _build_network_inputs(self, inputs: torch.Tensor, pos: torch.Tensor, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.

        """
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        print("Shape of pos encodings in image preprocessor:", pos_enc.shape)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])

        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        if self.prep_type == "conv":
            # Convnet image featurization.
            # Downsamples spatially by a factor of 4
            inputs = self.convnet(inputs)

        elif self.prep_type == "conv1x1":
            # print("Shape of inputs:", inputs.shape)

            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)

            # print("Shape of inputs after conv1x1:", inputs.shape)

        elif self.prep_type == "pixels":
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[:: self.spatial_downsample, :: self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[
                    :, :: self.temporal_downsample, :, :: self.spatial_downsample, :: self.spatial_downsample
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        elif self.prep_type == "patches":
            # Space2depth featurization.
            # Video: B x T x C x H x W
            inputs = space_to_depth(
                inputs, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample
            )

            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # for flow
                inputs = inputs.squeeze(dim=1)

            # Optionally apply conv layer.
            inputs = self.conv_after_patches(inputs)

        if self.prep_type != "patches":
            # move channels to last dimension, as the _build_network_inputs method below expects this
            if inputs.ndim == 4:
                inputs = torch.moveaxis(inputs, 1, -1)
            elif inputs.ndim == 5:
                inputs = torch.moveaxis(inputs, 2, -1)
            else:
                raise ValueError("Unsupported data format for conv1x1.")

        # print("Shape of inputs before _build_network_inputs:", inputs.shape)
        # print("First elements of inputs before _build_network_inputs:", inputs[0,:3,:3,:3])

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        modality_sizes = None  # Size for each modality, only needed for multimodal

        return inputs, modality_sizes, inputs_without_pos


class PerceiverOneHotPreprocessor(AbstractPreprocessor):
    """One-hot preprocessor for Perceiver Encoder."""

    def __init__(self, config):
        super().__init__()
        self.config: PerceiverConfig = config

    @property
    def num_channels(self) -> int:
        return self.config.num_labels

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # Add a dummy index dimension.
        inputs = inputs[:, None, :]

        # No position encodings, so the 1st (input) and 3rd (inputs_without_pos)
        # outputs are identical.
        return inputs, None, inputs


class PerceiverAudioPreprocessor(AbstractPreprocessor):
    """Audio preprocessing for Perceiver Encoder."""

    def __init__(
        self,
        config,
        prep_type: str = "patches",
        samples_per_patch: int = 96,
        position_encoding_type: str = "fourier",
        concat_or_add_pos: str = "concat",
        out_channels=64,
        project_pos_dim=-1,
        **position_encoding_kwargs,
    ):
        super().__init__()
        self.config = config

        if prep_type not in ("patches",):
            raise ValueError(f"Prep_type {prep_type} is invalid, can only be 'patches'.")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Concat_or_pos {concat_or_add_pos} is invalid, can only be 'concat' or 'add'.")

        self.samples_per_patch = samples_per_patch
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.project_pos_dim = project_pos_dim

        # Position embeddings
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

    @property
    def num_channels(self) -> int:
        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size(1)
        if self.concat_or_add_pos == "add":
            return pos_dim
        return self.samples_per_patch + pos_dim

    def _build_network_inputs(self, inputs, pos):
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)

        print("Shape of pos enc in audio preprocessor:", pos_enc.shape)

        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(self, inputs, pos, network_input_is_1d: bool = True):
        print("Shape of inputs:", inputs.shape)

        inputs = torch.reshape(inputs, [inputs.shape[0], -1, self.samples_per_patch])

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos)
        modality_sizes = None  # Size for each modality, only needed for multimodal

        print("Output of audio preprocessor:", inputs.shape)

        return inputs, modality_sizes, inputs_without_pos


class PerceiverMultimodalPreprocessor(AbstractPreprocessor):
    """
    Multimodal preprocessing for Perceiver Encoder.

    Inputs for each modality are preprocessed, then padded with trainable position embeddings to have the same number
    of channels.
    """

    def __init__(self, modalities, mask_probs=None, min_padding_size=2):
        """
        Constructor.

        Args:
            modalities: dict mapping modality name to preprocessor
            mask_probs: dict mapping modality name to masking probability of that
                modality
            min_padding_size: the minimum padding size for all modalities.
                The final output will have num_channels equal to the maximum channels across all modalities plus
                min_padding_size.
        """
        super().__init__()
        self.modalities = modalities
        self.min_padding_size = min_padding_size
        self.mask_probs = mask_probs if mask_probs is not None else dict()
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, self.num_channels - preprocessor.num_channels))
                for modality, preprocessor in modalities.items()
            }
        )
        self.mask = nn.ParameterDict(
            {modality: nn.Parameter(torch.randn(1, self.num_channels)) for modality, _ in self.mask_probs.items()}
        )

    @property
    def num_channels(self) -> int:
        max_channel_size = max(processor.num_channels for _, processor in self.modalities.items())
        common_channel_size = max_channel_size + self.min_padding_size
        print("Max number of channels:", common_channel_size)
        return common_channel_size

    def forward(
        self, inputs: Mapping[str, torch.Tensor], pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True
    ):

        for k, v in inputs.items():
            print(k, v.shape)

        padded = {}
        modality_sizes = {}
        inputs_without_pos = {}
        for modality, preprocessor in self.modalities.items():
            print(f"Preprocessing modality {modality}")

            # preprocess each modality using the respective preprocessor.
            output, _, inputs_without_pos[modality] = preprocessor(
                inputs[modality], pos=pos, network_input_is_1d=network_input_is_1d
            )

            print("Shape of output:", output.shape)

            # pad to the same common_channel_size.
            batch_size, num_samples, num_channels = output.shape
            pos_enc = self.padding[modality].expand(batch_size, -1, -1)

            print("Shape of pos_enc:", pos_enc.shape)

            padding = torch.broadcast_to(
                pos_enc,
                [batch_size, num_samples, self.num_channels - num_channels],
            )
            output_padded = torch.cat([output, padding], dim=2)

            # mask if required
            if modality in self.mask_probs:
                mask_token = self.mask[modality].expand(batch_size, -1, -1)
                mask_prob = self.mask_probs[modality]
                mask = torch.bernoulli(torch.full([batch_size, num_samples], mask_prob))
                mask = torch.unsqueeze(mask, dim=2)
                output_padded = (1 - mask) * output_padded + mask * mask_token

            padded[modality] = output_padded
            # print("Modality:", modality)
            # print("Shape of output_padded:", output_padded.shape)
            modality_sizes[modality] = output_padded.shape[1]

        # Apply a predictable ordering to the modalities
        padded_ls = [padded[k] for k in sorted(padded.keys())]

        # Finally, concatenate along the time dimension
        final_inputs = torch.cat(padded_ls, dim=1)

        return final_inputs, modality_sizes, inputs_without_pos
