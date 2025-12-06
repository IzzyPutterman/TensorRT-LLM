from typing import Dict, Optional

import torch
from torch import nn
from transformers import LlamaConfig

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig, load_weight_shard)
from ..modules.logits_processor import LogitsProcessor
from ..speculative import SpecMetadata
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_llama import LlamaModel
from .modeling_utils import DecoderModelForCausalLM, register_auto_model


import triton
import triton.language as tl

# Block size for Triton kernel - power of 2 for efficiency
_SILU_RESIDUAL_BLOCK_SIZE = 1024


@triton.jit
def _silu_residual_kernel(
    output_ptr,
    input_ptr,
    residual_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for fused SiLU activation + residual add.
    
    Computes: output = silu(input) + residual = input * sigmoid(input) + residual
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)

    # Fused SiLU + residual: x * sigmoid(x) + residual
    result = x * tl.sigmoid(x) + residual

    # Cast back to input dtype
    tl.store(output_ptr + offsets, result.to(output_ptr.dtype.element_ty), mask=mask)


def _silu_residual_impl(
    input: torch.Tensor,
    residual: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch the Triton kernel for fused silu + residual.
    
    Args:
        input: Input tensor after linear layer
        residual: Residual tensor to add
        output: Pre-allocated output tensor (can be same as input for in-place)
    """
    assert input.is_contiguous() and residual.is_contiguous()
    n_elements = input.numel()
    
    # Grid size - one program per block of elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _silu_residual_kernel[grid](
        output,
        input,
        residual,
        n_elements,
        BLOCK_SIZE=_SILU_RESIDUAL_BLOCK_SIZE,
    )


@torch.library.custom_op("trtllm::silu_residual", mutates_args=())
def silu_residual(input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Fused SiLU activation + residual add.
    
    Computes: output = silu(input) + residual
    
    This is registered as a custom op for compatibility with torch.compile
    and CUDA graphs.
    
    Args:
        input: Input tensor (typically output of a linear layer)
        residual: Residual tensor to add after activation
        
    Returns:
        output = input * sigmoid(input) + residual
    """
    output = torch.empty_like(input)
    _silu_residual_impl(input, residual, output)
    return output


@silu_residual.register_fake
def _silu_residual_fake(input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Fake tensor implementation for torch.compile tracing."""
    return torch.empty_like(input)


@torch.library.custom_op("trtllm::silu_residual_inplace", mutates_args=("input",))
def silu_residual_inplace(input: torch.Tensor, residual: torch.Tensor) -> None:
    """In-place fused SiLU activation + residual add.
    
    Computes: input = silu(input) + residual (in-place)
    
    This variant writes the result back to the input tensor, saving memory
    allocation overhead. Useful when the input tensor is no longer needed.
    
    Args:
        input: Input tensor, will be modified in-place with the result
        residual: Residual tensor to add after activation
    """
    _silu_residual_impl(input, residual, input)


@silu_residual_inplace.register_fake
def _silu_residual_inplace_fake(input: torch.Tensor, residual: torch.Tensor) -> None:
    """Fake tensor implementation for torch.compile tracing."""
    pass

class MedusaModel(nn.Module):
    """
    Medusa model that combines a base LLM with a single Medusa head.

    This model processes hidden states through the base model and then
    through a Medusa head for draft token generation.
    """

    def __init__(self, model_config: ModelConfig[LlamaConfig], head_idx: int):
        super().__init__()
        self.model_config = model_config
        self.head_idx = head_idx
        self.hidden_size = model_config.pretrained_config.hidden_size
        self.vocab_size = model_config.pretrained_config.vocab_size

        # Get the number of layers from eagle_config
        # (Medusa uses the same config structure for parallel draft heads)
        self.num_layers = model_config.pretrained_config.eagle_config.get(
            "parallel_draft_heads_num_layers", 0
        )
        assert self.num_layers > 0, "parallel_draft_heads_num_layers must be > 0"

        # Create the Medusa layers
        self.medusa_layers = nn.ModuleList(
            [Linear(self.hidden_size,
                         self.hidden_size,
                         bias=True,
                         dtype=model_config.torch_dtype) for _ in range(self.num_layers)]
        )
        self.act = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the Medusa head.

        Args:
            hidden_states: Hidden states from the base model

        Returns:
            Processed hidden states after Medusa head
        """
        for layer in self.medusa_layers:
            hidden_states_out = layer(hidden_states)
            hidden_states = self.act(hidden_states_out) + hidden_states
        return hidden_states


class FusedMedusaModel(nn.Module):
    """
    Fused Medusa model with Tensor Parallelism support.

    This module fuses all parallel draft heads into a single module using
    grouped 1x1 Conv1d layers. Each conv layer acts as a block-diagonal
    matrix multiply where each head (group) uses its own weights.

    Using Conv1d with groups=num_heads is more efficient than looping over
    heads because:
    1. Single kernel launch processes all heads in parallel
    2. Block-diagonal structure is handled natively by grouped convolution
    3. Better memory access patterns
    """

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        num_heads: int,
    ):
        super().__init__()
        self.model_config = model_config
        self.num_heads = num_heads
        self.hidden_size = model_config.pretrained_config.hidden_size
        self.vocab_size = model_config.pretrained_config.vocab_size
        self.mapping = model_config.mapping

        # Get the number of layers from eagle_config
        self.num_layers = model_config.pretrained_config.eagle_config.get(
            "parallel_draft_heads_num_layers", 0
        )
        assert self.num_layers > 0, "parallel_draft_heads_num_layers must be > 0"

        # ALL layers: Grouped 1x1 Conv1d
        # Each layer: Conv1d with groups=num_heads acts as block-diagonal Linear
        # Input channels: num_heads * hidden_size (grouped by head)
        # Output channels: num_heads * hidden_size (grouped by head)
        # Each group (head) has weights [hidden_size, hidden_size, 1]
        self.fused_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.num_heads * self.hidden_size,
                out_channels=self.num_heads * self.hidden_size,
                kernel_size=1,
                groups=self.num_heads,  # Each head uses its own weights
                bias=True,
                dtype=model_config.torch_dtype,
            )
            for _ in range(self.num_layers)
        ])

        self.act = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through all fused Medusa heads using grouped conv1d.

        Args:
            hidden_states: Hidden states from the base model [batch, hidden_size]

        Returns:
            Head states tensor of shape [batch, num_heads, hidden_size]
        """
        batch_size = hidden_states.shape[0]

        # Tile input for all heads: [batch, hidden_size] -> [batch, num_heads * hidden_size]
        # Then reshape for conv1d: [batch, num_heads * hidden_size, 1]
        head_states = hidden_states.unsqueeze(1).expand(-1, self.num_heads, -1)
        head_states = head_states.reshape(batch_size, self.num_heads * self.hidden_size, 1).contiguous()

        # All layers use grouped conv1d - single kernel processes all heads
        for layer in self.fused_layers:
            # head_states: [batch, num_heads * hidden_size, 1]
            # Conv1d with groups=num_heads applies block-diagonal matmul
            out = layer(head_states)  # [batch, num_heads * hidden_size, 1]
            # Fused activation and residual using custom Triton kernel
            # This is CUDA graph compatible via torch.library.custom_op registration
            head_states = silu_residual(out, head_states)

        # Reshape output: [batch, num_heads * hidden_size, 1] -> [batch, num_heads, hidden_size]
        head_states = head_states.view(batch_size, self.num_heads, self.hidden_size)

        return head_states


class FusedMedusaForCausalLM(DecoderModelForCausalLM[FusedMedusaModel, LlamaConfig]):
    """
    Fused Medusa model for causal language modeling.

    This class inherits from DecoderModelForCausalLM to get logits_processor
    and lm_head handling. It fuses all Medusa heads and their LM heads into
    single modules using grouped 1x1 Conv1d for efficient parallel processing.

    Each Medusa head has its own LM head, fused as:
    Conv1d(num_heads * hidden_size, num_heads * vocab_size, kernel_size=1, groups=num_heads)
    """

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        num_heads: int,
    ):
        self.num_heads = num_heads
        self._model_config = model_config
        self._hidden_size = model_config.pretrained_config.hidden_size
        self._vocab_size = model_config.pretrained_config.vocab_size

        # Initialize parent with FusedMedusaModel
        super().__init__(
            FusedMedusaModel(model_config, num_heads),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

        # Fused LM head using grouped 1x1 Conv1d
        # Each head has its own LM head: Linear(hidden_size, vocab_size)
        # Fused: Conv1d with groups=num_heads acts as block-diagonal projection
        self.fused_lm_head = nn.Conv1d(
            in_channels=num_heads * self._hidden_size,
            out_channels=num_heads * self._vocab_size,
            kernel_size=1,
            groups=num_heads,  # Each head uses its own weights
            bias=False,
            dtype=model_config.torch_dtype,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the fused Medusa model.

        Args:
            attn_metadata: Attention metadata for the forward pass
            input_ids: Input token IDs (unused, hidden_states expected)
            position_ids: Position IDs (unused)
            inputs_embeds: Pre-computed input embeddings (unused)
            return_context_logits: Whether to return context logits
            spec_metadata: Speculative decoding metadata
            hidden_states: Hidden states from the base model [batch, hidden_size]
            **kwargs: Additional keyword arguments

        Returns:
            Logits tensor of shape [batch, num_heads, vocab_size]
        """
        batch_size = hidden_states.shape[0]

        # Process through fused Medusa layers (returns [batch, num_heads, hidden_size])
        head_states = self.model(hidden_states)

        # Reshape for grouped conv1d: [batch, num_heads, hidden_size] -> [batch, num_heads * hidden_size, 1]
        head_states_conv = head_states.view(batch_size, self.num_heads * self._hidden_size, 1)

        # Apply fused LM head (grouped conv1d)
        # [batch, num_heads * hidden_size, 1] -> [batch, num_heads * vocab_size, 1]
        logits_conv = self.fused_lm_head(head_states_conv)

        # Reshape to [batch, num_heads, vocab_size]
        logits = logits_conv.view(batch_size, self.num_heads, self._vocab_size)

        return logits

    def load_weights(self, weights: Dict, weight_mapper: Optional[BaseWeightMapper] = None):
        """
        Load weights from checkpoint into fused Conv1d format.

        Expected checkpoint format for layers:
            parallel_draft_heads.{head_idx}.{layer_idx}.linear.weight
            parallel_draft_heads.{head_idx}.{layer_idx}.linear.bias

        Expected checkpoint format for LM heads:
            parallel_draft_heads.{head_idx}.lm_head.weight

        Args:
            weights: Dictionary mapping weight names to tensors
            weight_mapper: Optional weight mapper (unused for custom loading)
        """
        device = torch.device('cuda')
        num_layers = self.model.num_layers

        # Collect layer weights per layer across all heads
        layer_weights = [[] for _ in range(num_layers)]
        layer_biases = [[] for _ in range(num_layers)]
        lm_head_weights = []

        for head_idx in range(self.num_heads):
            # Load Medusa layer weights
            for layer_idx in range(num_layers):
                weight_key = f'parallel_draft_heads.{head_idx}.{layer_idx}.linear.weight'
                bias_key = f'parallel_draft_heads.{head_idx}.{layer_idx}.linear.bias'

                if weight_key in weights:
                    w = weights[weight_key]
                    if hasattr(w, 'get_shape'):  # safetensor slice
                        w = w[:].to(device)
                    else:
                        w = w.to(device)
                    layer_weights[layer_idx].append(w)

                if bias_key in weights:
                    b = weights[bias_key]
                    if hasattr(b, 'get_shape'):
                        b = b[:].to(device)
                    else:
                        b = b.to(device)
                    layer_biases[layer_idx].append(b)

            # Load LM head weights
            lm_head_key = f'parallel_draft_heads.{head_idx}.lm_head.weight'
            if lm_head_key in weights:
                w = weights[lm_head_key]
                if hasattr(w, 'get_shape'):
                    w = w[:].to(device)
                else:
                    w = w.to(device)
                lm_head_weights.append(w)

        # Load fused layer weights into Conv1d format
        # Conv1d weight shape: [out_channels, in_channels/groups, kernel_size]
        # For grouped conv: [num_heads * hidden_size, hidden_size, 1]
        for layer_idx, (weights_list, biases_list) in enumerate(
            zip(layer_weights, layer_biases)
        ):
            if weights_list:
                # Each head's weight: [hidden_size, hidden_size]
                # Stack and reshape for Conv1d: [num_heads * hidden_size, hidden_size, 1]
                fused_weight = torch.cat(weights_list, dim=0)  # [num_heads * H, H]
                fused_weight = fused_weight.unsqueeze(-1)  # [num_heads * H, H, 1]
                self.model.fused_layers[layer_idx].weight.data.copy_(fused_weight)

            if biases_list:
                # Conv1d bias shape: [out_channels] = [num_heads * hidden_size]
                fused_bias = torch.cat(biases_list, dim=0)
                self.model.fused_layers[layer_idx].bias.data.copy_(fused_bias)

        # Load fused LM head weights into Conv1d format
        # Each head's lm_head weight: [vocab_size, hidden_size]
        # Fused Conv1d weight: [num_heads * vocab_size, hidden_size, 1]
        if lm_head_weights:
            fused_lm_weight = torch.cat(lm_head_weights, dim=0)  # [num_heads * V, H]
            fused_lm_weight = fused_lm_weight.unsqueeze(-1)  # [num_heads * V, H, 1]
            self.fused_lm_head.weight.data.copy_(fused_lm_weight)


@register_auto_model("MedusaForCausalLM")
class MedusaForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):
    """
    Medusa model for causal language modeling with speculative decoding.

    This represents a single Medusa head with its own logits processor.
    Multiple instances of this class should be created for parallel draft heads.
    """

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        start_layer_idx: int = 0,
        head_idx: int = 0,
    ):
        # Initialize with the base model (shared or new)

        super().__init__(MedusaModel(model_config, start_layer_idx + head_idx),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

        self.head_idx = head_idx

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the Medusa model.

        Args:
            attn_metadata: Attention metadata for the forward pass
            input_ids: Input token IDs
            position_ids: Position IDs for positional encoding
            inputs_embeds: Pre-computed input embeddings (optional)
            return_context_logits: Whether to return context logits
            spec_metadata: Speculative decoding metadata
            hidden_states: Pre-computed hidden states (optional, for reusing base model output)
            **kwargs: Additional keyword arguments

        Returns:
            Logits from the language model head after Medusa head processing
        """

        # Process through Medusa head
        output = self.model(hidden_states)

        # Process through logits processor (each head has its own)
        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        """
        Load weights into the model.

        Args:
            weights: Dictionary mapping weight names to tensors
            weight_mapper: Weight mapper for handling weight name conversions
        """
        # Prepend "model." to weight keys if not already present
        # (except for lm_head and medusa_head)
        new_weights = {}
        for k, v in weights.items():
            if "lm_head" not in k and "medusa_head" not in k:
                new_k = "model." + k
            else:
                new_k = k
            new_weights[new_k] = v

        super().load_weights(weights=new_weights, weight_mapper=weight_mapper)
