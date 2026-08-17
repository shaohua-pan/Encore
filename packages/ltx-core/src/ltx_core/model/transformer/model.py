from dataclasses import replace
from enum import Enum
import warnings

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
        enable_condition_routing: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

        # Condition Routing Table: R[L, H, 4] → [anchor, continuation, semantic, sync]
        # Stored as logits and mapped to positive routing factors with 0.0 -> 1.0
        # so the initial behavior matches baseline exactly while allowing both
        # enhancement (>1) and suppression (<1) during training.
        self.enable_condition_routing = enable_condition_routing
        if enable_condition_routing:
            self.routing_table = torch.nn.Parameter(
                torch.zeros((num_layers, num_attention_heads, 4))
            )
        else:
            self.routing_table = None
        # Routing metadata (set externally before forward when routing is enabled)
        self._routing_metadata: dict | None = None

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""
        routing_weights = None
        base_video_self_attention_mask = video.self_attention_mask if video is not None else None
        if self.routing_table is not None and self._routing_metadata is not None:
            routing_weights = _normalize_routing_table(
                self.routing_table,
                num_layers=len(self.transformer_blocks),
                num_heads=self.num_attention_heads,
            )

        # Process transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # === Condition Routing ===
            semantic_scale = None
            sync_scale = None
            if routing_weights is not None:
                w = _routing_factors_from_logits(routing_weights[i])  # [H, 4]
                w_anchor = w[:, 0]  # [H]
                w_cont = w[:, 1]  # [H]
                semantic_scale = w[:, 2]  # [H]
                sync_scale = w[:, 3]  # [H]

                # A+B routing: construct per-head mask bias and add to self_attention_mask
                meta = self._routing_metadata
                if video is not None and base_video_self_attention_mask is not None:
                    video = replace(
                        video,
                        self_attention_mask=_apply_ab_routing(
                            base_video_self_attention_mask,
                            w_anchor,
                            w_cont,
                            meta["ref_token_count"],
                            meta["cond_token_count"],
                            meta.get("cont_pos_decay"),
                        ),
                    )

            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    semantic_scale,
                    sync_scale,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                    semantic_scale=semantic_scale,
                    sync_scale=sync_scale,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        # Auto-infer routing metadata during inference when not explicitly set
        if (
            self.enable_condition_routing
            and self.routing_table is not None
            and self._routing_metadata is None
            and video is not None
        ):
            self._routing_metadata = _infer_routing_metadata(video)

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        # When routing is enabled and no attention_mask was provided,
        # create a compact all-zeros mask so routing bias has a carrier to add to.
        # Shape (B, 1, 1, T): query dim is 1 because routing bias is row-uniform
        # (same for every query position).  This avoids allocating (B, 1, T, T).
        if (
            self.routing_table is not None
            and self._routing_metadata is not None
            and video_args is not None
            and video_args.self_attention_mask is None
        ):
            T = video_args.x.shape[1]
            video_args = replace(
                video_args,
                self_attention_mask=torch.zeros(
                    video_args.x.shape[0], 1, 1, T,
                    device=video_args.x.device, dtype=video_args.x.dtype,
                ),
            )

        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


def _normalize_routing_table(
    routing_table: torch.Tensor,
    *,
    num_layers: int,
    num_heads: int,
) -> torch.Tensor:
    """Normalize legacy routing table layouts to ``[L, H, 4]``."""
    target_shape = (num_layers, num_heads, 4)
    shape = tuple(routing_table.shape)

    if shape == target_shape:
        return routing_table

    normalized = _reshape_routing_table(routing_table, num_layers=num_layers, num_heads=num_heads)
    warnings.warn(
        (
            "Normalizing routing_table from shape "
            f"{shape} to {target_shape}. This usually means the checkpoint uses a legacy routing format."
        ),
        stacklevel=2,
    )
    return normalized


def _routing_factors_from_logits(routing_logits: torch.Tensor) -> torch.Tensor:
    """Map routing logits to positive baseline-centered factors in ``(0, 2)``.

    This parameterization keeps 0.0 as the identity point:
    - 0.0 -> 1.0 (baseline behavior)
    - >0  -> >1.0 (enhance)
    - <0  -> <1.0 (suppress)
    """
    return 2.0 * torch.sigmoid(routing_logits)


def _reshape_routing_table(routing_table: torch.Tensor, *, num_layers: int, num_heads: int) -> torch.Tensor:
    """Reshape or broadcast supported routing layouts to ``[L, H, 4]``."""
    if routing_table.ndim == 0:
        return routing_table.reshape(1, 1, 1).expand(num_layers, num_heads, 4)

    if routing_table.ndim == 1:
        size = routing_table.shape[0]
        if size == 4:
            return routing_table.reshape(1, 1, 4).expand(num_layers, num_heads, 4)
        if size == num_heads:
            return routing_table.reshape(1, num_heads, 1).expand(num_layers, num_heads, 4)
        if size == num_layers:
            return routing_table.reshape(num_layers, 1, 1).expand(num_layers, num_heads, 4)
        if size == num_heads * 4:
            return routing_table.reshape(1, num_heads, 4).expand(num_layers, num_heads, 4)
        if size == num_layers * 4:
            return routing_table.reshape(num_layers, 1, 4).expand(num_layers, num_heads, 4)
        if size == num_layers * num_heads:
            return routing_table.reshape(num_layers, num_heads, 1).expand(num_layers, num_heads, 4)
        if size == num_layers * num_heads * 4:
            return routing_table.reshape(num_layers, num_heads, 4)

    if routing_table.ndim == 2:
        layers, width = routing_table.shape
        if (layers, width) == (num_layers, 4):
            return routing_table.unsqueeze(1).expand(num_layers, num_heads, 4)
        if (layers, width) == (num_heads, 4):
            return routing_table.unsqueeze(0).expand(num_layers, num_heads, 4)
        if (layers, width) == (num_layers, num_heads):
            return routing_table.unsqueeze(-1).expand(num_layers, num_heads, 4)
        if (layers, width) == (4, num_heads):
            return routing_table.transpose(0, 1).unsqueeze(0).expand(num_layers, num_heads, 4)
        if (layers, width) == (1, 4):
            return routing_table.reshape(1, 1, 4).expand(num_layers, num_heads, 4)
        if (layers, width) == (1, num_heads):
            return routing_table.reshape(1, num_heads, 1).expand(num_layers, num_heads, 4)
        if (layers, width) == (num_layers, 1):
            return routing_table.reshape(num_layers, 1, 1).expand(num_layers, num_heads, 4)
        if (layers, width) == (num_layers * num_heads, 4):
            return routing_table.reshape(num_layers, num_heads, 4)

    if routing_table.ndim == 3:
        layers, heads, width = routing_table.shape
        if (layers, heads, width) == (num_layers, num_heads, 4):
            return routing_table
        if (layers, heads, width) == (1, num_heads, 4):
            return routing_table.expand(num_layers, num_heads, 4)
        if (layers, heads, width) == (num_layers, 1, 4):
            return routing_table.expand(num_layers, num_heads, 4)
        if (layers, heads, width) == (num_layers, num_heads, 1):
            return routing_table.expand(num_layers, num_heads, 4)

    raise ValueError(
        "Unsupported routing_table shape "
        f"{tuple(routing_table.shape)} for expected layout {(num_layers, num_heads, 4)}"
    )


def _apply_ab_routing(
    mask: torch.Tensor,
    w_anchor: torch.Tensor,
    w_cont: torch.Tensor,
    ref_token_count: int,
    cond_token_count: int,
    cont_pos_decay: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Anchor (A) and Continuation (B) per-head routing bias to self_attention_mask.

    The routing bias is row-uniform (independent of query position), so the
    returned mask is kept in compact ``(B, H, 1, T)`` form to avoid
    materializing a full ``(B, H, T, T)`` tensor.

    Args:
        mask: Self-attention mask — either ``(B, 1, T, T)`` (from
            ``_prepare_self_attention_mask``) or ``(B, H, 1, T)`` (output of a
            previous ``_apply_ab_routing`` call, which accumulates layer by
            layer).  The mask is additive log-space bias.
        w_anchor: Per-head anchor routing factors ``[H]``, values in (0, 2).
        w_cont: Per-head continuation routing factors ``[H]``, values in (0, 2).
        ref_token_count: Number of reference (anchor) tokens at the start of the sequence.
        cond_token_count: Number of conditioning tokens following ref tokens.
        cont_pos_decay: Optional log-space position decay ``[gen_len]`` for continuation routing.

    Returns:
        Compact mask of shape ``(B, H, 1, T)`` with per-head routing bias applied.
    """
    T = mask.shape[-1]
    H = w_anchor.shape[0]

    # Build per-head routing bias: [1, H, 1, T]
    routing_bias = torch.zeros(1, H, 1, T, device=mask.device, dtype=mask.dtype)

    # Anchor: bias on ref token columns (first ref_token_count columns)
    if ref_token_count > 0:
        routing_bias[:, :, :, :ref_token_count] = torch.log(w_anchor).view(1, H, 1, 1)

    # Continuation: bias on cond token columns
    cond_start = ref_token_count
    cond_end = ref_token_count + cond_token_count
    if cond_token_count > 0:
        routing_bias[:, :, :, cond_start:cond_end] = torch.log(w_cont).view(1, H, 1, 1)

    # Collapse the base mask to its first query-row to stay in compact form.
    # The base mask is row-uniform (all-zeros from training / inference), so
    # taking the first row is lossless.  On subsequent layers the mask is
    # already (B, H, 1, T) from the previous call, so this is a no-op slice.
    base_row = mask[:, :, :1, :]  # (B, *, 1, T)

    return base_row + routing_bias  # broadcasts to (B, H, 1, T)


def _infer_routing_metadata(video: Modality) -> dict:
    """Auto-infer routing metadata from video Modality during inference.

    Uses timesteps and positions to determine ref/cond token counts:
    - Ref (anchor) tokens: temporal position < 0 (shifted into negative time)
    - Cond (continuation) tokens: timestep == 0 AND temporal position >= 0
    - Gen tokens: timestep > 0

    Also builds cont_pos_decay with default lambda=0.3.
    """
    # positions: [B, ndim, T, 2], dim 0 = temporal, value [:, 0, :, 0] = temporal start
    temporal_start = video.positions[0, 0, :, 0]  # [T]
    ref_token_count = int((temporal_start < 0).sum().item())

    # Conditioning tokens have timestep == 0
    total_cond = int((video.timesteps[0] == 0).sum().item())
    cond_token_count = max(0, total_cond - ref_token_count)

    # Build log-space position decay for continuation routing (default λ=0.3)
    cont_pos_decay = None
    if cond_token_count > 0:
        T = video.timesteps.shape[1]
        gen_start = ref_token_count + cond_token_count
        gen_len = T - gen_start
        if gen_len > 0:
            lam = 0.3
            u = torch.arange(gen_len, device=video.timesteps.device, dtype=torch.float32) / gen_len
            cont_pos_decay = -u / lam

    return {
        "ref_token_count": ref_token_count,
        "cond_token_count": cond_token_count,
        "cont_pos_decay": cont_pos_decay,
    }


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
