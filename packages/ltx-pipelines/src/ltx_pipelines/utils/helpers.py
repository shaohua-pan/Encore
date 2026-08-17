import gc
import logging
from dataclasses import replace

import torch
from tqdm import tqdm

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderFactory
from ltx_core.components.noisers import Noiser
from ltx_core.components.protocols import DiffusionStepProtocol, GuiderProtocol
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    AudioConditionByLatentIndex,
)
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer import Modality, X0Model
from ltx_core.model.video_vae import VideoEncoder
from ltx_core.text_encoders.gemma import GemmaTextEncoder
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.model.audio_vae import AudioEncoder
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.components.patchifiers import get_pixel_coords
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import decode_image, load_image_conditioning, load_audio_conditioning, resize_aspect_ratio_preserving
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.utils.types import (
    DenoisingFunc,
    DenoisingLoopFunc,
    PipelineComponents,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def encode_prompts(
    prompts: list[str],
    model_ledger: object,
    *,
    enhance_prompt_image: str | None = None,
    enhance_prompt_seed: int = 42,
    enhance_first_prompt: bool = False,
) -> list[EmbeddingsProcessorOutput]:
    """Encode prompts through Gemma → embeddings processor, freeing each after use.
    Loads the text encoder from *model_ledger*, optionally enhances the first
    prompt, encodes all *prompts*, frees the text encoder, then loads the
    embeddings processor to produce the final outputs.  Because the text encoder
    is loaded and freed entirely within this function, there are no lingering
    references that could prevent GPU memory reclamation.
    Args:
        prompts: Text prompts to encode.
        model_ledger: ModelLedger instance (used to load text encoder and embeddings processor).
        enhance_prompt_image: Optional image path for prompt enhancement.
        enhance_prompt_seed: Seed for prompt enhancement (default 42).
        enhance_first_prompt: If True, enhance ``prompts[0]`` before encoding.
    Returns:
        List of EmbeddingsProcessorOutput, one per prompt.
    """
    text_encoder = model_ledger.text_encoder()
    if enhance_first_prompt:
        prompts = list(prompts)
        prompts[0] = generate_enhanced_prompt(text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed)
    raw_outputs = [text_encoder.encode(p) for p in prompts]
    torch.cuda.synchronize()
    del text_encoder
    cleanup_memory()

    embeddings_processor = model_ledger.gemma_embeddings_processor()
    results: list[EmbeddingsProcessorOutput] = [
        embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs
    ]
    del embeddings_processor
    cleanup_memory()
    return results


def combined_image_conditionings(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    """Create a list of conditionings by replacing the latent at the first frame with the encoded image if present
    and using other encoded images as the keyframe conditionings."""
    conditionings = []
    for img in images:
        image = load_image_conditioning(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        if img.frame_idx == 0:
            conditioning = VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=0,
            )
        elif img.frame_idx == 1:
            conditioning = VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=1,
            )
        else:
            conditioning = VideoConditionByKeyframeIndex(
                keyframes=encoded_image,
                strength=img.strength,
                frame_idx=img.frame_idx,
            )
        conditionings.append(conditioning)
    return conditionings


def image_conditionings_by_replacing_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_conditioning(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=img.frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_conditioning(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=img.frame_idx, strength=img.strength)
        )
    return conditionings


def noise_video_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, VideoLatentTools]:
    """Initialize and noise a video latent state for the diffusion pipeline.
    Creates a video latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    video latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, output_shape.fps)
    video_state = create_noised_state(
        tools=video_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return video_state, video_tools


def noise_audio_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, AudioLatentTools]:
    """Initialize and noise an audio latent state for the diffusion pipeline.
    Creates an audio latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    audio latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
    audio_state = create_noised_state(
        tools=audio_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return audio_state, audio_tools


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    """
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Compute timesteps from a denoise mask and sigma value.
    Multiplies the denoise mask by sigma to produce timesteps for each position
    in the latent state. Areas where the mask is 0 will have zero timesteps.
    """
    return denoise_mask * sigma


def simple_denoising_func(
    video_context: torch.Tensor, audio_context: torch.Tensor, transformer: X0Model
) -> DenoisingFunc:
    def simple_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, video_context, sigma)
        pos_audio = modality_from_latent_state(audio_state, audio_context, sigma)

        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return denoised_video, denoised_audio

    return simple_denoising_step


def guider_denoising_func(
    guider: GuiderProtocol,
    v_context_p: torch.Tensor,
    v_context_n: torch.Tensor,
    a_context_p: torch.Tensor,
    a_context_n: torch.Tensor,
    transformer: X0Model,
) -> DenoisingFunc:
    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
        pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        if guider.enabled():
            neg_video = modality_from_latent_state(video_state, v_context_n, sigma)
            neg_audio = modality_from_latent_state(audio_state, a_context_n, sigma)

            neg_denoised_video, neg_denoised_audio = transformer(video=neg_video, audio=neg_audio, perturbations=None)

            denoised_video = denoised_video + guider.delta(denoised_video, neg_denoised_video)
            denoised_audio = denoised_audio + guider.delta(denoised_audio, neg_denoised_audio)

        return denoised_video, denoised_audio

    return guider_denoising_step


def multi_modal_guider_denoising_func(
    video_guider: MultiModalGuider,
    audio_guider: MultiModalGuider,
    v_context: torch.Tensor,
    a_context: torch.Tensor,
    transformer: X0Model,
    *,
    last_denoised_video: torch.Tensor | None = None,
    last_denoised_audio: torch.Tensor | None = None,
) -> DenoisingFunc:
    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal last_denoised_video, last_denoised_audio

        if video_guider.should_skip_step(step_index) and audio_guider.should_skip_step(step_index):
            return last_denoised_video, last_denoised_audio

        sigma = sigmas[step_index]
        pos_video_modality = modality_from_latent_state(
            video_state, v_context, sigma, enabled=not video_guider.should_skip_step(step_index)
        )
        pos_audio_modality = modality_from_latent_state(
            audio_state, a_context, sigma, enabled=not audio_guider.should_skip_step(step_index)
        )

        denoised_video, denoised_audio = transformer(
            video=pos_video_modality, audio=pos_audio_modality, perturbations=None
        )
        neg_denoised_video, neg_denoised_audio = 0.0, 0.0
        if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
            if video_guider.do_unconditional_generation() and video_guider.negative_context is None:
                raise ValueError("Negative context is required for unconditioned denoising")
            if audio_guider.do_unconditional_generation() and audio_guider.negative_context is None:
                raise ValueError("Negative context is required for unconditioned denoising")
            neg_video_modality = modality_from_latent_state(
                video_state,
                video_guider.negative_context
                if video_guider.negative_context is not None
                else pos_video_modality.context,
                sigma,
            )
            neg_audio_modality = modality_from_latent_state(
                audio_state,
                audio_guider.negative_context
                if audio_guider.negative_context is not None
                else pos_audio_modality.context,
                sigma,
            )

            neg_denoised_video, neg_denoised_audio = transformer(
                video=neg_video_modality, audio=neg_audio_modality, perturbations=None
            )

        ptb_denoised_video, ptb_denoised_audio = 0.0, 0.0
        if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
            perturbations = []
            if video_guider.do_perturbed_generation():
                perturbations.append(
                    Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
                )
            if audio_guider.do_perturbed_generation():
                perturbations.append(
                    Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
                )
            perturbation_config = PerturbationConfig(perturbations=perturbations)
            ptb_denoised_video, ptb_denoised_audio = transformer(
                video=pos_video_modality,
                audio=pos_audio_modality,
                perturbations=BatchedPerturbationConfig(perturbations=[perturbation_config]),
            )

        mod_denoised_video, mod_denoised_audio = 0.0, 0.0
        if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
            perturbations = [
                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ]
            perturbation_config = PerturbationConfig(perturbations=perturbations)
            mod_denoised_video, mod_denoised_audio = transformer(
                video=pos_video_modality,
                audio=pos_audio_modality,
                perturbations=BatchedPerturbationConfig(perturbations=[perturbation_config]),
            )

        if video_guider.should_skip_step(step_index):
            denoised_video = last_denoised_video
        else:
            denoised_video = video_guider.calculate(
                denoised_video, neg_denoised_video, ptb_denoised_video, mod_denoised_video
            )

        if audio_guider.should_skip_step(step_index):
            denoised_audio = last_denoised_audio
        else:
            denoised_audio = audio_guider.calculate(
                denoised_audio, neg_denoised_audio, ptb_denoised_audio, mod_denoised_audio
            )

        last_denoised_video = denoised_video
        last_denoised_audio = denoised_audio

        return denoised_video, denoised_audio

    return guider_denoising_step


def multi_modal_guider_factory_denoising_func(
    video_guider_factory: MultiModalGuiderFactory,
    audio_guider_factory: MultiModalGuiderFactory | None,
    v_context: torch.Tensor,
    a_context: torch.Tensor,
    transformer: X0Model,
) -> DenoisingFunc:
    """Resolve guiders per step via factory.build_from_sigma, then multi_modal_guider_denoising_func."""
    last_denoised_video: torch.Tensor | None = None
    last_denoised_audio: torch.Tensor | None = None
    sigma_vals_cached: list[float] | None = None

    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal last_denoised_video, last_denoised_audio, sigma_vals_cached
        if sigma_vals_cached is None:
            sigma_vals_cached = sigmas.detach().cpu().tolist()
        sigma_val = sigma_vals_cached[step_index]
        video_guider = video_guider_factory.build_from_sigma(sigma_val)
        audio_guider = (audio_guider_factory or video_guider_factory).build_from_sigma(sigma_val)
        denoise_fn = multi_modal_guider_denoising_func(
            video_guider,
            audio_guider,
            v_context,
            a_context,
            transformer,
            last_denoised_video=last_denoised_video,
            last_denoised_audio=last_denoised_audio,
        )
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_index)
        last_denoised_video, last_denoised_audio = denoised_video, denoised_audio
        return denoised_video, denoised_audio

    return guider_denoising_step


def denoise_audio_video(  # noqa: PLR0913
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: torch.Tensor,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
) -> tuple[LatentState, LatentState]:
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )
    audio_state, audio_tools = noise_audio_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_audio_latent,
    )

    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
    )

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


def denoise_video_only(  # noqa: PLR0913
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: torch.Tensor,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
) -> LatentState:
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )

    audio_state, _ = noise_audio_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=0.0,
        initial_latent=initial_audio_latent,
    )

    audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
    )

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)

    return video_state


def denoise_video_only_encore(  # noqa: PLR0913
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: torch.Tensor,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    audio_conditionings: list[ConditioningItem] = [],
    noise_scale: float = 1.0,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
    ref_audio_latent: torch.Tensor | None = None,
    ref_position_offset: float = 0.0,
) -> tuple[LatentState, LatentState]:
    fps = output_shape.fps

    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )

    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    total_latent_frames = video_latent_shape.frames
    target_latent_frames = total_latent_frames - 1

    ref_shape = video_latent_shape._replace(frames=1)
    vid_shape = video_latent_shape._replace(frames=target_latent_frames)

    ref_coords = components.video_patchifier.get_patch_grid_bounds(
        output_shape=ref_shape, device=device,
    )
    vid_coords = components.video_patchifier.get_patch_grid_bounds(
        output_shape=vid_shape, device=device,
    )

    ref_positions = get_pixel_coords(
        latent_coords=ref_coords,
        scale_factors=components.video_scale_factors,
        causal_fix=True,
    ).to(dtype)
    vid_positions = get_pixel_coords(
        latent_coords=vid_coords,
        scale_factors=components.video_scale_factors,
        causal_fix=True,
    ).to(dtype)

    ref_positions[:, 0, ...] /= fps
    vid_positions[:, 0, ...] /= fps
    vid_positions[:, 0, ...] += (1.0 / fps)

    if ref_position_offset > 0:
        ref_positions[:, 0, ...] -= ref_position_offset

    combined_video_positions = torch.cat([ref_positions, vid_positions], dim=2)
    combined_video_positions[:, 0, ...] -= (1.0 / fps)
    base_tokens = combined_video_positions.shape[2]
    video_state.positions[:, :, :base_tokens, :] = combined_video_positions

    output_shape_audio = output_shape._replace(frames=output_shape.frames - 8)
    audio_state, audio_tools = noise_audio_state(
        output_shape=output_shape_audio,
        noiser=noiser,
        conditionings=audio_conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=0.0,
        initial_latent=initial_audio_latent,
    )
    audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

    ref_audio_tokens = 0
    if ref_audio_latent is not None:
        ref_audio_patchified = components.audio_patchifier.patchify(ref_audio_latent)
        ref_audio_tokens = ref_audio_patchified.shape[1]

        ref_audio_mask = torch.zeros(
            ref_audio_patchified.shape[0], ref_audio_tokens, 1,
            device=device, dtype=torch.float32,
        )

        ref_audio_time_in_sec = components.audio_patchifier._get_audio_latent_time_in_sec(
            ref_audio_tokens, ref_audio_tokens + 1,
            dtype=audio_state.positions.dtype, device=device,
        ).item()

        target_audio_tokens = audio_state.positions.shape[2]
        combined_audio_len = ref_audio_tokens + target_audio_tokens
        combined_audio_shape = AudioLatentShape(
            batch=ref_audio_latent.shape[0],
            channels=ref_audio_latent.shape[1],
            frames=combined_audio_len,
            mel_bins=ref_audio_latent.shape[3],
        )
        combined_audio_positions = components.audio_patchifier.get_patch_grid_bounds(
            output_shape=combined_audio_shape, device=device,
        ).to(audio_state.positions.dtype)

        combined_audio_positions[:, 0, :, :] -= ref_audio_time_in_sec
        if ref_position_offset > 0:
            combined_audio_positions[:, 0, :ref_audio_tokens, :] -= ref_position_offset

        audio_state = replace(
            audio_state,
            latent=torch.cat([ref_audio_patchified, audio_state.latent], dim=1),
            denoise_mask=torch.cat([ref_audio_mask, audio_state.denoise_mask], dim=1),
            positions=combined_audio_positions,
            clean_latent=torch.cat([ref_audio_patchified, audio_state.clean_latent], dim=1),
        )

    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
    )

    if ref_audio_tokens > 0:
        audio_state = replace(
            audio_state,
            latent=audio_state.latent[:, ref_audio_tokens:],
            denoise_mask=audio_state.denoise_mask[:, ref_audio_tokens:],
            positions=audio_state.positions[:, :, ref_audio_tokens:],
            clean_latent=audio_state.clean_latent[:, ref_audio_tokens:],
        )

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


def denoise_audio_video_encore(  
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: torch.Tensor,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    audio_conditionings: list[ConditioningItem] = [],
    noise_scale: float = 1.0,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
    ref_audio_latent: torch.Tensor | None = None,
    ref_position_offset: float = 0.0,
) -> tuple[LatentState, LatentState]:
    fps = output_shape.fps

    # --- Video State ---
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )

    # ref_image: 1 latent frame (causal first = 1px)
    # video: N latent frames (own causal: frame0=1px, frame1+=8px)
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    total_latent_frames = video_latent_shape.frames
    target_latent_frames = total_latent_frames - 1

    ref_shape = video_latent_shape._replace(frames=1)
    vid_shape = video_latent_shape._replace(frames=target_latent_frames)

    ref_coords = components.video_patchifier.get_patch_grid_bounds(
        output_shape=ref_shape, device=device,
    )
    vid_coords = components.video_patchifier.get_patch_grid_bounds(
        output_shape=vid_shape, device=device,
    )

    ref_positions = get_pixel_coords(
        latent_coords=ref_coords,
        scale_factors=components.video_scale_factors,
        causal_fix=True,
    ).to(dtype)
    vid_positions = get_pixel_coords(
        latent_coords=vid_coords,
        scale_factors=components.video_scale_factors,
        causal_fix=True,
    ).to(dtype)

    # Convert temporal axis from pixel-frame index to seconds
    ref_positions[:, 0, ...] /= fps
    vid_positions[:, 0, ...] /= fps

    # Offset video by ref's pixel extent (1px = 1/fps sec)
    vid_positions[:, 0, ...] += (1.0 / fps)

    # Shift ref_positions further into negative time to isolate ref from video.
    if ref_position_offset > 0:
        ref_positions[:, 0, ...] -= ref_position_offset

    # Concatenate ref + video positions along seq_len (dim=2)
    combined_video_positions = torch.cat([ref_positions, vid_positions], dim=2)

    # Shift entire sequence so ref is at negative time
    combined_video_positions[:, 0, ...] -= (1.0 / fps)

    # Replace positions in video_state (in-place update of underlying tensor)
    video_state.positions[:] = combined_video_positions

    # --- Audio State ---
    # Audio covers the VIDEO portion only (not the ref_image placeholder).
    # output_shape.frames = num_frames + 8, so target video = num_frames = frames - 8
    output_shape_audio = output_shape._replace(frames=output_shape.frames - 8)
    audio_state, audio_tools = noise_audio_state(
        output_shape=output_shape_audio,
        noiser=noiser,
        conditionings=audio_conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_audio_latent,
    )

    # Prepend ref_audio to audio state (matching training structure)
    ref_audio_tokens = 0
    if ref_audio_latent is not None:
        ref_audio_patchified = components.audio_patchifier.patchify(ref_audio_latent)
        ref_audio_tokens = ref_audio_patchified.shape[1]

        # Ref audio is fully clean (denoise_mask = 0)
        ref_audio_mask = torch.zeros(
            ref_audio_patchified.shape[0], ref_audio_tokens, 1,
            device=device, dtype=torch.float32,
        )

        # Compute ref_audio time for position shift
        ref_audio_time_in_sec = components.audio_patchifier._get_audio_latent_time_in_sec(
            ref_audio_tokens, ref_audio_tokens + 1,
            dtype=audio_state.positions.dtype, device=device,
        ).item()

        # Generate UNIFIED positions for the combined (ref + target) sequence.
        # Training generates positions for the entire concatenated audio_seq_len,
        # NOT separately for ref and target. This matters because audio VAE is
        # causal: token 0 has a shorter time span than subsequent tokens. If we
        # generated target positions separately, target token 0 would get the
        # causal-first timing (0.01s) instead of the correct timing (0.04s).
        target_audio_tokens = audio_state.positions.shape[2]
        combined_audio_len = ref_audio_tokens + target_audio_tokens
        combined_audio_shape = AudioLatentShape(
            batch=ref_audio_latent.shape[0],
            channels=ref_audio_latent.shape[1],
            frames=combined_audio_len,
            mel_bins=ref_audio_latent.shape[3],
        )
        combined_audio_positions = components.audio_patchifier.get_patch_grid_bounds(
            output_shape=combined_audio_shape, device=device,
        ).to(audio_state.positions.dtype)

        # Shift entire audio sequence so ref_audio is at negative time
        combined_audio_positions[:, 0, :, :] -= ref_audio_time_in_sec
        # Additionally shift only ref audio tokens further into negative time
        if ref_position_offset > 0:
            combined_audio_positions[:, 0, :ref_audio_tokens, :] -= ref_position_offset

        # Prepend ref_audio to audio state
        audio_state = replace(
            audio_state,
            latent=torch.cat([ref_audio_patchified, audio_state.latent], dim=1),
            denoise_mask=torch.cat([ref_audio_mask, audio_state.denoise_mask], dim=1),
            positions=combined_audio_positions,
            clean_latent=torch.cat([ref_audio_patchified, audio_state.clean_latent], dim=1),
        )

    # --- Denoising Loop ---
    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
    )

    # --- Post-process: strip ref_audio before clear_conditioning/unpatchify ---
    if ref_audio_tokens > 0:
        audio_state = replace(
            audio_state,
            latent=audio_state.latent[:, ref_audio_tokens:],
            denoise_mask=audio_state.denoise_mask[:, ref_audio_tokens:],
            positions=audio_state.positions[:, :, ref_audio_tokens:],
            clean_latent=audio_state.clean_latent[:, ref_audio_tokens:],
        )

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoder,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    image = None
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info(f"Enhanced prompt: {prompt}")
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    """Assert that the resolution is divisible by the required divisor.
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    """
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )
