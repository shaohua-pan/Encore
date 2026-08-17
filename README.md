<div align="center">

# Encore (SIGGRAPH Asia 2026)

**Infinite Audio-Video Generation with Adaptive Signal Routing**

<!-- TODO(release): fill in the paper / project page / model links -->
[![Paper](https://img.shields.io/badge/Paper-arXiv-EC1C24?logo=arxiv&logoColor=white)](TODO_ARXIV_URL)
[![Project Page](https://img.shields.io/badge/Project-Page-181717?logo=googlechrome&logoColor=white)](https://shaohua-pan.github.io/encore-page/)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/panshaohua/Encore/tree/main)
[![Model](https://img.shields.io/badge/ModelScope-Model-blue)](https://www.modelscope.cn/models/panshaohua/Encore/files)

</div>

Encore generates minutes-long videos with synchronized audio, one ~5 s segment at
a time, from a single reference image and a text prompt. It is built on top of
[LTX-2.3](https://github.com/Lightricks/LTX-2).
## Setup

```bash
git clone TODO_REPO_URL Encore
cd Encore

uv sync
source .venv/bin/activate
```

Linux with a CUDA GPU is required (`triton` is Linux-only). Inference at
1920x1088 needs roughly 48 GB of VRAM; `ffmpeg` must be on `PATH`.

### Weights

Download these into one directory and point `MODEL_ROOT` at it.

From [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3):

- `ltx-2.3-22b-dev.safetensors` — base checkpoint
- `ltx-2.3-22b-distilled-lora-384.safetensors` — distilled LoRA, used by stage 2
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` — 2x spatial upscaler

The Gemma text encoder:

- `gemma-3-12b-it-qat-q4_0-unquantized` — from
  [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)

Ours:

- `encore_weight.safetensors` — the encore LoRA.
  <!-- TODO(release): publish and link the LoRA on HuggingFace / ModelScope -->
  **Not released yet** — see the badges above.

### Placeholders

Nothing in this repository points at a real path. Fill in your own:

```bash
cp scripts/env.sh.example scripts/env.sh
${EDITOR:-vi} scripts/env.sh
```

`scripts/env.sh` is git-ignored and is sourced by every script under `scripts/`.

- `MODEL_ROOT` — directory holding the weights listed above
- `DATA_ROOT` — raw clips, preprocessed latents, validation samples
- `OUTPUT_ROOT` — generated videos, checkpoints, logs
- `ENCORE_LORA` — path to `encore-routing-lora.safetensors`
- `CUDA_VISIBLE_DEVICES` — GPUs to use

The training config `packages/ltx-trainer/configs/ltx2_av_lora_encore.yaml` uses the
same placeholders written as `<MODEL_ROOT>`, `<DATA_ROOT>` and `<OUTPUT_ROOT>`;
replace them there too.

## Inference

### One segment from an image and a prompt

```bash
IMAGE=/path/to/portrait.png \
PROMPT='A woman looks into the camera and speaks. Speech: "Hello, this is a demo." Voice: a woman in her twenties, medium pace, warm and clear.' \
OUTPUT=$OUTPUT_ROOT/demo.mp4 \
    bash scripts/inference/inference_encore_i2v.sh
```

### Long-form generation

```bash
REF_IMAGE=/path/to/portrait.png \
REF_AUDIO=/path/to/voice.wav \
PROMPTS='["prompt for segment 1", "prompt for segment 2"]' \
    bash scripts/inference/inference_encore_long.sh
```

`PROMPTS` must be a valid JSON array of non-empty strings, with one prompt per
segment. The array length determines the number of segments. If `PROMPTS` is not
set, `PROMPT` is reused `NUM_SEGMENTS` times.

Writes `seg_0.mp4 ... seg_{N-1}.mp4` plus the concatenated `full.mp4`. The
optional reference audio pins the speaker's timbre across all segments. Omit
`REF_AUDIO` or set `REF_AUDIO=none` to use the first segment's generated audio
as the reference for the remaining segments.

#### Worked example: a sung performance across five segments

<p align="center">
  <img src="assets/example_singer.png" width="480" alt="Reference image for the long-form example">
</p>

[`examples/long_form_prompts.json`](examples/long_form_prompts.json) holds five
prompts that continue one performance: each prompt carries the next lines of the
song, the body motion, and the camera move. No reference audio is given, so the
singing voice generated in segment 1 becomes the reference for segments 2-5.

```bash
REF_IMAGE=assets/example_singer.png \
REF_AUDIO=none \
PROMPTS="$(cat examples/long_form_prompts.json)" \
    bash scripts/inference/inference_encore_long.sh
```

This produces `seg_0.mp4 ... seg_4.mp4` and `full.mp4` (~25 s total) at
1920x1088 in `$OUTPUT_ROOT/encore_long_example_singer_rscale_0.5`.

Prompt structure that works well for long-form: keep the subject description
stable across prompts, put the spoken or sung text in quotes, then describe body
motion and the camera move. Continuity across the cut is handled by the pipeline
(last frame plus audio tail), not by the prompt.

### Audio-driven video (A2V)

```bash
IMAGE=/path/to/portrait.png A2V_AUDIO=/path/to/song.mp3 \
    bash scripts/inference/inference_encore_a2v.sh
```

### Video-to-audio (V2A)

```bash
VIDEO=/path/to/silent.mp4 \
PROMPT='Two people playing a suona and an electric keyboard, the suona loud and bright over the keyboard.' \
    bash scripts/inference/inference_v2a_long.sh
```

## Training

### 1. Prepare data

Our training data cannot be released. `examples/` documents the exact schema and
ships a generator for a synthetic dataset you can use to smoke-test the pipeline:

```bash
python examples/make_example_dataset.py --num-clips 3
```

See [`examples/README.md`](examples/README.md) for the field-by-field schema
(`caption_all`, `frame_idx`, ...) and for how to build a real dataset from your
own videos.

### 2. Precompute latents and text embeddings

```bash
VIDEO_PREFIX_PATH=/path/to/clips NUM_GPUS=8 \
    bash scripts/preprocess/process_dataset.sh /path/to/dataset.jsonl
```

The JSONL is sharded across GPUs; each worker writes video latents, audio latents
and text embeddings into `$DATA_ROOT/precomputed/{latents,audio_latents,conditions}`.
Add `--decode` to a worker command to decode the latents back to mp4/wav and
eyeball them.

### 3. Train

Edit `packages/ltx-trainer/configs/ltx2_av_lora_encore.yaml` (model paths,
`preprocessed_data_root`, validation samples, `output_dir`), then:

```bash
bash scripts/train/train_lora_encore.sh                       # single node, 8 GPUs, FSDP

MACHINE_RANK=0 MASTER_ADDR=<rank-0-ip> NUM_NODES=3 \
    bash scripts/train/train_lora_encore_multinode.sh         # multi-node
```

## License and acknowledgements

This repository is a derivative work of
[Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) and is distributed under
the **LTX-2 Community License** — see [LICENSE](LICENSE). The LTX-2.3 weights and
the Gemma text encoder are covered by their own licences.

Verse-Bench is the work of the UniVerse-1 authors; our released prompt set is a
derivative of it and is not a substitute for the original benchmark.

## Citation

<!-- TODO(release): replace with the final bibtex -->
```bibtex
@inproceedings{encore,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

```bibtex
@article{ltx2,
  title   = {LTX-2: Efficient Joint Audio-Visual Foundation Model},
  author  = {HaCohen, Yoav and others},
  journal = {arXiv preprint arXiv:2601.03233},
  year    = {2026}
}
```
