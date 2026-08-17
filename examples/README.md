# Example data

The training data used in the paper cannot be released. This directory holds a
synthetic stand-in with **exactly the same schema**, so you can verify the
preprocessing and training path end to end before pointing the pipeline at your
own videos.

```bash
# writes examples/data/clips/*.mp4 and examples/data/example_dataset.jsonl
python examples/make_example_dataset.py --num-clips 3
```

The clips are ffmpeg test patterns with a sine tone, so a model trained on them
learns nothing useful — they only exercise the code paths.

## Dataset schema

One JSON object per line. `examples/data/example_dataset.jsonl` is a checked-in
sample of the real format.

- `video_path` (string) — path to the clip. The clip must carry an audio track
  for audio-video training. `--video-prefix-path` is stripped from this path to
  derive where the sample's latents are written, so the output mirrors your input
  directory tree.
- `audio_path` (string) — path to the audio. Usually the same file as
  `video_path`.
- `caption_video` (string) — visual-only description: subject, scene, and
  crucially the **motion**. The model is prone to producing near-static video, so
  describe continuous movement explicitly.
- `caption_audio` (string) — the transcript plus a description of the voice and
  background.
- `caption_audio_speech` (string) — transcript only.
- `caption_audio_bg` (string) — voice/background description only.
- `caption_all` (string) — the caption actually fed to the text encoder
  (`--caption-column caption_all`). For joint audio-video training this must
  contain the visual description, the transcript and the voice description in one
  string; the other caption fields are kept so you can ablate the caption
  composition without re-captioning.
- `frame_idx` (list of int) — which frames of the source clip form the training
  window. Its length must equal the frame count of the resolution bucket you
  preprocess with (e.g. 121 for `960x544x121`). The first index also defines the
  audio offset: `audio_start = frame_idx[0] / fps`. This is what lets you cut a
  fixed-length window out of a longer clip without re-encoding it.
- `fps` (float) — frame rate of the source clip.

Only `caption_all`, `video_path` and `frame_idx` are read by the default
preprocessing command; the remaining caption fields are metadata.

## Building your own dataset

1. Cut long videos into single-shot clips. `packages/ltx-trainer/scripts/split_scenes.py`
   does shot detection and splitting.
2. Caption them. `packages/ltx-trainer/scripts/caption_videos.py` produces visual
   captions; the audio captions in our data combine an ASR transcript with a
   voice-attribute description. Write the fields above into a JSONL file.
3. Precompute latents and text embeddings:

   ```bash
   VIDEO_PREFIX_PATH=/path/to/clips NUM_GPUS=8 \
       bash scripts/preprocess/process_dataset.sh /path/to/dataset.jsonl
   ```

4. Point `data.preprocessed_data_root` in
   `packages/ltx-trainer/configs/ltx2_av_lora_encore.yaml` at the output directory.

Constraints worth remembering: width and height must be divisible by 32, and the
frame count must satisfy `frames % 8 == 1` (121, 97, 89, ...).

## Validation samples

`validation.images` and `validation.audios` in the training config expect one
first-frame image and one reference audio clip per validation prompt. Use a few
held-out clips from your own data — the config ships placeholder paths under
`<DATA_ROOT>/val/`.
