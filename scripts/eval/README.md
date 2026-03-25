# Pilot Evaluation Framework

## Overview

This script evaluates whether existing IQA models and VLMs can capture **perceptual sensitivity** -- the non-linear, region-dependent gap between pixel-level change and perceived quality difference.

Given paired images (same seed, different generation parameters) with human annotations, it:

1. Scores each pair with **10 IQA metrics** (8 NR + 2 FR) via `pyiqa`
2. Prompts **VLMs** (GPT-5.4, GPT-4.1, Qwen3-VL-8B) with 3 evaluation strategies
3. Computes **SRCC, PLCC, bootstrap CI, rank agreement** against human scores
4. Outputs structured JSON + CSV for analysis

The core thesis: if current models score a background blur change similarly to a facial distortion, while humans consider the face far more critical, these models fundamentally lack perceptual sensitivity modeling.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[iqa]"

# 2. Prepare your data (see "Data Format" below)
cp data/pilot/manifest_example.json data/pilot/manifest.json
# Edit manifest.json: set real image paths and human scores

# 3. Run IQA-only evaluation (no API keys needed)
python scripts/eval/pilot_eval.py --config configs/eval/pilot.yaml --skip-vlm --device cpu

# 4. Check results
cat experiments/pilot_eval/summary.json
```

## Installation

### Core (always required)

```bash
pip install -e .
# Installs: torch, torchvision, omegaconf, scipy, pillow, tqdm, numpy
```

### IQA Models

```bash
pip install -e ".[iqa]"
# Installs: pyiqa (provides TOPIQ, NIQE, BRISQUE, MUSIQ, MANIQA, CLIP-IQA+, LIQE, Q-Align, LPIPS, DISTS)
```

Note: `qalign` requires ~15 GB GPU memory (fp16). The script auto-falls back to `qalign_8bit` (~8 GB) or `qalign_4bit` (~5 GB) on OOM.

### VLM API Clients

```bash
pip install -e ".[vlm]"
# Installs: openai, google-generativeai, anthropic
```

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."       # Required for GPT-5.4 / GPT-4.1
# export GOOGLE_API_KEY="..."        # Gemini (not implemented yet)
# export ANTHROPIC_API_KEY="..."     # Claude (not implemented yet)
```

### Qwen3-VL Local Inference

```bash
pip install "transformers>=4.57.0"
# Model weights (~16 GB) downloaded automatically on first run from:
# https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
```

Requires a CUDA GPU with sufficient VRAM (~18 GB for 8B model in auto dtype).

## Data Format

### Manifest JSON Schema

Create a JSON file (default path: `data/pilot/manifest.json`):

```json
{
  "dataset_info": {
    "name": "string (required)",
    "description": "string (optional)"
  },
  "pairs": [
    {
      "pair_id": "string (required) -- unique identifier",
      "prompt": "string (required) -- the generation prompt",
      "seed": "int (optional) -- random seed used",
      "reference": "string (required) -- path to reference image (best quality variant)",
      "variant": "string (required) -- path to variant image (degraded quality)",
      "variant_type": "string (optional) -- e.g. 'cfg_scale', 'var_early_exit'",
      "variant_params": "object (optional) -- generation parameter details",
      "human_scores": {
        "reference_mos": "float (optional) -- human MOS for reference (0-100)",
        "variant_mos": "float (optional) -- human MOS for variant (0-100)",
        "regions": {
          "<region_name>": {
            "ref": "float -- region score for reference",
            "var": "float -- region score for variant"
          }
        }
      }
    }
  ]
}
```

**Image paths**: Relative paths resolve against the manifest file's parent directory. Absolute paths also work. All paths are validated at startup -- missing images cause an immediate error.

**`human_scores`**: Optional per pair. If present, SRCC/PLCC/rank agreement are computed. If absent, only raw model scores are collected.

**`reference` vs `variant`**: The reference should be the higher-quality image (e.g., default/optimal CFG scale). FR metrics compute `distance(variant, reference)`.

### Directory Layout Example

```
data/pilot/
  manifest.json
  images/
    001_cfg7.5.png     # reference
    001_cfg2.0.png     # variant
    002_cfg7.5.png
    002_cfg3.0.png
    ...
```

See `data/pilot/manifest_example.json` for a complete example with 3 pairs.

## Configuration

The config file is at `configs/eval/pilot.yaml`. Key sections:

### `experiment`

```yaml
experiment:
  name: "pilot_v1"              # Experiment name (logged)
  output_dir: "experiments/pilot_eval"  # Where results are saved
```

### `data`

```yaml
data:
  manifest: "data/pilot/manifest.json"  # Path to manifest
  max_image_size: null   # null = original size
                          # Set to e.g. 1024 to cap the long edge (reduces GPU memory)
```

### `models.iqa`

```yaml
models:
  iqa:
    enabled: true        # Set false or use --skip-iqa to skip
    device: "cuda"       # "cpu" for local debug (slow but works)
    nr_models: [...]     # List of pyiqa NR metric names
    fr_models: [...]     # List of pyiqa FR metric names
```

### `models.vlm`

```yaml
  vlm:
    enabled: true        # Set false or use --skip-vlm to skip
    models:
      - name: "gpt-5.4"           # Display name in results
        provider: "openai"         # Provider type
        model_id: "gpt-5.4"       # API model ID
        api_key_env: "OPENAI_API_KEY"  # Environment variable name
        max_tokens: 1500           # Max response tokens
        temperature: 0.1           # Low temperature for consistent scoring
    prompts:
      - "holistic"       # Independent per-image rating
      - "comparative"    # Paired comparison
      - "sensitivity"    # Sensitivity probing
    rate_limit_delay: 2.0   # Seconds between API calls
    max_retries: 3          # Retry count for API errors
    retry_delay: 10.0       # Base delay for exponential backoff (10, 20, 40s)
```

## Running Evaluations

### IQA-only (no API keys, no GPU required)

```bash
python scripts/eval/pilot_eval.py \
  --config configs/eval/pilot.yaml \
  --skip-vlm \
  --device cpu
```

### VLM-only (requires API keys)

```bash
export OPENAI_API_KEY="sk-..."
python scripts/eval/pilot_eval.py \
  --config configs/eval/pilot.yaml \
  --skip-iqa
```

### Specific models only

```bash
# Only run TOPIQ and NIQE from IQA, plus GPT-5.4 from VLM
python scripts/eval/pilot_eval.py \
  --config configs/eval/pilot.yaml \
  --models topiq_nr niqe gpt-5.4
```

The `--models` filter applies to both IQA and VLM model names.

### Debug mode (few pairs, CPU)

```bash
python scripts/eval/pilot_eval.py \
  --config configs/eval/pilot.yaml \
  --max-pairs 2 \
  --device cpu \
  --skip-vlm
```

### Resuming interrupted VLM runs

VLM results are checkpointed to `experiments/pilot_eval/vlm_partial.json` after each pair. If the script crashes mid-run:

```bash
# Just re-run the same command -- completed pairs are skipped automatically
python scripts/eval/pilot_eval.py --config configs/eval/pilot.yaml --skip-iqa
```

To force a clean re-run, delete the checkpoint:

```bash
rm experiments/pilot_eval/vlm_partial.json
```

### Custom output directory

```bash
python scripts/eval/pilot_eval.py \
  --config configs/eval/pilot.yaml \
  --output-dir experiments/pilot_v2
```

## CLI Reference

```
python scripts/eval/pilot_eval.py --config CONFIG [OPTIONS]

Required:
  --config PATH          Path to YAML config file

Options:
  --manifest PATH        Override manifest path from config
  --models NAME [NAME]   Run only these models (matches IQA + VLM names)
  --skip-iqa             Skip all IQA model evaluation
  --skip-vlm             Skip all VLM evaluation
  --max-pairs N          Limit to first N pairs (for debugging)
  --device DEVICE        Override device: "cuda" or "cpu"
  --output-dir PATH      Override output directory
```

## Models

### IQA Models

| Category | Model Name | pyiqa ID | What It Measures | Score Direction |
|----------|-----------|----------|-----------------|----------------|
| Handcrafted | NIQE | `niqe` | Natural image statistics deviation | Lower = better |
| Handcrafted | BRISQUE | `brisque` | Spatial domain natural scene statistics | Lower = better |
| Deep NR-IQA | TOPIQ | `topiq_nr` | Top-down semantic-to-distortion quality | Higher = better |
| Deep NR-IQA | MUSIQ | `musiq` | Multi-scale image quality (transformer) | Higher = better |
| ViT-based | MANIQA | `maniqa` | Patch-weighted dual-branch attention | Higher = better |
| CLIP-guided | CLIP-IQA+ | `clipiqa+` | CLIP-based quality with learned prompts | Higher = better |
| CLIP-guided | LIQE | `liqe` | CLIP multi-task (scene+distortion+quality) | Higher = better |
| MLLM-based | Q-Align | `qalign` | mPLUG-Owl2 fine-tuned for IQA | Higher = better |
| FR perceptual | LPIPS | `lpips` | Learned perceptual distance (VGG) | Lower = better |
| FR perceptual | DISTS | `dists` | Structure + texture similarity | Lower = better |

**Why these models?** They span 5 distinct categories of IQA approaches. If our hypothesis is correct, ALL categories should struggle with perceptual sensitivity -- even Q-Align (an MLLM specifically trained for IQA) and LIQE (which uses language guidance). This strengthens the argument that the gap is fundamental, not a matter of model architecture.

**Score direction handling**: The script stores raw scores with a `lower_better` flag. When computing correlations and rank agreement, score direction is normalized automatically.

### VLM Models

| Provider | Model | Status | API Key Env Var |
|----------|-------|--------|----------------|
| OpenAI | GPT-5.4 | Implemented | `OPENAI_API_KEY` |
| OpenAI | GPT-4.1 | Implemented | `OPENAI_API_KEY` |
| Local | Qwen3-VL-8B | Implemented | (none, local GPU) |
| Google | Gemini 2.5 Flash | Stub (not implemented) | `GOOGLE_API_KEY` |
| Anthropic | Claude Sonnet 4.5 | Stub (not implemented) | `ANTHROPIC_API_KEY` |

Missing API keys or unimplemented providers are skipped with a warning -- the script continues with available models.

## Prompt Design

Each VLM is tested with 3 prompts that probe increasingly specific aspects of perceptual sensitivity:

### Prompt A: Holistic (`holistic`)

**What it tests**: Basic quality perception.

Each image is scored **independently** (reference and variant are shown in separate API calls). This avoids comparison bias and tests whether the model's absolute quality rating correlates with human MOS.

**Output fields**: `overall_score` (0-100), `key_observations`, `worst_region`, `worst_region_issue`

### Prompt B: Comparative (`comparative`)

**What it tests**: Region-aware quality comparison.

Both images are shown together. The model rates each, identifies regions with quality differences, and assigns a perceptual significance score (1-5) per region.

**Output fields**: `image_a_score`, `image_b_score`, `better_image`, `region_differences` (with `perceptual_significance`), `overall_explanation`

### Prompt C: Sensitivity (`sensitivity`)

**What it tests**: The core hypothesis -- can the model distinguish between pixel-level change magnitude and perceptual impact?

The model is explicitly asked to separately estimate `pixel_magnitude` (small/medium/large) and `perceptual_impact` (minimal/moderate/significant/severe) for each region, then explain the gap.

**Output fields**: `regions_analyzed` (with `pixel_magnitude` + `perceptual_impact` + `explanation`), `most_impactful_change`, `least_impactful_change`, `sensitivity_reasoning`

### Position Bias Mitigation

For Prompts B and C, the order of reference/variant as "Image A"/"Image B" is **randomized per pair** to counter VLM position bias. The actual order is recorded in each result's `image_order` field:

```json
{
  "image_order": {
    "A": "/path/to/001_cfg7.5.png",
    "B": "/path/to/001_cfg2.0.png",
    "A_is": "reference",
    "B_is": "variant"
  }
}
```

### Image Encoding

Images are sent as **lossless PNG** (not JPEG) to preserve subtle pixel-level differences -- since the entire experiment is about detecting small quality variations, lossy compression could mask the signal.

## Output Files

All outputs are saved to `experiment.output_dir` (default: `experiments/pilot_eval/`).

### `results.json`

Complete structured results including experiment config, all IQA scores, all VLM responses, and summary statistics. This is the single source of truth.

### `summary.json`

Aggregate statistics only:

```json
{
  "iqa": {
    "topiq_nr": {
      "srcc": 0.7234,
      "srcc_p": 0.0012,
      "srcc_ci_95": [0.5123, 0.8901],
      "plcc": 0.6891,
      "plcc_p": 0.0023,
      "plcc_ci_95": [0.4567, 0.8456],
      "rank_agreement": 0.7500,
      "n_pairs": 20
    }
  },
  "vlm": {
    "vlm_gpt-5.4_holistic": {
      "srcc": 0.5432,
      "srcc_p": 0.0134,
      "srcc_ci_95": [0.2345, 0.7654],
      "plcc": 0.4987,
      "plcc_p": 0.0256,
      "plcc_ci_95": [0.1890, 0.7234],
      "rank_agreement": 0.6500,
      "n_pairs": 20
    }
  }
}
```

### `iqa_scores.csv`

Flat table for spreadsheet analysis. Columns:

```
pair_id, prompt, variant_type, human_ref_mos, human_var_mos, human_delta,
niqe_ref, niqe_var, niqe_delta,
topiq_nr_ref, topiq_nr_var, topiq_nr_delta,
...,
lpips_dist, dists_dist
```

NR metrics have 3 columns each (`_ref`, `_var`, `_delta`). FR metrics have 1 column (`_dist`).

### `vlm_responses.json`

Raw VLM responses for qualitative analysis. Contains both the parsed JSON and the raw text for each model x prompt type x pair combination. Useful for debugging parse failures and analyzing VLM reasoning.

### `vlm_partial.json`

Checkpoint file for crash recovery. Maps `pair_id -> VLM results`. Created incrementally during VLM evaluation. Automatically loaded on restart to skip completed pairs.

### `config_snapshot.yaml`

Copy of the configuration used for this run. Ensures reproducibility.

### `pilot_eval.log`

Full log with timestamps. Includes model loading progress, per-pair evaluation status, API errors, OOM events, and timing.

## Understanding Results

### Console Output

After evaluation, a summary table is printed:

```
================================================================================
PILOT EVALUATION SUMMARY
================================================================================

--- IQA Metrics ---
Metric              SRCC             95% CI     PLCC             95% CI  Rank Agr    N
----------------------------------------------------------------------------------------
brisque           0.3456  [0.123, 0.567]   0.2987  [0.089, 0.512]     0.600   20
niqe              0.2891  [0.067, 0.498]   0.2345  [0.012, 0.456]     0.550   20
topiq_nr          0.7234  [0.512, 0.890]   0.6891  [0.457, 0.846]     0.750   20
qalign            0.6543  [0.423, 0.834]   0.6012  [0.378, 0.789]     0.700   20

--- VLM Holistic Scores ---
  vlm_gpt-5.4_holistic: SRCC=0.5432 [0.235, 0.765], PLCC=0.4987, Rank Agr=0.65, N=20
================================================================================
```

### What the Statistics Mean

- **SRCC** (Spearman Rank Correlation): Do model score deltas and human score deltas rank pairs in the same order? Range [-1, 1]. Higher = better alignment with human perception.

- **PLCC** (Pearson Linear Correlation): Is the relationship between model and human deltas linear? Range [-1, 1]. High PLCC with low SRCC suggests monotonic but non-linear relationship.

- **p-value**: Statistical significance. p < 0.05 means the correlation is unlikely due to chance. At pilot scale (20-50 pairs), many correlations may not reach significance.

- **Bootstrap 95% CI**: Confidence interval from 1000 bootstrap resamples. Wide intervals (e.g., [0.1, 0.8]) indicate the correlation estimate is unreliable at current sample size. Narrow intervals (e.g., [0.6, 0.8]) indicate stability.

- **Rank Agreement**: Fraction of pairs where model and human agree on which image is better. 0.5 = random chance for binary comparison. Only computed for NR metrics (not FR, since FR doesn't distinguish direction).

### Interpreting Results for the Paper

The key finding you're looking for:

1. **All models have low SRCC** (< 0.7) when pairs involve region-dependent sensitivity (e.g., face changes vs background changes score similarly by the model but very differently by humans).

2. **Even Q-Align fails** -- despite being an MLLM trained specifically for IQA, it cannot capture the non-linear sensitivity because its training data doesn't encode region-dependent perceptual importance.

3. **VLM sensitivity prompts reveal the gap** -- in Prompt C responses, VLMs may correctly identify that a face changed but fail to assign appropriately higher `perceptual_impact` compared to background changes.

## Troubleshooting

### `ImportError: pyiqa not installed`

```bash
pip install -e ".[iqa]"
```

### `OPENAI_API_KEY not set, skipping OpenAI models`

```bash
export OPENAI_API_KEY="sk-..."
```

### GPU OOM on `qalign`

The script auto-detects OOM and skips the metric for that pair. To reduce memory:
- Use `--device cpu` (very slow but works)
- Or edit the config to use `qalign_8bit` or `qalign_4bit` instead of `qalign`

### GPU OOM on Qwen3-VL-8B

Qwen3-VL-8B requires ~18 GB VRAM. Options:
- Remove `Qwen3-VL-8B` from the config's model list
- Or use `--models gpt-5.4 gpt-4.1` to skip local models

### VLM response parse failures

Check `vlm_responses.json` for entries with `"_parse_failed": true`. Common causes:
- Model returned non-JSON text despite instructions
- Model wrapped JSON in markdown code blocks (handled automatically)
- Model returned truncated response (increase `max_tokens`)

### Script crashed mid-VLM-evaluation

Just re-run the same command. The checkpoint (`vlm_partial.json`) ensures completed pairs are skipped. Delete the checkpoint file to force a full re-run.

### `FileNotFoundError: Missing N image(s)`

The manifest references image files that don't exist. Check paths in your manifest -- they are resolved relative to the manifest file's directory.
