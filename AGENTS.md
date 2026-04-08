# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**Perceptual Sensitivity-Aware Quality Assessment for AIGC**

Core problem: Pixel-level change ≠ human perceptual quality difference. A tiny facial anomaly (uncanny valley) causes huge quality drop, while background sharpness loss is barely noticed. This gap is region-dependent, semantically-dependent, and highly non-linear. Current IQA methods cannot model this.

Data: Same-seed controlled variants (different CFG scales, VAR early exit) with region-level human perceptual quality annotations. This isolates detail quality differences while controlling composition.

Method: The model learns human perceptual sensitivity distribution from data (where small changes matter most). Prompt provides dual guidance: (1) quality criteria (what's acceptable — e.g., "vintage film grain" → noise OK) and (2) sensitivity modulation (e.g., "portrait" → face region hypersensitive). LLM reasons criteria from prompt → Grounding DINO + SAM2 localize regions → criteria-conditioned, sensitivity-aware quality model → per-region scores + semantically-weighted overall score.

Key docs: docs/idea_v2.md (current idea), docs/survey.md (literature), docs/review_discussion.md (idea evolution)

## Project Structure

```
iqa/
├── docs/              # Survey report, paper drafts, design docs
├── configs/           # YAML configs for models, data, training
│   ├── model/         # Model architecture configs
│   ├── data/          # Dataset paths, preprocessing params
│   └── train/         # Training hyperparams, schedules
├── src/               # Core source code (importable package)
│   ├── models/        # Model definitions (QA-MSFA, MFE, heads)
│   ├── data/          # Dataset classes, dataloaders, transforms
│   ├── evaluation/    # Metrics (SRCC, PLCC, KRCC), eval pipelines
│   ├── grounding/     # Entity extraction, Grounding DINO, SAM2 wrappers
│   └── utils/         # Logging, visualization, config parsing
├── scripts/           # Standalone scripts (not importable)
│   ├── train/         # Training entry points
│   ├── eval/          # Evaluation entry points
│   ├── data/          # Data download, preprocessing, denoising sequence generation
│   └── sync/          # Server sync utilities (rsync wrappers)
├── experiments/       # Experiment configs + results (gitignored outputs)
├── notebooks/         # Exploratory analysis, visualization
├── tests/             # Unit tests
└── tools/             # One-off utilities (annotation tools, visualization)
```

## Key Technical Decisions

- **Grounding**: Grounding DINO + SAM 2 pipeline for entity localization
- **Quality Module**: SEAGULL-style MFE (Global + Local dual-view) + Q-Ground MSFA (multi-scale feature abstraction)
- **MLLM Backbone**: Qwen2-VL-7B (for end-to-end model)
- **Training Data**: Diffusion denoising sequences + Q-Eval-100K/AGIQA-3K quality labels + TOPIQ pseudo-labels

## Development Workflow

- Local dev on macOS, training on remote GPU servers
- Use `scripts/sync/` for code sync to server (`rsync` based)
- Configs are YAML-based, parsed with OmegaConf
- Experiments tracked with wandb

## Commands

```bash
# Environment setup
pip install -e ".[dev]"

# Run training (local debug)
python scripts/train/train.py --config configs/train/debug.yaml

# Run evaluation
python scripts/eval/evaluate.py --config configs/train/eval.yaml

# Sync to server
bash scripts/sync/push.sh

# Generate denoising sequence data
python scripts/data/gen_denoising_seq.py --config configs/data/denoising.yaml
```

## Related Work (key repos to reference)

- SEAGULL: https://github.com/chencn2020/SEAGULL (MFE module, LoRA training)
- Q-Ground: https://github.com/Q-Future/Q-Ground (MSFA, quality grounding)
- IQA-PyTorch: https://github.com/chaofengc/IQA-PyTorch (TOPIQ, NIQE, BRISQUE metrics)
- Grounded-SAM-2: https://github.com/IDEA-Research/Grounded-SAM-2
