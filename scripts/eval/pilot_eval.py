#!/usr/bin/env python3
"""Pilot evaluation: test existing IQA models and VLMs on perceptual sensitivity task.

Evaluates whether current models can identify images where slight pixel-level changes
create significantly different visual impressions on humans.

Usage:
    python scripts/eval/pilot_eval.py --config configs/eval/pilot.yaml
    python scripts/eval/pilot_eval.py --config configs/eval/pilot.yaml --skip-vlm --device cpu
    python scripts/eval/pilot_eval.py --config configs/eval/pilot.yaml --models topiq_nr gpt-5.4

Requirements:
    pip install -e ".[iqa]"       # for pyiqa models
    pip install -e ".[vlm]"       # for OpenAI/Gemini/Anthropic APIs
    pip install transformers>=4.57.0  # for Qwen3-VL local inference
"""

import argparse
import base64
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger("pilot_eval")

# ---------------------------------------------------------------------------
# VLM Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in perceptual image quality assessment for AI-generated images. "
    "You understand that human perception of quality is non-linear and region-dependent: "
    "a tiny facial artifact can be far more disturbing than significant background blur. "
    "Always respond in valid JSON format as specified in the user prompt."
)

PROMPT_HOLISTIC = """\
I will show you an AI-generated image. The generation prompt was: "{prompt}"

Evaluate its perceptual quality on a scale of 0-100, where:
- 0-20: Severe artifacts, distortions, or quality issues
- 21-40: Noticeable quality problems that significantly detract
- 41-60: Acceptable quality with some visible issues
- 61-80: Good quality with minor imperfections
- 81-100: Excellent quality, no perceptible issues

Consider both technical quality (sharpness, artifacts, distortions) and how well the \
image quality matches the intent of the prompt.

Respond in JSON:
{{"overall_score": <int 0-100>, "key_observations": ["<observation 1>", ...], \
"worst_region": "<region name or none>", "worst_region_issue": "<description or none>"}}"""

PROMPT_COMPARATIVE = """\
I will show you two AI-generated images created from the same prompt with the same seed \
(same composition), but with different generation parameters that affect detail quality. \
The prompt was: "{prompt}"

Image A is shown first. Image B is shown second.

Please compare their perceptual quality:
1. Rate each image independently (0-100)
2. Identify the regions where you notice quality differences
3. For each region with differences, rate how perceptually significant the difference is \
(1=barely noticeable, 5=dramatically different quality)
4. Which image has better overall perceptual quality?

Respond in JSON:
{{"image_a_score": <int 0-100>, "image_b_score": <int 0-100>, "better_image": "A" or "B", \
"region_differences": [{{"region": "<name>", "description": "<what differs>", \
"perceptual_significance": <int 1-5>, "which_better": "A" or "B"}}], \
"overall_explanation": "<1-2 sentence summary>"}}"""

PROMPT_SENSITIVITY = """\
I will show you two AI-generated images of "{prompt}", created with the same seed but \
different generation parameters.

These images have the same composition but differ in detail quality in various regions. \
Your task is to assess PERCEPTUAL SENSITIVITY — how much each regional difference \
actually matters to a human viewer.

A key insight: a tiny artifact on a face can be far more disturbing than significant blur \
in the background. We want to understand if you can capture this human tendency.

For each region where you notice a difference:
1. Describe the difference
2. Estimate pixel-level magnitude (small/medium/large change)
3. Estimate perceptual impact (how much a human would care: minimal/moderate/significant/severe)
4. Explain WHY this region's change has the perceptual impact it does

Respond in JSON:
{{"regions_analyzed": [{{"region": "<name>", "difference_description": "<what changed>", \
"pixel_magnitude": "small"|"medium"|"large", \
"perceptual_impact": "minimal"|"moderate"|"significant"|"severe", \
"explanation": "<why this matters or doesn't>"}}], \
"most_impactful_change": "<region name>", "least_impactful_change": "<region name>", \
"sensitivity_reasoning": "<explain the gap between pixel change and perceptual impact>"}}"""


# ---------------------------------------------------------------------------
# CLI & Config
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Pilot evaluation for perceptual sensitivity")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest path")
    parser.add_argument("--models", nargs="+", default=None, help="Run only these models")
    parser.add_argument("--skip-iqa", action="store_true", help="Skip IQA model evaluation")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM evaluation")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.manifest:
        cfg.data.manifest = args.manifest
    if args.device:
        cfg.models.iqa.device = args.device
    if args.skip_iqa:
        cfg.models.iqa.enabled = False
    if args.skip_vlm:
        cfg.models.vlm.enabled = False
    if args.output_dir:
        cfg.experiment.output_dir = args.output_dir
    cfg._cli_models = args.models
    cfg._cli_max_pairs = args.max_pairs
    return cfg


def setup_logging(cfg):
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = cfg.logging.get("log_file")
    if log_file is None:
        log_file = output_dir / "pilot_eval.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logger.info("Logging to %s", log_file)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    base_dir = manifest_path.parent
    pairs = manifest["pairs"]
    missing = []
    for pair in pairs:
        for key in ("reference", "variant"):
            img_path = Path(pair[key])
            if not img_path.is_absolute():
                img_path = base_dir / img_path
            pair[f"_{key}_abs"] = str(img_path)
            if not img_path.exists():
                missing.append(str(img_path))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} image(s):\n" + "\n".join(missing[:10])
        )

    n_human = sum(1 for p in pairs if "human_scores" in p)
    logger.info(
        "Loaded manifest: %d pairs, %d with human scores, variant types: %s",
        len(pairs), n_human,
        list({p.get("variant_type", "unknown") for p in pairs}),
    )
    return manifest


def load_image_tensor(path: str, max_size=None):
    """Load image as [0,1] float tensor (1, C, H, W)."""
    import torch
    import torchvision.transforms.functional as TF

    img = Image.open(path).convert("RGB")
    if max_size and max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.height * ratio), int(img.width * ratio))
        img = img.resize((new_size[1], new_size[0]), Image.LANCZOS)
    tensor = TF.to_tensor(img).unsqueeze(0)  # (1, 3, H, W), [0, 1]
    return tensor


# ---------------------------------------------------------------------------
# IQA Evaluator
# ---------------------------------------------------------------------------

class IQAEvaluator:
    def __init__(self, cfg):
        try:
            import pyiqa
        except ImportError:
            raise ImportError(
                "pyiqa not installed. Run: pip install -e \".[iqa]\""
            )
        import torch

        self.device = torch.device(cfg.models.iqa.device)
        self.max_size = cfg.data.get("max_image_size")
        self.nr_metrics = {}
        self.fr_metrics = {}
        self.lower_better = {}

        filter_models = cfg.get("_cli_models")

        # Load NR models
        for name in cfg.models.iqa.nr_models:
            if filter_models and name not in filter_models:
                continue
            try:
                logger.info("Loading NR metric: %s", name)
                m = pyiqa.create_metric(name, device=self.device)
                self.nr_metrics[name] = m
                self.lower_better[name] = m.lower_better
            except Exception as e:
                logger.warning("Failed to load NR metric %s: %s", name, e)

        # Load FR models
        for name in cfg.models.iqa.fr_models:
            if filter_models and name not in filter_models:
                continue
            try:
                logger.info("Loading FR metric: %s", name)
                m = pyiqa.create_metric(name, device=self.device)
                self.fr_metrics[name] = m
                self.lower_better[name] = m.lower_better
            except Exception as e:
                logger.warning("Failed to load FR metric %s: %s", name, e)

        logger.info(
            "IQA ready: %d NR + %d FR metrics",
            len(self.nr_metrics), len(self.fr_metrics),
        )

    def evaluate_pair(self, pair: dict) -> dict:
        import torch

        ref_path = pair["_reference_abs"]
        var_path = pair["_variant_abs"]
        ref_tensor = load_image_tensor(ref_path, self.max_size).to(self.device)
        var_tensor = load_image_tensor(var_path, self.max_size).to(self.device)

        results = {}
        with torch.no_grad():
            for name, metric in self.nr_metrics.items():
                try:
                    ref_score = metric(ref_tensor).item()
                    var_score = metric(var_tensor).item()
                    results[name] = {
                        "ref_score": ref_score,
                        "var_score": var_score,
                        "delta": var_score - ref_score,
                        "lower_better": self.lower_better[name],
                    }
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM on %s for pair %s, skipping", name, pair["pair_id"])
                        torch.cuda.empty_cache()
                        results[name] = None
                    else:
                        raise

            for name, metric in self.fr_metrics.items():
                try:
                    score = metric(var_tensor, ref_tensor).item()
                    results[name] = {
                        "score": score,
                        "lower_better": self.lower_better[name],
                    }
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM on %s for pair %s, skipping", name, pair["pair_id"])
                        torch.cuda.empty_cache()
                        results[name] = None
                    else:
                        raise

        # Free GPU memory
        del ref_tensor, var_tensor
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
        return results

    def evaluate_all(self, pairs: list[dict]) -> list[dict]:
        results = []
        for pair in tqdm(pairs, desc="IQA evaluation"):
            r = self.evaluate_pair(pair)
            results.append({"pair_id": pair["pair_id"], "iqa": r})
        return results


# ---------------------------------------------------------------------------
# VLM Evaluator
# ---------------------------------------------------------------------------

class VLMEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vlm_cfg = cfg.models.vlm
        self.prompt_types = list(self.vlm_cfg.prompts)
        self.rate_limit_delay = self.vlm_cfg.get("rate_limit_delay", 2.0)
        self.max_retries = self.vlm_cfg.get("max_retries", 3)
        self.retry_delay = self.vlm_cfg.get("retry_delay", 10.0)
        self.max_size = cfg.data.get("max_image_size")

        self.models = []
        self.openai_client = None
        self.local_model = None
        self.local_processor = None

        filter_models = cfg.get("_cli_models")

        for model_cfg in self.vlm_cfg.models:
            name = model_cfg["name"]
            if filter_models and name not in filter_models:
                continue
            provider = model_cfg["provider"]

            if provider == "openai":
                if self.openai_client is None:
                    self.openai_client = self._init_openai(model_cfg)
                if self.openai_client is not None:
                    self.models.append(model_cfg)
            elif provider == "local":
                self.models.append(model_cfg)
            elif provider in ("google", "anthropic"):
                logger.info("%s provider not implemented yet, skipping %s", provider, name)
            else:
                logger.warning("Unknown provider: %s", provider)

        logger.info("VLM ready: %d models", len(self.models))

        # Checkpoint path
        self.output_dir = Path(cfg.experiment.output_dir)
        self.checkpoint_path = self.output_dir / "vlm_partial.json"

    def _init_openai(self, model_cfg):
        api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            logger.warning("Environment variable %s not set, skipping OpenAI models", api_key_env)
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
            return client
        except ImportError:
            logger.warning("openai package not installed, skipping. Run: pip install -e \".[vlm]\"")
            return None

    def _load_local_model(self, model_cfg):
        if self.local_model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_id = model_cfg["model_id"]
            device = model_cfg.get("device", "cuda")
            logger.info("Loading local model: %s", model_id)
            self.local_model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype="auto", device_map=device,
            )
            self.local_processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Local model loaded: %s", model_id)
        except Exception as e:
            logger.warning("Failed to load local model: %s", e)
            self.local_model = None

    def _encode_image_base64(self, path: str) -> str:
        """Encode image as base64 PNG (lossless to preserve subtle differences)."""
        img = Image.open(path).convert("RGB")
        if self.max_size and max(img.size) > self.max_size:
            ratio = self.max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_prompt(self, prompt_type: str, pair: dict, image_order: dict) -> tuple[str, list[str]]:
        """Build prompt text and list of image paths in presentation order.

        Returns (prompt_text, [image_path_1, image_path_2, ...]).
        For holistic: returns two separate prompts (one per image), each with one image.
        For comparative/sensitivity: returns one prompt with two images in randomized order.
        """
        gen_prompt = pair.get("prompt", "")

        if prompt_type == "holistic":
            return PROMPT_HOLISTIC.format(prompt=gen_prompt), None

        order = image_order  # {"A": ref_or_var_path, "B": ...}
        if prompt_type == "comparative":
            text = PROMPT_COMPARATIVE.format(prompt=gen_prompt)
        else:  # sensitivity
            text = PROMPT_SENSITIVITY.format(prompt=gen_prompt)

        return text, [order["A"], order["B"]]

    def _call_openai(self, model_id: str, images: list[str], prompt: str) -> str:
        """Call OpenAI Responses API with images (base64 PNG)."""
        input_parts = [{"type": "input_text", "text": prompt}]
        for img_b64 in images:
            input_parts.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_b64}",
            })

        for attempt in range(self.max_retries + 1):
            try:
                response = self.openai_client.responses.create(
                    model=model_id,
                    instructions=SYSTEM_PROMPT,
                    input=input_parts,
                    text={"format": {"type": "json_object"}},
                )
                return response.output_text
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "500" in err_str or "503" in err_str:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning("OpenAI API error (attempt %d/%d): %s. Retrying in %.0fs",
                                   attempt + 1, self.max_retries + 1, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("OpenAI API error (non-retryable): %s", e)
                    return None
        logger.error("OpenAI API failed after %d retries", self.max_retries + 1)
        return None

    def _call_local(self, model_cfg: dict, images: list[str], prompt: str) -> str:
        """Call local Qwen3-VL model."""
        self._load_local_model(model_cfg)
        if self.local_model is None:
            return None

        import torch

        content = []
        for img_path in images:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": SYSTEM_PROMPT + "\n\n" + prompt})

        messages = [{"role": "user", "content": content}]

        try:
            inputs = self.local_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = inputs.to(self.local_model.device)

            max_tokens = model_cfg.get("max_tokens", 1500)
            with torch.no_grad():
                generated_ids = self.local_model.generate(**inputs, max_new_tokens=max_tokens)

            trimmed = [
                out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)
            ]
            text = self.local_processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
            return text
        except Exception as e:
            logger.error("Local model error: %s", e)
            return None

    def _parse_response(self, text: str | None) -> dict | None:
        if text is None:
            return None
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse VLM JSON response (length=%d)", len(text))
        return {"_raw": text, "_parse_failed": True}

    def _randomize_order(self, pair: dict) -> dict:
        """Randomly assign reference/variant to A/B positions. Returns order mapping."""
        ref_path = pair["_reference_abs"]
        var_path = pair["_variant_abs"]
        if random.random() < 0.5:
            return {"A": ref_path, "B": var_path, "A_is": "reference", "B_is": "variant"}
        else:
            return {"A": var_path, "B": ref_path, "A_is": "variant", "B_is": "reference"}

    def evaluate_pair(self, pair: dict) -> dict:
        image_order = self._randomize_order(pair)
        results = {"image_order": image_order}

        for model_cfg in self.models:
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
            name = model_cfg["name"]
            provider = model_cfg["provider"]
            model_id = model_cfg.get("model_id", name)
            model_results = {}

            for prompt_type in self.prompt_types:
                if prompt_type == "holistic":
                    # Score each image independently
                    prompt_text = PROMPT_HOLISTIC.format(prompt=pair.get("prompt", ""))
                    holistic_results = {}
                    for label, img_path in [("reference", pair["_reference_abs"]),
                                            ("variant", pair["_variant_abs"])]:
                        img_b64 = self._encode_image_base64(img_path)
                        if provider == "openai":
                            raw = self._call_openai(model_id, [img_b64], prompt_text)
                        elif provider == "local":
                            raw = self._call_local(model_cfg, [img_path], prompt_text)
                        else:
                            raw = None
                        holistic_results[label] = {
                            "parsed": self._parse_response(raw),
                            "raw": raw,
                        }
                        time.sleep(self.rate_limit_delay)
                    model_results[prompt_type] = holistic_results
                else:
                    # Comparative / sensitivity: send both images
                    prompt_text, img_paths = self._build_prompt(prompt_type, pair, image_order)
                    if provider == "openai":
                        imgs_b64 = [self._encode_image_base64(p) for p in img_paths]
                        raw = self._call_openai(model_id, imgs_b64, prompt_text)
                    elif provider == "local":
                        raw = self._call_local(model_cfg, img_paths, prompt_text)
                    else:
                        raw = None
                    model_results[prompt_type] = {
                        "parsed": self._parse_response(raw),
                        "raw": raw,
                    }
                    time.sleep(self.rate_limit_delay)

            results[name] = model_results

        return results

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            logger.info("Loaded checkpoint: %d pairs completed", len(data))
            return data
        return {}

    def _save_checkpoint(self, completed: dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)

    def evaluate_all(self, pairs: list[dict]) -> list[dict]:
        checkpoint = self._load_checkpoint()
        results = []

        for pair in tqdm(pairs, desc="VLM evaluation"):
            pid = pair["pair_id"]
            if pid in checkpoint:
                logger.info("Skipping pair %s (already in checkpoint)", pid)
                results.append({"pair_id": pid, "vlm": checkpoint[pid]})
                continue

            r = self.evaluate_pair(pair)
            results.append({"pair_id": pid, "vlm": r})
            checkpoint[pid] = r
            self._save_checkpoint(checkpoint)

        return results


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(x, y, stat_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a correlation statistic."""
    rng = np.random.RandomState(seed)
    n = len(x)
    bootstrapped = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            val = stat_fn(x[idx], y[idx])
            if hasattr(val, "__len__"):
                val = val[0]  # spearmanr/pearsonr return (stat, pvalue)
            bootstrapped.append(val)
        except Exception:
            continue
    if not bootstrapped:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrapped, alpha * 100)
    upper = np.percentile(bootstrapped, (1 - alpha) * 100)
    return (lower, upper)


def compute_summary_stats(iqa_results: list[dict], vlm_results: list[dict],
                          pairs: list[dict]) -> dict:
    """Compute SRCC, PLCC, rank agreement, and bootstrap CI."""
    summary = {"iqa": {}, "vlm": {}}

    # Build human delta array
    human_deltas = {}
    for p in pairs:
        hs = p.get("human_scores")
        if hs and "reference_mos" in hs and "variant_mos" in hs:
            human_deltas[p["pair_id"]] = hs["variant_mos"] - hs["reference_mos"]

    if not human_deltas:
        logger.warning("No human scores available, skipping correlation analysis")
        return summary

    # IQA metrics
    if iqa_results:
        # Collect all metric names
        metric_names = set()
        for r in iqa_results:
            metric_names.update(r.get("iqa", {}).keys())

        for metric_name in sorted(metric_names):
            model_deltas = []
            human_vals = []
            agreements = []

            for r in iqa_results:
                pid = r["pair_id"]
                if pid not in human_deltas:
                    continue
                metric_data = r.get("iqa", {}).get(metric_name)
                if metric_data is None:
                    continue

                hd = human_deltas[pid]

                if "delta" in metric_data:
                    md = metric_data["delta"]
                    lb = metric_data.get("lower_better", False)
                    # Normalize direction: positive delta = variant better
                    if lb:
                        md = -md  # For lower_better, negative delta means variant is better
                    model_deltas.append(md)
                    human_vals.append(hd)
                    # Rank agreement: do they agree on which image is better?
                    agreements.append(1 if (md > 0) == (hd > 0) else 0)
                elif "score" in metric_data:
                    # FR metric: just a distance, no directional delta
                    model_deltas.append(metric_data["score"])
                    human_vals.append(abs(hd))

            if len(model_deltas) < 3:
                continue

            model_arr = np.array(model_deltas)
            human_arr = np.array(human_vals)

            srcc, srcc_p = stats.spearmanr(model_arr, human_arr)
            plcc, plcc_p = stats.pearsonr(model_arr, human_arr)
            srcc_ci = bootstrap_ci(model_arr, human_arr, lambda x, y: stats.spearmanr(x, y)[0])
            plcc_ci = bootstrap_ci(model_arr, human_arr, lambda x, y: stats.pearsonr(x, y)[0])

            entry = {
                "srcc": round(srcc, 4),
                "srcc_p": round(srcc_p, 4),
                "srcc_ci_95": [round(srcc_ci[0], 4), round(srcc_ci[1], 4)],
                "plcc": round(plcc, 4),
                "plcc_p": round(plcc_p, 4),
                "plcc_ci_95": [round(plcc_ci[0], 4), round(plcc_ci[1], 4)],
                "n_pairs": len(model_deltas),
            }
            if agreements:
                entry["rank_agreement"] = round(sum(agreements) / len(agreements), 4)
            summary["iqa"][metric_name] = entry

    # VLM holistic scores
    if vlm_results:
        for r in vlm_results:
            pid = r["pair_id"]
            vlm_data = r.get("vlm", {})
            for model_name, model_data in vlm_data.items():
                if model_name == "image_order":
                    continue
                holistic = model_data.get("holistic")
                if not holistic:
                    continue

                ref_parsed = holistic.get("reference", {}).get("parsed")
                var_parsed = holistic.get("variant", {}).get("parsed")
                if not ref_parsed or not var_parsed:
                    continue
                if ref_parsed.get("_parse_failed") or var_parsed.get("_parse_failed"):
                    continue

                ref_score = ref_parsed.get("overall_score")
                var_score = var_parsed.get("overall_score")
                if ref_score is None or var_score is None:
                    continue

                key = f"vlm_{model_name}_holistic"
                if key not in summary["vlm"]:
                    summary["vlm"][key] = {"model_deltas": [], "human_deltas": [], "agreements": []}

                if pid in human_deltas:
                    md = var_score - ref_score
                    hd = human_deltas[pid]
                    summary["vlm"][key]["model_deltas"].append(md)
                    summary["vlm"][key]["human_deltas"].append(hd)
                    summary["vlm"][key]["agreements"].append(1 if (md > 0) == (hd > 0) else 0)

        # Compute stats for VLM results
        for key, data in list(summary["vlm"].items()):
            if len(data["model_deltas"]) < 3:
                summary["vlm"][key] = {"n_pairs": len(data["model_deltas"]), "insufficient_data": True}
                continue

            model_arr = np.array(data["model_deltas"])
            human_arr = np.array(data["human_deltas"])

            srcc, srcc_p = stats.spearmanr(model_arr, human_arr)
            plcc, plcc_p = stats.pearsonr(model_arr, human_arr)
            srcc_ci = bootstrap_ci(model_arr, human_arr, lambda x, y: stats.spearmanr(x, y)[0])
            plcc_ci = bootstrap_ci(model_arr, human_arr, lambda x, y: stats.pearsonr(x, y)[0])

            summary["vlm"][key] = {
                "srcc": round(srcc, 4),
                "srcc_p": round(srcc_p, 4),
                "srcc_ci_95": [round(srcc_ci[0], 4), round(srcc_ci[1], 4)],
                "plcc": round(plcc, 4),
                "plcc_p": round(plcc_p, 4),
                "plcc_ci_95": [round(plcc_ci[0], 4), round(plcc_ci[1], 4)],
                "rank_agreement": round(sum(data["agreements"]) / len(data["agreements"]), 4),
                "n_pairs": len(data["model_deltas"]),
            }

    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(iqa_results, vlm_results, summary, cfg, pairs):
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full results JSON
    full = {
        "experiment": OmegaConf.to_container(cfg.experiment, resolve=True),
        "iqa_results": iqa_results,
        "vlm_results": vlm_results,
        "summary": summary,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(full, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved results.json")

    # Summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary.json")

    # VLM raw responses
    if vlm_results:
        with open(output_dir / "vlm_responses.json", "w") as f:
            json.dump(vlm_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved vlm_responses.json")

    # IQA CSV
    if iqa_results and cfg.output.save_csv:
        _save_iqa_csv(iqa_results, pairs, output_dir / "iqa_scores.csv")

    # Config snapshot
    OmegaConf.save(cfg, output_dir / "config_snapshot.yaml")
    logger.info("Saved config_snapshot.yaml")


def _save_iqa_csv(iqa_results, pairs, path):
    import csv as csv_mod

    pair_map = {p["pair_id"]: p for p in pairs}

    # Classify metrics as NR (has delta) or FR (has score)
    metric_type = {}  # name -> "nr" | "fr"
    for r in iqa_results:
        for m, data in r.get("iqa", {}).items():
            if m in metric_type or data is None:
                continue
            metric_type[m] = "nr" if "delta" in data else "fr"
    all_metrics = sorted(metric_type.keys())

    # Build header
    header = ["pair_id", "prompt", "variant_type", "human_ref_mos", "human_var_mos", "human_delta"]
    for m in all_metrics:
        if metric_type[m] == "nr":
            header.extend([f"{m}_ref", f"{m}_var", f"{m}_delta"])
        else:
            header.append(f"{m}_dist")

    with open(path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(header)

        for r in iqa_results:
            pid = r["pair_id"]
            pair = pair_map.get(pid, {})
            hs = pair.get("human_scores", {})

            ref_mos = hs.get("reference_mos")
            var_mos = hs.get("variant_mos")
            row = [
                pid,
                pair.get("prompt", ""),
                pair.get("variant_type", ""),
                ref_mos if ref_mos is not None else "",
                var_mos if var_mos is not None else "",
                var_mos - ref_mos if ref_mos is not None and var_mos is not None else "",
            ]

            for m in all_metrics:
                data = r.get("iqa", {}).get(m)
                if data is None:
                    row.extend([""] * (3 if metric_type[m] == "nr" else 1))
                elif metric_type[m] == "nr":
                    row.extend([
                        f"{data['ref_score']:.4f}",
                        f"{data['var_score']:.4f}",
                        f"{data['delta']:.4f}",
                    ])
                else:
                    row.append(f"{data['score']:.4f}")

            writer.writerow(row)

    logger.info("Saved %s (%d rows)", path, len(iqa_results))


def print_summary_table(summary):
    """Print a human-readable summary to console."""
    print("\n" + "=" * 80)
    print("PILOT EVALUATION SUMMARY")
    print("=" * 80)

    if summary.get("iqa"):
        print("\n--- IQA Metrics ---")
        print(f"{'Metric':<15} {'SRCC':>8} {'95% CI':>20} {'PLCC':>8} {'95% CI':>20} {'Rank Agr':>9} {'N':>4}")
        print("-" * 88)
        for name, s in sorted(summary["iqa"].items()):
            ci_s = f"[{s['srcc_ci_95'][0]:.3f}, {s['srcc_ci_95'][1]:.3f}]"
            ci_p = f"[{s['plcc_ci_95'][0]:.3f}, {s['plcc_ci_95'][1]:.3f}]"
            ra_val = s.get('rank_agreement')
            ra = f"{ra_val:.3f}" if ra_val is not None else "N/A"
            print(f"{name:<15} {s['srcc']:>8.4f} {ci_s:>20} {s['plcc']:>8.4f} {ci_p:>20} {ra:>9} {s['n_pairs']:>4}")

    if summary.get("vlm"):
        print("\n--- VLM Holistic Scores ---")
        for name, s in sorted(summary["vlm"].items()):
            if s.get("insufficient_data"):
                print(f"  {name}: insufficient data ({s['n_pairs']} pairs)")
                continue
            ci_s = f"[{s['srcc_ci_95'][0]:.3f}, {s['srcc_ci_95'][1]:.3f}]"
            print(f"  {name}: SRCC={s['srcc']:.4f} {ci_s}, "
                  f"PLCC={s['plcc']:.4f}, Rank Agr={s.get('rank_agreement', 'N/A')}, "
                  f"N={s['n_pairs']}")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()
    setup_logging(cfg)

    logger.info("Experiment: %s", cfg.experiment.name)
    logger.info("Output: %s", cfg.experiment.output_dir)

    # Load data
    manifest = load_manifest(cfg.data.manifest)
    pairs = manifest["pairs"]
    if cfg.get("_cli_max_pairs"):
        pairs = pairs[:cfg._cli_max_pairs]
        logger.info("Limited to %d pairs", len(pairs))

    iqa_results = []
    vlm_results = []

    # IQA evaluation
    if cfg.models.iqa.enabled:
        logger.info("=== IQA Evaluation ===")
        evaluator = IQAEvaluator(cfg)
        iqa_results = evaluator.evaluate_all(pairs)
    else:
        logger.info("IQA evaluation skipped")

    # VLM evaluation
    if cfg.models.vlm.enabled:
        logger.info("=== VLM Evaluation ===")
        evaluator = VLMEvaluator(cfg)
        vlm_results = evaluator.evaluate_all(pairs)
    else:
        logger.info("VLM evaluation skipped")

    # Summary statistics
    summary = compute_summary_stats(iqa_results, vlm_results, pairs)

    # Save outputs
    save_results(iqa_results, vlm_results, summary, cfg, pairs)
    print_summary_table(summary)

    logger.info("Done.")


if __name__ == "__main__":
    main()
