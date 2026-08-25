"""Runtime-only launcher for the upstream adapter script.

This file keeps the upstream implementation unchanged: it only supplies the
local config merge and a two-GPU device map for the cached Qwen checkpoint.
"""
from __future__ import annotations

import os
import runpy
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
if extra_site := os.environ.get("THOUGHTCOMM_EXTRA_SITE"):
    sys.path.insert(0, extra_site)

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from thoughtcomm.models.hf import HFModelBundle
import thoughtcomm.models.hf as hfmod
import thoughtcomm.core.config as cfgmod
import thoughtcomm.thoughts.adapter as adaptermod


def load_sharded(
    path: str,
    dtype,
    attn_implementation: str = "sdpa",
    quantization=None,
    device: str = "cuda",
    gradient_checkpointing: bool = False,
    compile_model: bool = False,
) -> HFModelBundle:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if os.environ.get("THOUGHTCOMM_SINGLE_GPU") == "1":
        device_map = {"": 0}
    else:
        n_layers = 28
        device_map = {
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
            "model.norm": 0,
            "lm_head": 0,
        }
        for i in range(n_layers):
            device_map[f"model.layers.{i}"] = 0 if i < n_layers // 2 else 1

    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    model.eval()
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model config")
    print(
        "Loaded upstream model across logical CUDA devices 0/1 "
        "(physical GPUs 1/2 via CUDA_VISIBLE_DEVICES).",
        flush=True,
    )
    return HFModelBundle(model=model, tok=tok, hidden_size=int(hidden_size))


def load_config(config_path: str, overrides=None):
    base = OmegaConf.load(str(REPO / "configs/default.yaml"))
    local = OmegaConf.load(str(config_path))
    cfg = OmegaConf.merge(base, local)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


hfmod.load_hf_causal_lm = load_sharded
cfgmod.load_config = load_config

# Runtime-only numerical stabilization. The upstream objective, prefix
# mapping, sampling, and training loop remain unchanged; this only bounds the
# adapter optimizer update when the launcher is explicitly configured to do so.
grad_clip = float(os.environ.get("THOUGHTCOMM_GRAD_CLIP", "0"))
if grad_clip > 0:
    class ClippedAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            params = [
                p
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=grad_clip, error_if_nonfinite=True
            )
            return super().step(closure)

    torch.optim.AdamW = ClippedAdamW

prefix_init_scale_raw = os.environ.get("THOUGHTCOMM_PREFIX_INIT_SCALE")
prefix_init_scale = float(prefix_init_scale_raw) if prefix_init_scale_raw is not None else 0.0
adapter_ckpt = os.environ.get("THOUGHTCOMM_ADAPTER_CKPT")
if prefix_init_scale_raw is not None or adapter_ckpt:
    original_adapter_init = adaptermod.PrefixAdapter.__init__

    def scaled_adapter_init(self, *args, **kwargs):
        original_adapter_init(self, *args, **kwargs)
        with torch.no_grad():
            final = self.net[-1]
            final.weight.mul_(prefix_init_scale)
            if final.bias is not None:
                final.bias.mul_(prefix_init_scale)
        if adapter_ckpt:
            self.load_state_dict(torch.load(adapter_ckpt, map_location="cpu"))

    adaptermod.PrefixAdapter.__init__ = scaled_adapter_init

adapter_lr = os.environ.get("THOUGHTCOMM_ADAPTER_LR")
model_dtype = os.environ.get("THOUGHTCOMM_MODEL_DTYPE", "bfloat16")
overrides = [f"model.dtype={model_dtype}", "runtime.device=cuda"]
if model_path := os.environ.get("THOUGHTCOMM_MODEL_PATH"):
    overrides.append(f"model.path={model_path}")
if adapter_lr:
    overrides.append(f"adapter.lr={adapter_lr}")
overrides.extend(shlex.split(os.environ.get("THOUGHTCOMM_OVERRIDES", "")))

config_path = os.environ.get("THOUGHTCOMM_CONFIG", "configs/local_qwen3_exp1.yaml")
output_dir = os.environ.get(
    "THOUGHTCOMM_OUTPUT_DIR", str(REPO / "artifacts/full_qwen3_exp1/adapter")
)
contexts_path = os.environ.get(
    "THOUGHTCOMM_CONTEXTS_PATH",
    str(REPO / "artifacts/full_qwen3_exp1/merged/contexts.jsonl"),
)
states_path = os.environ.get(
    "THOUGHTCOMM_STATES_PATH",
    str(REPO / "artifacts/full_qwen3_exp1/merged/states.pt"),
)
ae_ckpt = os.environ.get(
    "THOUGHTCOMM_AE_CKPT", str(REPO / "artifacts/full_qwen3_exp1/ae/ae.pt")
)
mask_path = os.environ.get(
    "THOUGHTCOMM_MASK_PATH", str(REPO / "artifacts/full_qwen3_exp1/structure/mask.pt")
)

stable_sampling = os.environ.get("THOUGHTCOMM_STABLE_SAMPLING", "0") == "1"
stable_loss = os.environ.get("THOUGHTCOMM_STABLE_LOSS", "0") == "1"

sys.argv = [
    "scripts/train_adapter.py",
    "--config",
    config_path,
    *overrides,
    "--output_dir",
    output_dir,
    "--contexts_path",
    contexts_path,
    "--states_path",
    states_path,
    "--ae_ckpt",
    ae_ckpt,
    "--mask_path",
    mask_path,
]

# Runtime-only graph cleanup: keep the upstream file and objective unchanged,
# but release step-local autograd references after the progress update.
train_path = REPO / "scripts/train_adapter.py"
train_source = train_path.read_text(encoding="utf-8")
graph_marker = "            pbar.set_postfix(loss=float(loss.item()), sem=float(sem.item()), flu=float(flu.item()))" + chr(10)
if train_source.count(graph_marker) != 1:
    raise RuntimeError("upstream train_adapter.py progress marker changed")
train_source = train_source.replace(graph_marker, graph_marker + "            del loss_sum, loss, prefixes, sem, flu, gen_emb" + chr(10), 1)
train_globals = {}
exec(compile(train_source, str(train_path), "exec"), train_globals)
if stable_sampling:
    def stable_top_p_sample(logits, top_p, temperature):
        logits = logits.float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite prefix logits before top-p sampling")
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        probs = F.softmax(logits / temperature, dim=-1)
        if not bool(torch.isfinite(probs).all()):
            raise FloatingPointError("non-finite float32 probabilities in top-p sampling")
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        keep[0] = True
        filt_idx = sorted_idx[keep]
        filt_probs = sorted_probs[keep]
        filt_probs = filt_probs / filt_probs.sum()
        if not bool(torch.isfinite(filt_probs).all()):
            raise FloatingPointError("non-finite filtered probabilities in top-p sampling")
        sampled = torch.multinomial(filt_probs, 1)
        return int(filt_idx[sampled].item())

    train_globals["top_p_sample"] = stable_top_p_sample
if stable_loss:
    def stable_mean_hidden_no_prefix(model, context_ids, suffix_ids):
        device = model.device
        full = torch.cat([context_ids.to(device), suffix_ids.to(device)], dim=0).unsqueeze(0)
        out = model.model(
            input_ids=full,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state[0, context_ids.shape[0] :, :].mean(dim=0).float()

    def stable_mean_hidden_with_prefix(model, context_ids, suffix_ids, prefix):
        device = model.device
        emb = model.get_input_embeddings()
        ctx = context_ids.to(device).unsqueeze(0)
        suf = suffix_ids.to(device).unsqueeze(0)
        pref = prefix.to(device).unsqueeze(0)
        inputs_embeds = torch.cat([pref, emb(ctx), emb(suf)], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
        out = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )
        m, length = pref.shape[1], ctx.shape[1]
        return out.last_hidden_state[0, m + length :, :].mean(dim=0).float()

    def stable_nll_of_suffix_with_prefix(model, tok, context_ids, suffix_ids, prefix):
        device = model.device
        prefix = prefix.to(device).unsqueeze(0)
        ctx = context_ids.to(device).unsqueeze(0)
        suf = suffix_ids.to(device).unsqueeze(0)
        emb = model.get_input_embeddings()
        inputs_embeds = torch.cat([prefix, emb(ctx), emb(suf)], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )
        m, length, suffix_length = prefix.shape[1], ctx.shape[1], suf.shape[1]
        start = m + length
        pred_logits = out.logits[:, start - 1 : start + suffix_length - 1, :].float()
        return F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            suf.reshape(-1),
            reduction="mean",
        )

    train_globals["mean_hidden_no_prefix"] = stable_mean_hidden_no_prefix
    train_globals["mean_hidden_with_prefix"] = stable_mean_hidden_with_prefix
    train_globals["nll_of_suffix_with_prefix"] = stable_nll_of_suffix_with_prefix
train_globals["main"]()
