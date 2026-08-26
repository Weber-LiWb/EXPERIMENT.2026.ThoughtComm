#!/usr/bin/env python
"""Paired no-prefix versus trained-prefix behavioral evaluation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from thoughtcomm.thoughts.adapter import PrefixAdapter
from thoughtcomm.thoughts.autoencoder import SparseJacobianAutoEncoder
from thoughtcomm.thoughts.routing import agreement_alpha, build_weight_vector, route_latents


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--adapter_ckpt", required=True)
    ap.add_argument("--contexts_path", required=True)
    ap.add_argument("--states_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--num_states", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_context_tokens", type=int, default=4096)
    ap.add_argument("--single_gpu", action="store_true")
    ap.add_argument("--sample_texts", type=int, default=6)
    return ap.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(path: str, dtype: torch.dtype, attn: str, single_gpu: bool):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if single_gpu or torch.cuda.device_count() < 2:
        device_map = {"": 0}
    else:
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        n_layers = int(getattr(cfg, "num_hidden_layers", 28))
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
        attn_implementation=attn,
        device_map=device_map,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def greedy_no_prefix(model, tok, context_ids: torch.Tensor, max_new: int) -> torch.Tensor:
    device = model.device
    inp = context_ids.unsqueeze(0).to(device)
    out = model.generate(
        input_ids=inp,
        do_sample=False,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=False,
    )
    return out[0, inp.shape[1]:].detach().cpu()


@torch.no_grad()
def greedy_with_prefix(model, tok, context_ids: torch.Tensor, prefix: torch.Tensor, max_new: int) -> torch.Tensor:
    device = model.device
    ctx = context_ids.to(device).unsqueeze(0)
    pref = prefix.to(device=device, dtype=model.dtype).unsqueeze(0)
    emb = model.get_input_embeddings()
    inputs_embeds = torch.cat([pref, emb(ctx)], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_logits = out.logits[0, -1].float()
    generated = []
    for _ in range(max_new):
        token = int(torch.argmax(next_logits).item())
        generated.append(token)
        inp = torch.tensor([[token]], device=device, dtype=torch.long)
        out = model(input_ids=inp, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1].float()
        if tok.eos_token_id is not None and token == tok.eos_token_id:
            break
    return torch.tensor(generated, dtype=torch.long)


@torch.no_grad()
def hidden_no_prefix(model, context_ids: torch.Tensor, suffix_ids: torch.Tensor) -> torch.Tensor:
    device = model.device
    full = torch.cat([context_ids, suffix_ids]).to(device).unsqueeze(0)
    out = model.model(input_ids=full, use_cache=False, return_dict=True)
    return out.last_hidden_state[0, context_ids.numel():].float().mean(dim=0)


@torch.no_grad()
def hidden_with_prefix(
    model, context_ids: torch.Tensor, suffix_ids: torch.Tensor, prefix: torch.Tensor
) -> torch.Tensor:
    device = model.device
    ctx = context_ids.to(device).unsqueeze(0)
    suf = suffix_ids.to(device).unsqueeze(0)
    pref = prefix.to(device=device, dtype=model.dtype).unsqueeze(0)
    emb = model.get_input_embeddings()
    inputs_embeds = torch.cat([pref, emb(ctx), emb(suf)], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
    out = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        use_cache=False,
        return_dict=True,
    )
    start = pref.shape[1] + ctx.shape[1]
    return out.last_hidden_state[0, start:].float().mean(dim=0)


@torch.no_grad()
def nll_no_prefix(model, context_ids: torch.Tensor, suffix_ids: torch.Tensor) -> float:
    if suffix_ids.numel() == 0:
        return float("nan")
    device = model.device
    ctx = context_ids.to(device).unsqueeze(0)
    suf = suffix_ids.to(device).unsqueeze(0)
    full = torch.cat([ctx, suf], dim=1)
    out = model(input_ids=full, use_cache=False, return_dict=True)
    start = ctx.shape[1]
    logits = out.logits[:, start - 1:start + suf.shape[1] - 1].float()
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), suf.reshape(-1)).item())


@torch.no_grad()
def nll_with_prefix(
    model, context_ids: torch.Tensor, suffix_ids: torch.Tensor, prefix: torch.Tensor
) -> float:
    if suffix_ids.numel() == 0:
        return float("nan")
    device = model.device
    ctx = context_ids.to(device).unsqueeze(0)
    suf = suffix_ids.to(device).unsqueeze(0)
    pref = prefix.to(device=device, dtype=model.dtype).unsqueeze(0)
    emb = model.get_input_embeddings()
    inputs_embeds = torch.cat([pref, emb(ctx), emb(suf)], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        use_cache=False,
        return_dict=True,
    )
    start = pref.shape[1] + ctx.shape[1]
    logits = out.logits[:, start - 1:start + suf.shape[1] - 1].float()
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), suf.reshape(-1)).item())


def finite_mean(rows, key):
    vals = [float(r[key]) for r in rows if torch.isfinite(torch.tensor(float(r[key])))]
    return sum(vals) / len(vals) if vals else None


def select_states(keys, count):
    if count >= len(keys):
        return keys
    if count <= 1:
        return [keys[0]]
    return sorted({keys[round(i * (len(keys) - 1) / (count - 1))] for i in range(count)})


def main():
    args = parse_args()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name != "cuda":
        raise RuntimeError("This evaluation requires CUDA.")
    model, tok = load_model(
        args.model_path,
        torch_dtype(args.dtype),
        args.attn_implementation,
        args.single_gpu,
    )
    device = model.device

    states = torch.load(args.states_path, map_location="cpu").float()
    ae_sd = torch.load(args.ae_ckpt, map_location="cpu")
    enc_weight_keys = sorted(
        (k for k in ae_sd if k.startswith("enc.net.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[2]),
    )
    if not enc_weight_keys:
        raise RuntimeError("could not infer encoder dimensions from AE checkpoint")
    nz = int(ae_sd[enc_weight_keys[-1]].shape[0])
    hidden = int(ae_sd[enc_weight_keys[0]].shape[0])
    depth = len(enc_weight_keys) - 1
    ae = SparseJacobianAutoEncoder(nh=states.shape[1], nz=nz, hidden=hidden, depth=depth)
    ae.load_state_dict(ae_sd)
    ae.to(device).eval()

    mask = torch.load(args.mask_path, map_location="cpu").bool()
    adapter = PrefixAdapter(nz=nz, d=int(model.config.hidden_size), m=1, hidden=2048, depth=2)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location="cpu"))
    adapter.to(device).eval()

    grouped = defaultdict(list)
    with Path(args.contexts_path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                grouped[int(rec["state_index"])].append(rec)
    for recs in grouped.values():
        recs.sort(key=lambda r: int(r["agent_id"]))

    keys = sorted(grouped)
    selected = select_states(keys, args.num_states)
    rows = []
    samples = []

    with torch.no_grad():
        alpha = agreement_alpha(mask.to(device))
        weights = build_weight_vector(alpha, mode="linear").to(device)
        for state_idx in selected:
            recs = grouped[state_idx]
            if len(recs) == 0:
                continue
            h = states[state_idx:state_idx + 1].to(device)
            z = ae.encode(h).squeeze(0)
            z_tilde = route_latents(z, mask.to(device), weights)
            prefixes = adapter(z_tilde)
            for rec in recs:
                context = rec["continuation_context"]
                context_ids = tok(
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_context_tokens,
                ).input_ids[0].cpu()
                agent = int(rec["agent_id"])
                prefix = prefixes[agent]
                base_suffix = greedy_no_prefix(model, tok, context_ids, args.max_new_tokens)
                adapter_suffix = greedy_with_prefix(
                    model, tok, context_ids, prefix, args.max_new_tokens
                )
                base_hidden = hidden_no_prefix(model, context_ids, base_suffix)
                adapter_hidden = hidden_with_prefix(
                    model, context_ids, adapter_suffix, prefix
                )
                base_nll = nll_no_prefix(model, context_ids, base_suffix)
                adapter_nll = nll_with_prefix(
                    model, context_ids, adapter_suffix, prefix
                )
                adapter_suffix_base_nll = nll_no_prefix(
                    model, context_ids, adapter_suffix
                )
                common = min(base_suffix.numel(), adapter_suffix.numel())
                overlap = (
                    float((base_suffix[:common] == adapter_suffix[:common]).float().mean().item())
                    if common else 0.0
                )
                row = {
                    "state_index": state_idx,
                    "agent_id": agent,
                    "example_id": rec.get("example_id"),
                    "context_tokens": int(context_ids.numel()),
                    "base_tokens": int(base_suffix.numel()),
                    "adapter_tokens": int(adapter_suffix.numel()),
                    "token_overlap": overlap,
                    "hidden_cosine": float(
                        F.cosine_similarity(adapter_hidden, base_hidden, dim=0).item()
                    ),
                    "base_nll_on_base": base_nll,
                    "adapter_nll_on_adapter": adapter_nll,
                    "base_nll_on_adapter": adapter_suffix_base_nll,
                    "adapter_minus_base_nll_on_adapter": adapter_nll - adapter_suffix_base_nll,
                }
                rows.append(row)
                if len(samples) < args.sample_texts:
                    samples.append({
                        "state_index": state_idx,
                        "agent_id": agent,
                        "example_id": rec.get("example_id"),
                        "baseline": tok.decode(base_suffix, skip_special_tokens=True),
                        "adapter": tok.decode(adapter_suffix, skip_special_tokens=True),
                    })

    aggregate = {
        "records": len(rows),
        "states_requested": len(selected),
        "states_evaluated": sorted({int(r["state_index"]) for r in rows}),
        "mean_context_tokens": finite_mean(rows, "context_tokens"),
        "mean_hidden_cosine": finite_mean(rows, "hidden_cosine"),
        "mean_token_overlap": finite_mean(rows, "token_overlap"),
        "mean_base_nll_on_base": finite_mean(rows, "base_nll_on_base"),
        "mean_adapter_nll_on_adapter": finite_mean(rows, "adapter_nll_on_adapter"),
        "mean_base_nll_on_adapter": finite_mean(rows, "base_nll_on_adapter"),
        "mean_adapter_minus_base_nll_on_adapter": finite_mean(
            rows, "adapter_minus_base_nll_on_adapter"
        ),
    }
    result = {
        "settings": vars(args),
        "aggregate": aggregate,
        "rows": rows,
        "samples": samples,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
