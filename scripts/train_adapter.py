from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from thoughtcomm.core.config import load_config, resolve_path
from thoughtcomm.core.utils import set_seed, ensure_dir
from thoughtcomm.core.hash import sha1_text
from thoughtcomm.models.hf import load_hf_causal_lm, nll_of_suffix_with_prefix
from thoughtcomm.thoughts.autoencoder import SparseJacobianAutoEncoder
from thoughtcomm.thoughts.routing import agreement_alpha, build_weight_vector, route_latents
from thoughtcomm.thoughts.adapter import PrefixAdapter


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    # logits: (V,)
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum <= top_p
    # ensure at least one token
    mask[0] = True
    filt_idx = sorted_idx[mask]
    filt_probs = sorted_probs[mask]
    filt_probs = filt_probs / filt_probs.sum()
    sampled = torch.multinomial(filt_probs, 1)
    return int(filt_idx[sampled].item())


@torch.no_grad()
def generate_suffix_no_prefix(model, tok, context_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float) -> torch.Tensor:
    # Use generate for baseline reference
    inp = context_ids.unsqueeze(0).to(model.device)
    out = model.generate(
        input_ids=inp,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=False,
    )[0]
    return out[inp.shape[1]:].detach().cpu()


@torch.no_grad()
def generate_suffix_with_prefix(
    model,
    tok,
    context_ids: torch.Tensor,   # (L,)
    prefix: torch.Tensor,        # (m,d)
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Manual autoregressive generation with prefix, returning suffix token ids (CPU)."""
    device = model.device
    emb = model.get_input_embeddings()

    ctx = context_ids.to(device).unsqueeze(0)  # (1,L)
    ctx_emb = emb(ctx)  # (1,L,d)

    pref = prefix.to(device).unsqueeze(0)  # (1,m,d)
    inputs_embeds = torch.cat([pref, ctx_emb], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_logits = out.logits[0, -1, :]  # last position predicts next token

    gen = []
    for _ in range(max_new_tokens):
        token = top_p_sample(next_logits, top_p=top_p, temperature=temperature)
        gen.append(token)
        inp = torch.tensor([[token]], device=device, dtype=torch.long)
        out = model(input_ids=inp, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]
        if tok.eos_token_id is not None and token == tok.eos_token_id:
            break
    return torch.tensor(gen, dtype=torch.long)


@torch.no_grad()
def mean_hidden_no_prefix(model, context_ids: torch.Tensor, suffix_ids: torch.Tensor) -> torch.Tensor:
    device = model.device
    full = torch.cat([context_ids.to(device), suffix_ids.to(device)], dim=0).unsqueeze(0)
    out = model(input_ids=full, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states[-1][0, context_ids.shape[0]:, :]
    return hs.mean(dim=0).detach()


def mean_hidden_with_prefix(model, context_ids: torch.Tensor, suffix_ids: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
    device = model.device
    emb = model.get_input_embeddings()

    ctx = context_ids.to(device).unsqueeze(0)
    suf = suffix_ids.to(device).unsqueeze(0)
    pref = prefix.to(device).unsqueeze(0)

    ctx_emb = emb(ctx)
    suf_emb = emb(suf)
    inputs_embeds = torch.cat([pref, ctx_emb, suf_emb], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

    out = model(inputs_embeds=inputs_embeds, attention_mask=attn,
                output_hidden_states=True, use_cache=False, return_dict=True)
    m = pref.shape[1]
    L = ctx.shape[1]
    hs = out.hidden_states[-1][0, m+L:, :]
    return hs.mean(dim=0)


def load_grouped_contexts(contexts_path: Path, num_agents: int) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for line in contexts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        grouped.setdefault(int(rec["state_index"]), []).append(rec)

    # sort per state by agent_id
    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda r: int(r["agent_id"]))
        if len(grouped[k]) != num_agents:
            # allow partial but warn by leaving as-is
            pass
    return grouped


def get_ref_suffix_cached(cache_dir: Path, key_text: str, gen_fn) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = sha1_text(key_text)
    p = cache_dir / f"{key}.pt"
    if p.exists():
        return torch.load(p, map_location="cpu")
    suf = gen_fn()
    torch.save(suf, p)
    return suf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--contexts_path", required=True)
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--states_path", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_seed(int(cfg.seed))
    out_dir = ensure_dir(args.output_dir)

    contexts_path = Path(args.contexts_path)
    states_path = Path(args.states_path) if args.states_path else contexts_path.parent / "states.pt"
    states = torch.load(resolve_path(str(states_path)), map_location="cpu").float()  # (N, nh)

    bundle = load_hf_causal_lm(
        resolve_path(cfg.model.path),
        dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16 if cfg.model.dtype == "float16" else torch.float32,
        attn_implementation=str(cfg.model.attn_implementation),
        quantization=cfg.model.quantization,
        device=str(cfg.runtime.device),
        gradient_checkpointing=False,
        compile_model=False,
    )
    model, tok = bundle.model, bundle.tok

    # Freeze LM params (train adapter only)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    num_agents = int(cfg.agents.num_agents)
    nz = int(cfg.thoughtcomm.nz)
    nh = states.shape[1]
    hidden_size = nh // num_agents

    ae = SparseJacobianAutoEncoder(nh=nh, nz=nz, hidden=int(cfg.autoencoder.hidden),
                                   depth=int(cfg.autoencoder.depth), dropout=float(cfg.autoencoder.dropout))
    ae.load_state_dict(torch.load(resolve_path(args.ae_ckpt), map_location="cpu"))
    ae.to(model.device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    mask = torch.load(resolve_path(args.mask_path), map_location="cpu").bool()
    mask = mask[:num_agents]  # safety

    alpha = agreement_alpha(mask.to(model.device))
    w = build_weight_vector(alpha, mode=str(cfg.thoughtcomm.agreement_weights),
                            custom=dict(cfg.thoughtcomm.custom_weights) if cfg.thoughtcomm.custom_weights else None).to(model.device)

    adapter = PrefixAdapter(nz=nz, d=hidden_size, m=int(cfg.thoughtcomm.prefix_len),
                            hidden=int(cfg.adapter.hidden), depth=int(cfg.adapter.depth)).to(model.device)
    adapter.train()

    opt = torch.optim.AdamW(adapter.parameters(), lr=float(cfg.adapter.lr))

    grouped = load_grouped_contexts(contexts_path, num_agents=num_agents)
    keys = list(grouped.keys())
    cache_dir = Path(cfg.adapter.cache_dir)

    max_new = int(cfg.adapter.continuation_max_new_tokens)
    temp = float(cfg.adapter.continuation_temperature)
    top_p = float(cfg.adapter.continuation_top_p)

    lambda_sem = float(cfg.adapter.lambda_sem)
    lambda_flu = float(cfg.adapter.lambda_flu)

    grad_accum = int(cfg.adapter.grad_accum)
    opt.zero_grad(set_to_none=True)

    step = 0
    for ep in range(int(cfg.adapter.epochs)):
        pbar = tqdm(keys, desc=f"adapter epoch {ep}")
        for state_idx in pbar:
            recs = grouped[state_idx]
            if len(recs) < num_agents:
                continue

            H = states[state_idx:state_idx+1].to(model.device)  # (1, nh)
            with torch.no_grad():
                z = ae.encode(H).squeeze(0)  # (nz,)
                z_tilde = route_latents(z, mask.to(model.device), w)  # (A,nz)
            prefixes = adapter(z_tilde)  # (A,m,d)

            loss_sum = 0.0
            for a in range(num_agents):
                ctx_text = recs[a]["continuation_context"]
                ctx_ids = tok(ctx_text, return_tensors="pt", truncation=True, max_length=4096).input_ids[0].cpu()

                def gen_ref():
                    return generate_suffix_no_prefix(model, tok, ctx_ids, max_new, temp, top_p)

                y_ref = get_ref_suffix_cached(cache_dir, ctx_text, gen_ref)
                y_gen = generate_suffix_with_prefix(model, tok, ctx_ids, prefixes[a].to(model.dtype), max_new, temp, top_p).cpu()

                # semantic term: 1 - cosine(mean_hidden(prefix-run), mean_hidden(baseline))
                ref_emb = mean_hidden_no_prefix(model, ctx_ids.to(model.device), y_ref.to(model.device))
                gen_emb = mean_hidden_with_prefix(model, ctx_ids, y_gen, prefixes[a].to(model.dtype))
                sem = 1.0 - F.cosine_similarity(gen_emb, ref_emb, dim=0)

                # fluency term: -log p(y_gen | context, prefix)
                flu = nll_of_suffix_with_prefix(model, tok, ctx_ids, y_gen, prefixes[a].to(model.dtype))

                loss_sum = loss_sum + (lambda_sem * sem + lambda_flu * flu)

            loss = loss_sum / num_agents
            (loss / grad_accum).backward()

            step += 1
            if step % grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=float(loss.item()), sem=float(sem.item()), flu=float(flu.item()))

        # checkpoint each epoch
        torch.save(adapter.state_dict(), out_dir / f"adapter_epoch{ep+1}.pt")

    torch.save(adapter.state_dict(), out_dir / "adapter.pt")
    print(f"Saved adapter to {out_dir/'adapter.pt'}")

if __name__ == "__main__":
    main()
