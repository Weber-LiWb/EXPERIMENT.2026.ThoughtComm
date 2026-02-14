from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _device_map_for(device: str):
    # For bitsandbytes quantized models, passing device_map is recommended.
    # device could be "cuda" or "cuda:0".
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        idx = int(device.split(":")[1])
        return {"": idx}
    return {"": 0}

@dataclass
class HFModelBundle:
    model: AutoModelForCausalLM
    tok: AutoTokenizer
    hidden_size: int

def load_hf_causal_lm(
    path: str,
    dtype: torch.dtype,
    attn_implementation: str = "sdpa",
    quantization: Optional[str] = None,
    device: str = "cuda",
    gradient_checkpointing: bool = False,
    compile_model: bool = False,
) -> HFModelBundle:
    """Load a *local* Hugging Face CausalLM (weights must exist on disk)."""
    tok = AutoTokenizer.from_pretrained(path,  trust_remote_code=True,    local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: Dict[str, Any] = dict(
        local_files_only=True,
        attn_implementation=attn_implementation,
    )

    if quantization in {"8bit", "4bit"}:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            raise RuntimeError("bitsandbytes not installed but quantization requested.") from e
        kwargs["device_map"] = _device_map_for(device)
        kwargs["load_in_8bit"] = quantization == "8bit"
        kwargs["load_in_4bit"] = quantization == "4bit"
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if quantization not in {"8bit", "4bit"}:
        model.to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.eval()

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore

    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model config.")
    return HFModelBundle(model=model, tok=tok, hidden_size=int(hidden_size))

@torch.no_grad()
def generate_ids(
    bundle: HFModelBundle,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, int]:
    """Returns (output_ids, prompt_len)."""
    tok, model = bundle.tok, bundle.model
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=False,
    )
    return out[0], prompt_len

@torch.no_grad()
def decode(bundle: HFModelBundle, ids: torch.Tensor) -> str:
    return bundle.tok.decode(ids, skip_special_tokens=True)

@torch.no_grad()
def last_token_state_from_ids(bundle: HFModelBundle, ids: torch.Tensor) -> torch.Tensor:
    """Last-layer hidden state of last token for a given token sequence."""
    tok, model = bundle.tok, bundle.model
    ids = ids.unsqueeze(0).to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states[-1]  # (1, L, d)
    return hs[0, -1, :].detach().float()

@torch.no_grad()
def mean_hidden_of_suffix(
    bundle: HFModelBundle,
    full_ids: torch.Tensor,
    suffix_start: int,
) -> torch.Tensor:
    r"""Mean last-layer hidden state over tokens in [suffix_start:].

    Used as \bar\phi(y) in Eq.(12) (contextual token embeddings).
    """
    model = bundle.model
    ids = full_ids.unsqueeze(0).to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states[-1][0, suffix_start:, :]  # (S, d)
    return hs.mean(dim=0)

def nll_of_suffix_with_prefix(
    model,
    tok,
    context_ids: torch.Tensor,     # (L,)
    suffix_ids: torch.Tensor,      # (S,)
    prefix: torch.Tensor,          # (m, d)
) -> torch.Tensor:
    """Compute -log p(suffix | context, prefix) with teacher forcing.

    Returns scalar NLL averaged over suffix tokens.
    This is differentiable w.r.t. prefix (adapter params).
    """
    device = model.device
    prefix = prefix.to(device).unsqueeze(0)  # (1, m, d)

    ctx = context_ids.to(device).unsqueeze(0)  # (1, L)
    suf = suffix_ids.to(device).unsqueeze(0)   # (1, S)

    # Build embeddings: [prefix, embed(ctx), embed(suf)]
    emb = model.get_input_embeddings()
    ctx_emb = emb(ctx)
    suf_emb = emb(suf)
    inputs_embeds = torch.cat([prefix, ctx_emb, suf_emb], dim=1)  # (1, m+L+S, d)

    # Labels for LM loss: predict next token; ignore prefix+context positions.
    # For suffix positions, we want to predict suffix token itself shifted by 1.
    # We implement explicit cross-entropy on logits corresponding to suffix tokens.
    attn = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False, return_dict=True)
    logits = out.logits  # (1, T, V), T=m+L+S

    # For position t, logits[t] predicts token at t (next-token convention differs).
    # With inputs_embeds, Transformers uses standard CausalLM: logits at position i predict token i+1.
    # So to score suffix token j at absolute position pos = m+L + j, we use logits at pos-1.
    m = prefix.shape[1]
    L = ctx.shape[1]
    S = suf.shape[1]
    start = m + L  # first suffix token position in inputs_embeds

    # Collect logits that predict each suffix token: positions [start-1 : start+S-2]
    pred_logits = logits[:, start-1 : start+S-1, :]  # (1, S, V)
    target = suf  # (1, S)

    loss = torch.nn.functional.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        target.reshape(-1),
        reduction="mean",
    )
    return loss
