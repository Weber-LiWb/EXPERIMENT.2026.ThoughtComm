from __future__ import annotations
import torch

@torch.no_grad()
def generate_with_prefix(
    model,
    tok,
    prompt: str,
    prefix: torch.Tensor,  # (m, d)
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    device = model.device
    prefix = prefix.to(device).unsqueeze(0)  # (1,m,d)

    inputs = tok(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]

    emb = model.get_input_embeddings()(input_ids)  # (1,L,d)
    inputs_embeds = torch.cat([prefix, emb], dim=1)
    attention_mask = torch.cat([torch.ones((1, prefix.shape[1]), device=device, dtype=attn.dtype), attn], dim=1)

    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=False,
    )
    return out[0]
