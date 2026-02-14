from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch

from .agents import Agent
from .models.hf import HFModelBundle, generate_ids, decode, last_token_state_from_ids
from .thoughts.autoencoder import SparseJacobianAutoEncoder
from .thoughts.routing import agreement_alpha, build_weight_vector, route_latents
from .thoughts.adapter import PrefixAdapter
from .thoughts.prefix_inject import generate_with_prefix


@dataclass
class ThoughtCommModules:
    incl_rounds_from: int
    ae: SparseJacobianAutoEncoder
    mask: torch.Tensor  # (A, nz) bool on CPU
    adapter: PrefixAdapter


def _decode_generated(bundle: HFModelBundle, out_ids: torch.Tensor, prompt_len: int) -> str:
    gen = out_ids[prompt_len:]
    return bundle.tok.decode(gen, skip_special_tokens=True)

def run_debate_sequential(
    bundle: HFModelBundle,
    agents: List[Agent],
    problem: str,
    rounds: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    thoughtcomm: Optional[ThoughtCommModules] = None,
    agreement_weight_mode: str = "linear",
    custom_weights: Optional[dict] = None,
) -> Dict[str, Any]:
    """Sequential multi-agent debate with optional THOUGHTCOMM prefix injection."""
    device = bundle.model.device
    A = len(agents)

    all_round_messages: List[str] = []
    per_round_states: List[torch.Tensor] = []
    per_round_meta: List[Dict[str, Any]] = []

    prev_H: Optional[torch.Tensor] = None  # (1, nh) on CPU

    for t in range(rounds):
        round_msgs = []
        round_states = []
        round_info = {"round": t, "agents": []}

        # If THOUGHTCOMM, compute all prefixes for this round based on prev_H.
        prefixes: Optional[torch.Tensor] = None  # (A, m, d)
        if thoughtcomm is not None and prev_H is not None and t >= thoughtcomm.incl_rounds_from:
            H = prev_H.to(device)  # (1, nh)
            with torch.no_grad():
                z = thoughtcomm.ae.encode(H).squeeze(0)  # (nz,)
                mask = thoughtcomm.mask.to(device)
                alpha = agreement_alpha(mask)
                w = build_weight_vector(alpha, mode=agreement_weight_mode, custom=custom_weights)
                z_tilde = route_latents(z, mask, w)  # (A, nz)
                prefixes = thoughtcomm.adapter(z_tilde.to(device))  # (A, m, d)

        for i, ag in enumerate(agents):
            prompt = ag.build_prompt(problem, t, all_round_messages)

            if prefixes is not None:
                out_ids = generate_with_prefix(
                    bundle.model, bundle.tok, prompt,
                    prefix=prefixes[i],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # Need prompt_len to split
                prompt_ids = bundle.tok(prompt, return_tensors="pt").input_ids[0]
                prompt_len = int(prompt_ids.shape[0])
            else:
                out_ids, prompt_len = generate_ids(
                    bundle=bundle,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            gen_text = _decode_generated(bundle, out_ids, prompt_len)
            msg = f"Agent {i+1}: {gen_text}"
            round_msgs.append(msg)

            # state = last token hidden state of full sequence (prompt+gen)
            h_i = last_token_state_from_ids(bundle, out_ids)  # (d,)
            round_states.append(h_i)

            round_info["agents"].append({
                "agent_id": i,
                "prompt": prompt,
                "generated": gen_text,
            })

        all_round_messages.extend(round_msgs)
        Ht = torch.cat(round_states, dim=0).unsqueeze(0).detach().cpu().float()  # (1, nh)
        per_round_states.append(Ht)
        per_round_meta.append(round_info)
        prev_H = Ht

    return {
        "messages": all_round_messages,
        "states": per_round_states,
        "meta": per_round_meta,
    }
