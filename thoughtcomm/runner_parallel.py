from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.distributed as dist

from .agents import Agent
from .models.hf import HFModelBundle, generate_ids, last_token_state_from_ids
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


def _decode_generated(tok, out_ids: torch.Tensor, prompt_len: int) -> str:
    gen = out_ids[prompt_len:]
    return tok.decode(gen, skip_special_tokens=True)

def run_debate_parallel(
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
) -> Optional[Dict[str, Any]]:
    """Parallel multi-agent debate: one process per agent (rank==agent_id)."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed not initialized; use accelerate/torchrun.")

    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == len(agents), "world_size must equal num_agents (one rank per agent)."

    device = bundle.model.device
    d = bundle.hidden_size
    tok = bundle.tok

    all_round_messages: List[str] = []
    per_round_states: List[torch.Tensor] = []   # only meaningful on rank0

    # prefix for current rank
    prefix_local: Optional[torch.Tensor] = None  # (m,d) on device

    for t in range(rounds):
        agent = agents[rank]
        prompt = agent.build_prompt(problem, t, all_round_messages)

        if thoughtcomm is not None and prefix_local is not None and t >= thoughtcomm.incl_rounds_from:
            out_ids = generate_with_prefix(
                bundle.model, tok, prompt, prefix_local,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
            prompt_len = int(tok(prompt, return_tensors="pt").input_ids.shape[1])
        else:
            out_ids, prompt_len = generate_ids(bundle, prompt, max_new_tokens, temperature, top_p)

        gen_text = _decode_generated(tok, out_ids, prompt_len)
        msg_local = f"Agent {rank+1}: {gen_text}"

        # gather messages
        gathered_msgs = [None for _ in range(world)]
        dist.all_gather_object(gathered_msgs, msg_local)

        # enforce deterministic ordering by rank
        all_round_messages.extend(list(gathered_msgs))

        # gather states
        h_local = last_token_state_from_ids(bundle, out_ids).to(device)  # (d,)
        h_buf = [torch.zeros_like(h_local) for _ in range(world)]
        dist.all_gather(h_buf, h_local)
        Ht = torch.cat(h_buf, dim=0).unsqueeze(0).detach().cpu().float()  # (1, world*d)

        # rank0 computes next prefixes
        if thoughtcomm is not None and t + 1 < rounds:
            if rank == 0:
                with torch.no_grad():
                    H = Ht.to(device)
                    z = thoughtcomm.ae.encode(H).squeeze(0)  # (nz,)
                    mask = thoughtcomm.mask.to(device)
                    alpha = agreement_alpha(mask)
                    w = build_weight_vector(alpha, mode=agreement_weight_mode, custom=custom_weights)
                    z_tilde = route_latents(z, mask, w)  # (A,nz)
                    pref = thoughtcomm.adapter(z_tilde.to(device))  # (A,m,d)
                    pref = pref.detach()
                # scatter each rank its prefix
                scatter_list = [pref[i].contiguous() for i in range(world)]
            else:
                scatter_list = None

            # allocate receive tensor
            m = int(thoughtcomm.adapter.m) if hasattr(thoughtcomm.adapter, "m") else thoughtcomm.adapter(torch.zeros((1, thoughtcomm.mask.shape[1]))).shape[1]
            recv = torch.zeros((thoughtcomm.adapter.m, d), device=device, dtype=torch.float32)
            dist.scatter(recv, scatter_list, src=0)
            prefix_local = recv.to(device)

        if rank == 0:
            per_round_states.append(Ht)

    if rank == 0:
        return {"messages": all_round_messages, "states": per_round_states}
    return None
