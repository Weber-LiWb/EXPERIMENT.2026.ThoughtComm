from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from datasets import load_dataset

from thoughtcomm.core.config import load_config, resolve_path
from thoughtcomm.core.utils import set_seed, ensure_dir
from thoughtcomm.models.hf import load_hf_causal_lm, generate_ids, last_token_state_from_ids
from thoughtcomm.agents import Agent


def load_examples(cfg):
    name = cfg.task.name
    if name in {"gsm8k", "math"}:
        ds = load_dataset(name, split=cfg.task.split)
        out = []
        for ex in ds.select(range(min(len(ds), int(cfg.task.max_examples)))):
            if name == "gsm8k":
                out.append({"id": ex.get("id", None), "problem": ex["question"]})
            else:
                out.append({"id": ex.get("id", None), "problem": ex["problem"]})
        return out
    raise ValueError(f"Unknown task: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_seed(int(cfg.seed))

    out_dir = ensure_dir(args.output_dir)
    (out_dir / "meta").mkdir(exist_ok=True)

    bundle = load_hf_causal_lm(
        resolve_path(cfg.model.path),
        dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16 if cfg.model.dtype == "float16" else torch.float32,
        attn_implementation=str(cfg.model.attn_implementation),
        quantization=cfg.model.quantization,
        device=str(cfg.runtime.device),
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
        compile_model=bool(cfg.model.compile),
    )

    num_agents = int(cfg.agents.num_agents)
    agents = [Agent(role=r) for r in list(cfg.agents.roles)[:num_agents]]

    rounds = int(cfg.agents.rounds)
    examples = load_examples(cfg)

    all_states = []
    contexts_path = out_dir / "contexts.jsonl"

    with contexts_path.open("w", encoding="utf-8") as fctx:
        for ex_i, ex in enumerate(examples):
            all_msgs = []
            for t in range(rounds):
                round_states = []
                round_msgs = []
                for a, ag in enumerate(agents):
                    prompt = ag.build_prompt(ex["problem"], t, all_msgs)
                    out_ids, prompt_len = generate_ids(
                        bundle=bundle,
                        prompt=prompt,
                        max_new_tokens=int(cfg.agents.max_new_tokens),
                        temperature=float(cfg.agents.temperature),
                        top_p=float(cfg.agents.top_p),
                    )
                    gen_text = bundle.tok.decode(out_ids[prompt_len:], skip_special_tokens=True)

                    # Save state of last token for this agent at this round
                    h_a = last_token_state_from_ids(bundle, out_ids)  # (d,)
                    round_states.append(h_a)

                    # Save context for adapter training: prompt + generated (so continuation starts after gen)
                    cont_prompt = prompt + gen_text
                    rec = {
                        "state_index": len(all_states),  # index of the concatenated H_t for this round
                        "example_index": ex_i,
                        "example_id": ex.get("id", None),
                        "round": t,
                        "agent_id": a,
                        "prompt": prompt,
                        "response": gen_text,
                        "continuation_context": cont_prompt,
                    }
                    fctx.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    msg = f"Agent {a+1}: {gen_text}"
                    round_msgs.append(msg)

                all_msgs.extend(round_msgs)
                Ht = torch.cat(round_states, dim=0).cpu().float()  # (nh,)
                all_states.append(Ht)

    states = torch.stack(all_states, dim=0)  # (N_rounds, nh)
    torch.save(states, out_dir / "states.pt")
    print(f"Saved states: {out_dir/'states.pt'}  contexts: {contexts_path}")

if __name__ == "__main__":
    main()
