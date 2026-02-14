"""Main CLI entry point for running ThoughtComm debates."""

# pylint: disable=E0401

from __future__ import annotations
import argparse
import json
from pathlib import Path
from datasets import load_dataset
import torch
import torch.distributed as dist

from thoughtcomm.core.config import load_config, resolve_path
from thoughtcomm.core.utils import set_seed
from thoughtcomm.core.run_dir import make_run_dir
from thoughtcomm.core.logging import setup_logger
from thoughtcomm.models.hf import load_hf_causal_lm
from thoughtcomm.agents import Agent
from thoughtcomm.runner import run_debate_sequential, ThoughtCommModules as TCSeq
from thoughtcomm.runner_parallel import run_debate_parallel, ThoughtCommModules as TCPar
from thoughtcomm.thoughts.autoencoder import SparseJacobianAutoEncoder
from thoughtcomm.thoughts.adapter import PrefixAdapter


def load_task_examples(cfg):
    """Load task examples based on the configuration."""
    name = cfg.task.name
    if name in {"gsm8k", "math"}:
        ds = load_dataset(name, "main")
        ds = ds[cfg.task.split]

        examples = []
        for ex in ds.select(range(min(len(ds), int(cfg.task.max_examples)))):
            if name == "gsm8k":
                examples.append(ex["question"])
            else:
                examples.append(ex["problem"])
        return examples
    if name == "custom_jsonl":
        path = Path(cfg.task.custom_path)
        return [
                    json.loads(x)["prompt"]
                    for x in path.read_text(encoding="utf-8").splitlines() if x.strip()
                ]
    raise ValueError(f"Unknown task: {name}")


def _load_thoughtcomm_modules(cfg, num_agents: int, hidden_size: int):
    if not cfg.thoughtcomm.enabled:
        return None
    if not (
        cfg.thoughtcomm.ae_ckpt and
        cfg.thoughtcomm.struct_mask and
        cfg.thoughtcomm.adapter_ckpt
    ):
        return None

    device = cfg.runtime.device
    nz = int(cfg.thoughtcomm.nz)
    nh = int(num_agents * hidden_size)

    ae = SparseJacobianAutoEncoder(
                                    nh=nh,
                                    nz=nz,
                                    hidden=int(cfg.autoencoder.hidden),
                                    depth=int(cfg.autoencoder.depth),
                                    dropout=float(cfg.autoencoder.dropout)
                                )
    ae.load_state_dict(torch.load(resolve_path(cfg.thoughtcomm.ae_ckpt), map_location="cpu"))
    ae.to(device).eval()

    mask = torch.load(resolve_path(cfg.thoughtcomm.struct_mask), map_location="cpu").bool()

    adapter = PrefixAdapter(nz=nz, d=hidden_size, m=int(cfg.thoughtcomm.prefix_len),
                            hidden=int(cfg.adapter.hidden), depth=int(cfg.adapter.depth))
    adapter.load_state_dict(torch.load(
                                        resolve_path(cfg.thoughtcomm.adapter_ckpt),
                                        map_location="cpu"
                                    )
                            )
    adapter.to(device).eval()

    return {"ae": ae, "mask": mask, "adapter": adapter}


def main():
    """Main entry point for running ThoughtComm debates."""
    # --------------------------------------------------------------------
    # Load config and set up logging
    # --------------------------------------------------------------------
    args = argparse.ArgumentParser()
    args.add_argument("--default_path", required=False, default="configs/default.yaml")
    args.add_argument("--override_path", required=True)
    args = args.parse_args()

    cfg = load_config(args.default_path, args.override_path)
    set_seed(int(cfg.seed))

    run_dir = make_run_dir(cfg)
    logger = setup_logger(str(run_dir / "run.log"))

    model_path = resolve_path(cfg.model.path)
    if not model_path:
        raise ValueError("Set model.path to a local HF model directory.")

    bundle = load_hf_causal_lm(
        model_path,
        dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16 if cfg.model.dtype == "float16" else torch.float32,
        attn_implementation=str(cfg.model.attn_implementation),
        quantization=cfg.model.quantization,
        device=str(cfg.runtime.device),
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
        compile_model=bool(cfg.model.compile),
    )

    num_agents = int(cfg.agents.num_agents)
    roles = list(cfg.agents.roles)[:num_agents]
    agents = [Agent(role=r) for r in roles]

    tc = _load_thoughtcomm_modules(cfg, num_agents=num_agents, hidden_size=bundle.hidden_size)
    modules = None
    if tc is not None:
        if cfg.runtime.mode == "parallel":
            modules = TCPar(incl_rounds_from=1, ae=tc["ae"], mask=tc["mask"], adapter=tc["adapter"])
        else:
            modules = TCSeq(incl_rounds_from=1, ae=tc["ae"], mask=tc["mask"], adapter=tc["adapter"])

    print(cfg.task)
    problems = load_task_examples(cfg)
    logger.info(f"Loaded {len(problems)} examples. mode={cfg.runtime.mode}")

    results = []
    for idx, problem in enumerate(problems):
        if cfg.runtime.mode == "parallel":
            out = run_debate_parallel(
                bundle=bundle,
                agents=agents,
                problem=problem,
                rounds=int(cfg.agents.rounds),
                max_new_tokens=int(cfg.agents.max_new_tokens),
                temperature=float(cfg.agents.temperature),
                top_p=float(cfg.agents.top_p),
                thoughtcomm=modules,
                agreement_weight_mode=str(cfg.thoughtcomm.agreement_weights),
                custom_weights=dict(cfg.thoughtcomm.custom_weights) if cfg.thoughtcomm.custom_weights else None,
            )
            if out is None:
                continue
        else:
            out = run_debate_sequential(
                bundle=bundle,
                agents=agents,
                problem=problem,
                rounds=int(cfg.agents.rounds),
                max_new_tokens=int(cfg.agents.max_new_tokens),
                temperature=float(cfg.agents.temperature),
                top_p=float(cfg.agents.top_p),
                thoughtcomm=modules,
                agreement_weight_mode=str(cfg.thoughtcomm.agreement_weights),
                custom_weights=dict(cfg.thoughtcomm.custom_weights) if cfg.thoughtcomm.custom_weights else None,
            )

        results.append(out)
        if (not torch.distributed.is_initialized()) or dist.get_rank() == 0:
            logger.info("="*80)
            logger.info(f"Example {idx}")
            logger.info("\n".join(out["messages"][-num_agents:]))

    if (not torch.distributed.is_initialized()) or dist.get_rank() == 0:
        (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved results to {run_dir/'results.json'}")


if __name__ == "__main__":
    main()
