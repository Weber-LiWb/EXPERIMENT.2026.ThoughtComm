from __future__ import annotations
import argparse
from pathlib import Path
import torch

from thoughtcomm.core.config import load_config, resolve_path
from thoughtcomm.core.utils import set_seed, ensure_dir
from thoughtcomm.thoughts.autoencoder import SparseJacobianAutoEncoder, estimate_structure_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ae_ckpt", dest="ae_ckpt", default=None)
    ap.add_argument("--states_path", dest="states_path", default=None)
    ap.add_argument("--num_agents", type=int, default=None)
    ap.add_argument("--hidden_size", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_seed(int(cfg.seed))
    out_dir = ensure_dir(args.output_dir)

    ae_ckpt = args.ae_ckpt or cfg.ae.ckpt_path if "ae" in cfg and "ckpt_path" in cfg.ae else None
    if ae_ckpt is None:
        raise ValueError("Provide --ae_ckpt or set ae.ckpt_path")
    states_path = args.states_path or cfg.data.states_path if "data" in cfg and "states_path" in cfg.data else None
    if states_path is None:
        raise ValueError("Provide --states_path or set data.states_path")

    states = torch.load(resolve_path(states_path), map_location="cpu").float()
    N, nh = states.shape

    num_agents = args.num_agents or int(cfg.agents.num_agents)
    if args.hidden_size is not None:
        hidden_size = args.hidden_size
    else:
        assert nh % num_agents == 0, "nh must be divisible by num_agents to infer hidden_size."
        hidden_size = nh // num_agents

    nz = int(cfg.thoughtcomm.nz)

    ae = SparseJacobianAutoEncoder(nh=nh, nz=nz, hidden=int(cfg.autoencoder.hidden),
                                   depth=int(cfg.autoencoder.depth), dropout=float(cfg.autoencoder.dropout))
    ae.load_state_dict(torch.load(resolve_path(ae_ckpt), map_location="cpu"))
    ae.to(str(cfg.runtime.device))

    mask = estimate_structure_mask(
        ae=ae,
        states=states,
        num_agents=num_agents,
        hidden_size=hidden_size,
        threshold=float(cfg.autoencoder.mask_threshold),
        rows_per_sample=int(cfg.autoencoder.jacobian_rows_k),
        max_samples=int(args.max_samples) if args.max_samples else None,
    )
    torch.save(mask, out_dir / "mask.pt")
    print(f"Saved mask: {out_dir/'mask.pt'}  shape={tuple(mask.shape)}")

if __name__ == "__main__":
    main()
