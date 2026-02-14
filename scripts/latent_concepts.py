from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import numpy as np

from thoughtcomm.thoughts.autoencoder import SparseJacobianAutoEncoder
from thoughtcomm.core.config import resolve_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--latent", type=int, required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    states = torch.load(resolve_path(args.states), map_location="cpu").float()
    N, nh = states.shape

    # infer nz from ckpt shapes
    sd = torch.load(resolve_path(args.ae_ckpt), map_location="cpu")
    nz = sd["enc.net.0.weight"].shape[0] if "enc.net.0.weight" in sd else sd[next(iter(sd))].shape[0]

    ae = SparseJacobianAutoEncoder(nh=nh, nz=nz)
    ae.load_state_dict(sd, strict=False)
    ae.to(args.device).eval()

    with torch.no_grad():
        z = ae.encode(states.to(args.device)).cpu().numpy()  # (N,nz)

    j = int(args.latent)
    scores = np.abs(z[:, j])
    top = np.argsort(-scores)[: args.topk]

    # Map state_index -> few contexts (agent-level)
    grouped = {}
    for line in Path(args.contexts).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        grouped.setdefault(int(rec["state_index"]), []).append(rec)

    print(f"Top-{args.topk} contexts for |z[{j}]|:")
    for idx in top:
        print("=" * 80)
        print(f"state_index={idx}  |z|={scores[idx]:.4f}")
        for rec in sorted(grouped.get(int(idx), []), key=lambda r: int(r["agent_id"])):
            print(f"- agent {rec['agent_id']}:")
            # show tail of context for readability
            ctx = rec["continuation_context"]
            tail = ctx[-800:] if len(ctx) > 800 else ctx
            print(tail.replace("\n", " ")[:800])

if __name__ == "__main__":
    main()
