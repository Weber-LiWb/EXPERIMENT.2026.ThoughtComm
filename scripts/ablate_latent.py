from __future__ import annotations
import argparse
import torch

from thoughtcomm.core.config import resolve_path
from thoughtcomm.thoughts.autoencoder import SparseJacobianAutoEncoder
from thoughtcomm.thoughts.adapter import PrefixAdapter
from thoughtcomm.thoughts.routing import agreement_alpha, build_weight_vector, route_latents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--adapter_ckpt", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--state", required=True, help="path to a single H vector saved with torch.save (nh,)")
    ap.add_argument("--latent", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prefix_len", type=int, default=1)
    ap.add_argument("--hidden_size", type=int, required=True)
    ap.add_argument("--num_agents", type=int, required=True)
    args = ap.parse_args()

    H = torch.load(resolve_path(args.state), map_location="cpu").float().unsqueeze(0)
    nh = H.shape[1]
    nz = None

    ae_sd = torch.load(resolve_path(args.ae_ckpt), map_location="cpu")
    # infer nz from decoder last layer
    for k, v in ae_sd.items():
        if k.endswith("enc.net.0.weight"):
            nz = v.shape[0]
            break
    if nz is None:
        raise ValueError("Could not infer nz from AE checkpoint.")

    ae = SparseJacobianAutoEncoder(nh=nh, nz=nz)
    ae.load_state_dict(ae_sd, strict=False)
    ae.to(args.device).eval()

    adapter = PrefixAdapter(nz=nz, d=args.hidden_size, m=args.prefix_len)
    adapter.load_state_dict(torch.load(resolve_path(args.adapter_ckpt), map_location="cpu"), strict=False)
    adapter.to(args.device).eval()

    mask = torch.load(resolve_path(args.mask), map_location="cpu").bool()[:args.num_agents]
    alpha = agreement_alpha(mask.to(args.device))
    w = build_weight_vector(alpha).to(args.device)

    with torch.no_grad():
        z = ae.encode(H.to(args.device)).squeeze(0)
        z_tilde = route_latents(z, mask.to(args.device), w)
        p0 = adapter(z_tilde).cpu()

        z2 = z.clone()
        z2[int(args.latent)] = 0.0
        z2_tilde = route_latents(z2, mask.to(args.device), w)
        p1 = adapter(z2_tilde).cpu()

    diff = (p0 - p1).norm(dim=-1).mean(dim=-1)  # per agent
    print("Mean prefix change (L2 per token) per agent after zeroing latent:", diff.tolist())

if __name__ == "__main__":
    main()
