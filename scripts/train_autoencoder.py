from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from thoughtcomm.core.config import load_config, resolve_path
from thoughtcomm.core.utils import set_seed, ensure_dir
from thoughtcomm.thoughts.autoencoder import SparseJacobianAutoEncoder, jacobian_l1_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--states_path", dest="states_path", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_seed(int(cfg.seed))
    out_dir = ensure_dir(args.output_dir)

    states_path = args.states_path or cfg.data.states_path if "data" in cfg and "states_path" in cfg.data else None
    if states_path is None:
        raise ValueError("Provide --states_path or set data.states_path in config.")
    states = torch.load(resolve_path(states_path), map_location="cpu").float()  # (N, nh)

    N, nh = states.shape
    nz = int(cfg.thoughtcomm.nz)
    device = str(cfg.runtime.device)

    ae = SparseJacobianAutoEncoder(
        nh=nh, nz=nz,
        hidden=int(cfg.autoencoder.hidden),
        depth=int(cfg.autoencoder.depth),
        dropout=float(cfg.autoencoder.dropout),
    ).to(device)

    ds = TensorDataset(states)
    dl = DataLoader(ds, batch_size=int(cfg.autoencoder.batch_size), shuffle=True, pin_memory=True)

    opt = torch.optim.AdamW(ae.parameters(), lr=float(cfg.autoencoder.lr))
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in device and (cfg.model.dtype in ["float16", "bfloat16"])))

    best = 1e30
    global_step = 0
    ae.train()

    for ep in range(int(cfg.autoencoder.epochs)):
        pbar = tqdm(dl, desc=f"ae epoch {ep}")
        opt.zero_grad(set_to_none=True)

        for step, (h,) in enumerate(pbar):
            h = h.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=("cuda" in device and cfg.model.dtype in ["float16", "bfloat16"])):
                z, h_hat = ae(h)
                rec = torch.mean((h - h_hat) ** 2)

                jac = jacobian_l1_penalty(
                    ae, z,
                    mode=str(cfg.autoencoder.jacobian_mode),
                    sampled_rows_k=int(cfg.autoencoder.jacobian_rows_k),
                )
                loss = rec + float(cfg.autoencoder.jacobian_lambda) * jac

            scaler.scale(loss / int(cfg.autoencoder.grad_accum)).backward()

            if (step + 1) % int(cfg.autoencoder.grad_accum) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            if global_step % int(cfg.runtime.log_every) == 0:
                pbar.set_postfix(rec=float(rec.item()), jac=float(jac.item()), loss=float(loss.item()))

        # save checkpoint
        if (ep + 1) % int(cfg.autoencoder.save_every_epochs) == 0:
            ckpt = out_dir / f"ae_epoch{ep+1}.pt"
            torch.save(ae.state_dict(), ckpt)

        # best
        if float(loss.item()) < best:
            best = float(loss.item())
            torch.save(ae.state_dict(), out_dir / "ae.pt")

    print(f"Saved best AE to {out_dir/'ae.pt'}")

if __name__ == "__main__":
    main()
