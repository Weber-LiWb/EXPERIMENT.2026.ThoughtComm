from __future__ import annotations
import argparse
import torch
import numpy as np

from thoughtcomm.thoughts.routing import agreement_alpha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True)
    args = ap.parse_args()

    mask = torch.load(args.mask, map_location="cpu").bool()
    alpha = agreement_alpha(mask).numpy()
    vals, cnt = np.unique(alpha, return_counts=True)
    print("Agreement histogram (alpha -> count):")
    for v, c in zip(vals, cnt):
        print(f"  {int(v)} -> {int(c)}")
    top = np.argsort(-alpha)[:20]
    print("Top-20 most shared latent dims:", top.tolist())
    print("Their alpha:", alpha[top].tolist())

if __name__ == "__main__":
    main()
