#!/usr/bin/env python
"""Create a tokenizer-truncated context resource without changing states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    args = ap.parse_args()

    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    rows = []
    for line in (source / "contexts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = tok(
            row["continuation_context"],
            truncation=True,
            max_length=args.max_tokens,
            add_special_tokens=True,
        )["input_ids"]
        row["continuation_context"] = tok.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        rows.append(row)

    (output / "contexts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    states = torch.load(source / "states.pt", map_location="cpu")
    torch.save(states, output / "states.pt")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(source),
                "max_tokens": args.max_tokens,
                "records": len(rows),
                "states_shape": list(states.shape),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} records and states {tuple(states.shape)} to {output}")


if __name__ == "__main__":
    main()
