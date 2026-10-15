"""Generate-and-inspect qualitative eval prompts for SHINRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.generate import generate_chat
from configs.loader import load_config

PROBES = [
    [{"role": "user", "content": "Explain grouped-query attention in one paragraph."}],
    [{"role": "user", "content": "Write a Python function that computes RMSNorm."}],
    [{"role": "user", "content": "Реши: x^2 - 5x + 6 = 0. Покажи шаги."}],
    [{"role": "user", "content": "Return JSON with keys name, hidden_size, layers for SHINRA-4B."}],
]


def main() -> None:
    eval_config = load_config()["evaluation"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=f"{eval_config['report_dir']}/qualitative.json")
    args = parser.parse_args()
    rows = []
    for messages in PROBES:
        text = generate_chat(args.model, messages, max_new_tokens=eval_config["qualitative_max_new_tokens"], temperature=0.0)
        rows.append({"messages": messages, "output": text})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
