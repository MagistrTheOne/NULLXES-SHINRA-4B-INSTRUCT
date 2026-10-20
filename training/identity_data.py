"""Strict identity-only admission and answer-supervised rows for the shared trainer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.identity_stage import VERSION, digest, records
from data.hashing import sha256_file
from tokenizer.contract import encode_chat, encode_text


def read_identity(root: Path, configured_sha: str):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if digest(body) != manifest.get("manifest_sha256") or manifest["manifest_sha256"] != configured_sha:
        raise ValueError("identity corpus manifest SHA mismatch")
    if manifest.get("schema") != VERSION or manifest.get("stage") != "identity-only":
        raise ValueError("identity-only trainer requires the dedicated Stage 1 corpus")
    splits = {}
    for split in ("train", "validation"):
        path = root / f"{split}.jsonl"
        if sha256_file(path) != manifest["files"].get(path.name):
            raise ValueError(f"identity {split} file SHA mismatch")
        with path.open(encoding="utf-8") as stream:
            actual = [json.loads(line) for line in stream]
        expected = list(records(split))
        # Membership, ordering, provenance and canonical answers are all enforced.
        if actual != expected:
            raise ValueError(f"{split}: records differ from the authorized identity-only dataset")
        if manifest["record_counts"].get(split) != len(actual):
            raise ValueError(f"{split}: manifest count differs from records")
        splits[split] = actual
    return manifest, splits


class IdentityDataset:
    """One document per row; only assistant answer and EOT receive loss."""

    def __init__(self, rows, tokenizer, config, corpus_sha, bundle_sha, include_row_indices=False):
        self.rows = []
        self.include_row_indices = include_row_indices
        limit = int(config["training"]["sequence_length"])
        pad = int(config["tokenizer"]["pad_id"])
        eot = int(config["tokenizer"]["end_of_turn_id"])
        languages, targets = [], []
        for row in rows:
            prefix = encode_chat(tokenizer, [row["messages"][0]], limit, config)
            answer = encode_text(tokenizer, row["answer"], config) + [eot]
            ids = prefix + answer
            rendered = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False
            )
            if ids != rendered:
                raise ValueError("identity chat template does not match prompt/answer token boundary")
            if len(ids) > limit:
                raise ValueError("identity example exceeds sequence length; truncation is forbidden")
            padding = limit - len(ids)
            self.rows.append({
                "input_ids": ids + [pad] * padding,
                "attention_mask": [1] * len(ids) + [0] * padding,
                "labels": [-100] * len(prefix) + answer + [-100] * padding,
            })
            languages.append(0 if row["language"] == "en" else 1)
            targets.append(len(answer))
        self.language_codes = np.asarray(languages, dtype=np.uint8)
        self.target_tokens = np.asarray(targets, dtype=np.uint16)
        self.tokenizer_bundle_sha256 = bundle_sha
        self.corpus_manifest_sha256 = corpus_sha
        self.admission = "audit_only"
        encoded = json.dumps(self.rows, sort_keys=True).encode()
        self.packed_manifest_sha256 = hashlib.sha256(encoded).hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        result = {key: torch.tensor(value, dtype=torch.long) for key, value in self.rows[index].items()}
        if self.include_row_indices:
            result["row_index"] = index
        return result


@torch.no_grad()
def evaluate_identity(model, tokenizer, config, rows):
    """Record actual greedy completions, including failures, on held-out questions."""
    model.eval()
    results = []
    device = next(model.parameters()).device
    eot = int(config["tokenizer"]["end_of_turn_id"])
    from tokenizer.contract import suppressed_controls
    for row in rows:
        prefix = encode_chat(
            tokenizer, [row["messages"][0]], int(config["training"]["sequence_length"]), config
        )
        inputs = torch.tensor([prefix], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            do_sample=False, max_new_tokens=128, eos_token_id=eot,
            pad_token_id=int(config["tokenizer"]["pad_id"]),
            suppress_tokens=suppressed_controls(config, eot),
        )[0, len(prefix):].tolist()
        stopped = eot in generated
        if stopped:
            generated = generated[:generated.index(eot)]
        answer = tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        results.append({
            "record_id": row["record_id"], "language": row["language"],
            "prompt": row["prompt"], "expected": row["answer"], "actual": answer,
            "eot_generated": stopped, "exact_match": answer == row["answer"] and stopped,
        })
    return {
        "identity_validation": results,
        "identity_exact_matches": sum(row["exact_match"] for row in results),
        "identity_validation_count": len(results),
    }
