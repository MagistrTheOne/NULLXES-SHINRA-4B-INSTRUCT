# SHINRA Native tokenizer: one Colab cell
# Paste the complete cell into the repository's Google Colab runtime and run it manually.
# It does not train a tokenizer or model. It reads the existing clean corpus, writes a new
# versioned bundle beside the verified source, writes bounded smoke shards under /content,
# and never changes the source artifact or production corpus.

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO = Path("/content/NULLXES-SHINRA-4B-INSTRUCT")
VERIFIED = Path("/content/drive/MyDrive/NULLXES/SHINRA/tokenizer/artifacts/shinra_unigram_65k_v1_verified")
ORIGINAL_NATIVE = Path("/content/drive/MyDrive/NULLXES/SHINRA/tokenizer/artifacts/shinra_unigram_65k_v1_native")
CORPUS = Path("/content/drive/MyDrive/NULLXES/SHINRA/datasets/clean/tokenizer_wiki_v2")
EXPECTED_NATIVE_MODEL_SHA256 = "372eb8e5b82a950af7db3fd54d1ded4a36116016af6103076074ef99d9a0d83c"
EXPECTED_ORIGINAL_VOCAB_SHA256 = "193cbf2e1dca090497352ba3b567e18740d0c63515a3ac9e886fc17db6562acc"
EXPECTED_NATIVE_VOCAB_SIZE = 65_536
EXPECTED_VALIDATION_DOCUMENTS = 19_829
EXPECTED_VALIDATION_TOKENS = 15_147_605
SMOKE_DOCUMENT_LIMIT = 32

if not REPO.is_dir():
    raise FileNotFoundError(f"Repository checkout unavailable: {REPO}")
if not VERIFIED.is_dir() or not ORIGINAL_NATIVE.is_dir() or not CORPUS.is_dir():
    raise FileNotFoundError("Mount Google Drive and check both tokenizer artifact paths and the clean corpus path")
sys.path.insert(0, str(REPO))

from configs.loader import load_config
from data.pack import file_sha256, iter_clean_records, pack_documents
from tokenizer import load_shinra_tokenizer
from tokenizer.bundle import _digest, verify_bundle, write_manifest, write_native_metadata
from tokenizer.contract import encode_text

config = copy.deepcopy(load_config())
verified_model_path = VERIFIED / "tokenizer.model"
original_model_path = ORIGINAL_NATIVE / "tokenizer.model"
original_vocab_path = ORIGINAL_NATIVE / "tokenizer.vocab"
for required_path in (verified_model_path, original_model_path, original_vocab_path):
    if not required_path.is_file():
        raise FileNotFoundError(f"Required Native tokenizer source file is missing: {required_path}")

verified_model_sha256 = file_sha256(verified_model_path)
if verified_model_sha256 != EXPECTED_NATIVE_MODEL_SHA256:
    raise ValueError(
        "Verified SentencePiece model SHA256 differs from the approved baseline: "
        f"expected={EXPECTED_NATIVE_MODEL_SHA256}, actual={verified_model_sha256}"
    )
original_model_sha256 = file_sha256(original_model_path)
if original_model_sha256 != EXPECTED_NATIVE_MODEL_SHA256:
    raise ValueError(
        "Original Native SentencePiece model SHA256 differs from the approved baseline: "
        f"expected={EXPECTED_NATIVE_MODEL_SHA256}, actual={original_model_sha256}"
    )
if verified_model_sha256 != original_model_sha256:
    raise ValueError(
        "Verified and Original Native SentencePiece model SHA256 differ: "
        f"verified={verified_model_sha256}, original={original_model_sha256}"
    )
original_vocab_sha256 = file_sha256(original_vocab_path)
if original_vocab_sha256 != EXPECTED_ORIGINAL_VOCAB_SHA256:
    raise ValueError(
        "Original Native SentencePiece vocab SHA256 differs from the approved artifact: "
        f"expected={EXPECTED_ORIGINAL_VOCAB_SHA256}, actual={original_vocab_sha256}"
    )
if not (CORPUS / "manifest.json").is_file() or not (CORPUS / "validation.jsonl").is_file():
    raise FileNotFoundError("Clean corpus must provide manifest.json and validation.jsonl")
import sentencepiece as spm
processor = spm.SentencePieceProcessor(model_file=str(verified_model_path))


def validate_vocab_companion(vocab_path: Path, sentencepiece_processor) -> None:
    """Parse by model piece boundaries, preserving literal CR/LF/tab bytes in pieces."""
    piece_count = sentencepiece_processor.get_piece_size()
    if piece_count != EXPECTED_NATIVE_VOCAB_SIZE:
        raise ValueError(
            f"SentencePiece model has {piece_count} pieces; expected exactly {EXPECTED_NATIVE_VOCAB_SIZE}"
        )
    vocab_bytes = vocab_path.read_bytes()
    cursor = 0
    score_pattern = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")

    for index in range(piece_count):
        expected_piece = sentencepiece_processor.id_to_piece(index)
        expected_piece_bytes = expected_piece.encode("utf-8", errors="strict")
        expected_score = Decimal(str(sentencepiece_processor.get_score(index)))
        piece_end = cursor + len(expected_piece_bytes)
        actual_piece_bytes = vocab_bytes[cursor:piece_end]
        if actual_piece_bytes != expected_piece_bytes:
            actual_piece = actual_piece_bytes.decode("utf-8", errors="replace")
            raise ValueError(
                f"SentencePiece vocab piece mismatch at ID {index}: "
                f"expected={expected_piece!r}, actual={actual_piece!r}"
            )
        cursor = piece_end

        if cursor >= len(vocab_bytes) or vocab_bytes[cursor] != 0x09:
            actual_byte = vocab_bytes[cursor:cursor + 1]
            raise ValueError(
                f"SentencePiece vocab record mismatch at ID {index}: "
                f"expected tab after piece={expected_piece!r}, actual byte={actual_byte!r}"
            )
        cursor += 1

        record_end = vocab_bytes.find(b"\n", cursor)
        if record_end < 0:
            if index != piece_count - 1:
                raise ValueError(
                    f"SentencePiece vocab record mismatch at ID {index}: "
                    "score record terminator is missing before the final ID"
                )
            record_end = len(vocab_bytes)
        score_bytes = vocab_bytes[cursor:record_end]
        if score_bytes.endswith(b"\r"):
            score_bytes = score_bytes[:-1]
        if not score_pattern.fullmatch(score_bytes):
            raise ValueError(
                f"SentencePiece vocab score mismatch at ID {index}: "
                f"expected piece={expected_piece!r}, score={str(expected_score)!r}; "
                f"actual score bytes={score_bytes!r}"
            )
        try:
            actual_score = Decimal(score_bytes.decode("ascii", errors="strict"))
        except (UnicodeDecodeError, InvalidOperation) as exc:
            raise ValueError(
                f"SentencePiece vocab score mismatch at ID {index}: "
                f"expected piece={expected_piece!r}, score={str(expected_score)!r}; "
                f"actual score bytes={score_bytes!r}"
            ) from exc
        score_exponent = actual_score.as_tuple().exponent
        score_rounding_tolerance = Decimal((0, (5,), score_exponent - 1))
        score_quantum = Decimal((0, (1,), score_exponent))
        if (not actual_score.is_finite()
                or abs(actual_score - expected_score) > score_rounding_tolerance):
            raise ValueError(
                f"SentencePiece vocab score mismatch at ID {index}: "
                f"expected piece={expected_piece!r}, score={str(expected_score)!r}; "
                f"actual piece={expected_piece!r}, score={str(actual_score)!r}; "
                f"written score quantum={score_quantum}, "
                f"allowed absolute rounding error={score_rounding_tolerance} "
                "(half a written unit; exact boundary is accepted)"
            )

        cursor = record_end
        if cursor < len(vocab_bytes):
            cursor += 1  # Consume the score record LF; CRLF was handled above.
        elif index != piece_count - 1:
            raise ValueError(f"SentencePiece vocab ended before ID {index + 1}")

    if cursor != len(vocab_bytes):
        raise ValueError(
            f"SentencePiece vocab has trailing bytes after {piece_count} entries: "
            f"offset={cursor}, trailing_bytes={len(vocab_bytes) - cursor}"
        )


validate_vocab_companion(original_vocab_path, processor)

corpus_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
corpus_hash = corpus_manifest.pop("manifest_sha256", None)
if corpus_hash != _digest(corpus_manifest):
    raise ValueError("Clean corpus manifest self-hash mismatch")
corpus_manifest["manifest_sha256"] = corpus_hash

# Validate the entire real held-out split against SentencePiece and the production wrapper.
validation = {"documents": 0, "tokens": 0, "bytes": 0, "languages": {"en": 0, "ru": 0}}
for record in iter_clean_records(CORPUS, "validation", corpus_manifest):
    text = record["text"]
    native_ids = processor.encode(text, out_type=int)
    validation["documents"] += 1
    validation["tokens"] += len(native_ids)
    validation["bytes"] += len(text.encode("utf-8"))
    language = record.get("language")
    if language not in validation["languages"]:
        raise ValueError(f"Unexpected held-out language: {language!r}")
    validation["languages"][language] += 1
if validation["documents"] != EXPECTED_VALIDATION_DOCUMENTS:
    raise ValueError(f"Unexpected validation document count: {validation['documents']}")
if validation["tokens"] != EXPECTED_VALIDATION_TOKENS:
    raise ValueError(f"Unexpected Native token count: {validation['tokens']}")
if set(validation["languages"]) != {"en", "ru"} or min(validation["languages"].values()) == 0:
    raise ValueError("Held-out corpus must contain English and Russian documents")

# All source checks passed. Create a new bundle now; source artifacts remain read-only.
bundle_root = VERIFIED.parent / f"shinra_native_unigram_65k_bundle_{uuid.uuid4().hex[:10]}"
bundle_root.mkdir(parents=False, exist_ok=False)
shutil.copy2(verified_model_path, bundle_root / "tokenizer.model")
shutil.copy2(original_vocab_path, bundle_root / "tokenizer.vocab")
copied_vocab_sha256 = file_sha256(bundle_root / "tokenizer.vocab")
if copied_vocab_sha256 != EXPECTED_ORIGINAL_VOCAB_SHA256:
    raise ValueError(
        "Copied Native tokenizer.vocab SHA256 differs from the approved artifact: "
        f"expected={EXPECTED_ORIGINAL_VOCAB_SHA256}, actual={copied_vocab_sha256}"
    )

# The wrapper requires a signed bundle before loading. First stage real provenance/statistics,
# then write the manifest and freeze only the in-memory smoke-test configuration.
config["data"]["manifest_sha256"] = corpus_hash
stats = {
    **validation,
    "native_model_sha256": EXPECTED_NATIVE_MODEL_SHA256,
    "original_vocab_sha256": EXPECTED_ORIGINAL_VOCAB_SHA256,
    "validation_jsonl_sha256": corpus_manifest["files"]["validation.jsonl"],
    "wrapper_parity": "pending bundle load",
}
(bundle_root / "tokenizer_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
corpus_stats = {
    "source_manifest_sha256": corpus_hash,
    "validation_documents_checked": validation["documents"],
    "validation_tokens_checked": validation["tokens"],
    "purpose": "tokenizer-validation-and-bounded-packing-smoke-only",
}
(bundle_root / "corpus_stats.json").write_text(json.dumps(corpus_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
bundle_provenance = {
    "source_model_sha256": EXPECTED_NATIVE_MODEL_SHA256,
    "source_vocab_sha256": EXPECTED_ORIGINAL_VOCAB_SHA256,
    "clean_corpus_manifest_sha256": corpus_hash,
    "validation_jsonl_sha256": corpus_manifest["files"]["validation.jsonl"],
}
write_native_metadata(bundle_root, config, provenance=bundle_provenance)
bundle_sha = write_manifest(bundle_root, config, backend="sentencepiece_native", provenance=bundle_provenance)
config["tokenizer"]["artifact_sha256"] = bundle_sha
verify_bundle(bundle_root, config, backend="sentencepiece_native")
tokenizer = load_shinra_tokenizer(bundle_root, config)
if len(tokenizer) != 65_536:
    raise ValueError("Native loader changed vocabulary size")

# Repeat the full parity/roundtrip pass through the actual loaded wrapper before packing.
validation["wrapper_parity_mismatches"] = 0
validation["wrapper_roundtrip_errors"] = 0
for record in iter_clean_records(CORPUS, "validation", corpus_manifest):
    native_ids = processor.encode(record["text"], out_type=int)
    wrapped_ids = encode_text(tokenizer, record["text"], config)
    if native_ids != wrapped_ids:
        validation["wrapper_parity_mismatches"] += 1
        raise ValueError(f"Native wrapper token mismatch at document {validation['wrapper_parity_mismatches']}")
    if tokenizer.decode(wrapped_ids, skip_special_tokens=False) != record["text"]:
        validation["wrapper_roundtrip_errors"] += 1
        raise ValueError("Native wrapper changed an NFC-cleaned validation document")

# Measured parity is part of the bundle, so compute the final bundle identity after this pass.
validation["wrapper_parity"] = "verified against all validation documents"
stats.update({key: value for key, value in validation.items() if key.startswith("wrapper_")})
(bundle_root / "tokenizer_stats.json").write_text(
    json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
bundle_sha = write_manifest(bundle_root, config, backend="sentencepiece_native", provenance=bundle_provenance)
config["tokenizer"]["artifact_sha256"] = bundle_sha
verify_bundle(bundle_root, config, backend="sentencepiece_native")
tokenizer = load_shinra_tokenizer(bundle_root, config)

# The smoke output is disposable and explicitly marked so training/evaluation loaders reject it.
smoke_root = Path(tempfile.mkdtemp(prefix="shinra-native-packing-smoke-", dir="/content"))
packed_dir = smoke_root / "packed" / "validation"
report = pack_documents(
    CORPUS, bundle_root, packed_dir,
    int(config["model"]["train_sequence_length"]), int(config["data"]["packed_shard_tokens"]),
    config_path=REPO / "configs" / "shinra.yaml", required_split="validation",
    config=config, smoke_test_documents=SMOKE_DOCUMENT_LIMIT,
)
if not report["smoke_test_only"] or report["documents"] > SMOKE_DOCUMENT_LIMIT:
    raise AssertionError("Packing smoke run escaped its bounded test contract")
os.environ["SHINRA_NATIVE_BUNDLE"] = str(bundle_root)
os.environ["SHINRA_TOKENIZER_CORPUS"] = str(CORPUS)
os.environ["SHINRA_PACKED_SMOKE_DIR"] = str(packed_dir)

# Run the real-artifact integration tests in Colab only. No model or training code is imported.
import pytest
exit_code = pytest.main(["-q", str(REPO / "tests" / "test_native_tokenizer_integration.py")])
final_report = {
    "status": "PASS" if exit_code == pytest.ExitCode.OK else "FAIL",
    "pytest_exit_code": int(exit_code),
    "native_model_sha256": EXPECTED_NATIVE_MODEL_SHA256,
    "native_bundle_sha256": bundle_sha,
    "bundle_directory": str(bundle_root),
    "validation": validation,
    "packing_smoke": report,
    "packed_smoke_directory": str(packed_dir),
    "training_started": False,
    "production_corpus_modified": False,
    "source_verified_artifact_modified": False,
}
report_path = bundle_root.parent / f"{bundle_root.name}_smoke_report.json"
report_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(final_report, indent=2, ensure_ascii=False, sort_keys=True))
if exit_code != pytest.ExitCode.OK:
    raise RuntimeError(f"Native tokenizer integration tests failed; report: {report_path}")
