import json
import tempfile
import unittest
from pathlib import Path

from data.hashing import sha256_file
from data.identity_stage import build, digest
from training.identity_data import read_identity


class IdentityAdmissionTests(unittest.TestCase):
    def test_exact_stage_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "identity"
            manifest = build(root)
            _, splits = read_identity(root, manifest["manifest_sha256"])
            self.assertEqual(len(splits["train"]), 2800)
            self.assertEqual(len(splits["validation"]), 16)
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                read_identity(root, "0" * 64)

    def test_rehashed_foreign_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "identity"
            manifest = build(root)
            path = root / "train.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["prompt"] = "Calculate 2 + 2."
            rows[0]["answer"] = "4"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest["files"]["train.jsonl"] = sha256_file(path)
            manifest.pop("manifest_sha256")
            manifest["manifest_sha256"] = digest(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authorized identity-only"):
                read_identity(root, manifest["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
