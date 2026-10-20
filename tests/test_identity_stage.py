import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from data.identity_stage import ANSWERS, build, records


class IdentityStageTests(unittest.TestCase):
    def test_scope_and_split(self):
        train = list(records("train"))
        validation = list(records("validation"))
        self.assertEqual(Counter(r["language"] for r in train), {"en": 1400, "ru": 1400})
        prompts = [r["prompt"].casefold() for r in train + validation]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertEqual(ANSWERS["ru"], "Я — ШИНРА. СОЗДАНА NULLXES.")
        self.assertEqual(ANSWERS["en"], "I AM SHINRA. CREATED BY NULLXES.")
        for row in train + validation:
            self.assertEqual(row["answer"], ANSWERS[row["language"]])
            self.assertEqual(row["messages"][-1]["content"], row["answer"])

    def test_replay_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            self.assertEqual(build(a), build(b))
            manifest = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
            for filename, expected in manifest["files"].items():
                self.assertEqual(hashlib.sha256((a / filename).read_bytes()).hexdigest(), expected)
                self.assertEqual((a / filename).read_bytes(), (b / filename).read_bytes())
            with self.assertRaises(FileExistsError):
                build(a)


if __name__ == "__main__":
    unittest.main()
