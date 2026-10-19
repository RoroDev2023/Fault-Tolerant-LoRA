"""Offline checks for the single recorded-source-hash rule and its decision files."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.source_hashes import (DEPENDENCY_CHANGES, SOURCE_CHANGES, compare_source_hashes, load_accepted_changes, summarize_lockfile_change,
                                                 summarize_requirements_change, unique_changes)


OLD_LOCK = "# header\ntorch==2.14.1 --hash=sha256:" + "1" * 64 + "\nfsspec==2026.9.0 --hash=sha256:" + "2" * 64 + "\nnumpy==2.4.6 --hash=sha256:" + "3" * 64 + "\n"
NEW_LOCK = ("# header\nDatasets==5.1.0 --hash=sha256:" + "4" * 64 + "\ntorch==2.14.1 --hash=sha256:" + "1" * 64 + "\nfsspec==2026.7.0 --hash=sha256:" + "5" * 64
            + "\nnumpy==2.4.6 --hash=sha256:" + "6" * 64 + "\n")


class SourceHashTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "configs").mkdir()
        (self.root / "src/fault_lora").mkdir(parents=True)
        (self.root / "requirements.txt").write_text("torch==2.14.1\n")
        (self.root / "src/fault_lora/code.py").write_text("x = 1\n")
        self.old = {name: sha256_file(self.root / name) for name in ("requirements.txt", "src/fault_lora/code.py")}
        (self.root / "requirements.txt").write_text("torch==2.14.1\ndatasets==5.1.0\n")
        (self.root / "src/fault_lora/code.py").write_text("x = 2\n")
        self.new = {name: sha256_file(self.root / name) for name in self.old}

    def entry(self, path, old=None, new=None, **extra):
        return {"path": path, "old_sha256": old or self.old[path], "new_sha256": new or self.new[path], "reason": "merge added packages", "date": "2026-10-18", **extra}

    def write(self, relative, entries):
        (self.root / relative).write_text(json.dumps(entries))

    def test_exact_match_needs_no_decision(self):
        (self.root / "requirements.txt").write_text("torch==2.14.1\n")
        self.assertEqual(compare_source_hashes(self.root, {"requirements.txt": self.old["requirements.txt"]}, "label"), [])

    def test_listed_dependency_pair_is_accepted_and_reported(self):
        self.write(DEPENDENCY_CHANGES, [self.entry("requirements.txt")])
        used = compare_source_hashes(self.root, {"requirements.txt": self.old["requirements.txt"]}, "label")
        self.assertEqual(used, [{"path": "requirements.txt", "recorded_sha256": self.old["requirements.txt"], "current_sha256": self.new["requirements.txt"], "list": "dependency",
                                 "decision_file": DEPENDENCY_CHANGES, "reason": "merge added packages", "date": "2026-10-18"}])
        self.assertEqual(unique_changes(used + used)[0]["records"], 2)

    def test_unlisted_mismatch_is_rejected(self):
        for recorded in ({"requirements.txt": self.old["requirements.txt"]}, {"src/fault_lora/code.py": self.old["src/fault_lora/code.py"]}):
            with self.subTest(recorded=recorded), self.assertRaisesRegex(AssertionError, "label: "):
                compare_source_hashes(self.root, recorded, "label")
        # A listed dependency change does not excuse a different file.
        self.write(DEPENDENCY_CHANGES, [self.entry("requirements.txt")])
        with self.assertRaisesRegex(AssertionError, "code.py"):
            compare_source_hashes(self.root, {"requirements.txt": self.old["requirements.txt"], "src/fault_lora/code.py": self.old["src/fault_lora/code.py"]}, "label")

    def test_listed_pair_for_a_non_dependency_file_is_rejected(self):
        self.write(DEPENDENCY_CHANGES, [self.entry("src/fault_lora/code.py")])
        with self.assertRaisesRegex(ValueError, "not allowed"):
            compare_source_hashes(self.root, {"src/fault_lora/code.py": self.old["src/fault_lora/code.py"]}, "label")
        # And the code list cannot carry dependency files or files outside the package.
        for path in ("requirements.txt", "scripts/run.py", "src/fault_lora/../x.py", "src/other/code.py"):
            self.write(DEPENDENCY_CHANGES, [])
            self.write(SOURCE_CHANGES, [self.entry("src/fault_lora/code.py", commit=None) | {"path": path}])
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "not allowed"):
                load_accepted_changes(self.root)

    def test_wrong_new_or_old_hash_is_rejected(self):
        for old, new in ((None, "f" * 64), ("e" * 64, None)):
            self.write(DEPENDENCY_CHANGES, [self.entry("requirements.txt", old=old, new=new)])
            with self.subTest(old=old, new=new), self.assertRaises(AssertionError):
                compare_source_hashes(self.root, {"requirements.txt": self.old["requirements.txt"]}, "label")

    def test_source_list_accepts_only_its_exact_pairs_and_missing_files_fail(self):
        self.write(SOURCE_CHANGES, [self.entry("src/fault_lora/code.py", commit="abc1234")])
        used = compare_source_hashes(self.root, {"src/fault_lora/code.py": self.old["src/fault_lora/code.py"]}, "label")
        self.assertEqual((used[0]["list"], used[0]["decision_file"]), ("source", SOURCE_CHANGES))
        with self.assertRaises(AssertionError):
            compare_source_hashes(self.root, {"requirements.txt": self.old["requirements.txt"]}, "label")
        with self.assertRaisesRegex(AssertionError, "missing"):
            compare_source_hashes(self.root, {"src/fault_lora/gone.py": "0" * 64}, "label")

    def test_malformed_decisions_rejected(self):
        good = self.entry("requirements.txt")
        for bad in ({**good, "reason": " "}, {**good, "date": "18/10/2026"}, {**good, "new_sha256": good["old_sha256"]}, {**good, "old_sha256": "ABC"},
                    {key: value for key, value in good.items() if key != "date"}, {**good, "extra": 1}):
            self.write(DEPENDENCY_CHANGES, [bad])
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_accepted_changes(self.root)
        self.write(DEPENDENCY_CHANGES, [good, good])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_accepted_changes(self.root)
        self.write(DEPENDENCY_CHANGES, [])
        self.write(SOURCE_CHANGES, [self.entry("src/fault_lora/code.py", commit="not-a-commit")])
        with self.assertRaises(ValueError):
            load_accepted_changes(self.root)

    def test_lockfile_and_requirements_summaries(self):
        summary = summarize_lockfile_change(OLD_LOCK, NEW_LOCK)
        self.assertEqual(summary, {"old_packages": 3, "new_packages": 4, "added": {"datasets": "5.1.0"}, "removed": {},
                                   "version_changed": {"fsspec": {"old": "2026.9.0", "new": "2026.7.0"}}, "same_version": 2,
                                   "same_version_with_different_wheel_hashes": ["numpy"]})
        self.assertEqual(summarize_requirements_change("# c\ntorch==1\n", "torch==1\ndatasets==5\n")["added"], ["datasets==5"])
        with self.assertRaises(AssertionError):
            summarize_lockfile_change(OLD_LOCK + "torch==2.0\n", NEW_LOCK)


if __name__ == "__main__":
    unittest.main()
