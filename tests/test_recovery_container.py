import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import tarfile
import tempfile
import unittest
import zipfile

module_path = Path(__file__).resolve().parents[1] / "scripts/nonreal/recovery_container.py"
spec = importlib.util.spec_from_file_location("recovery_container_under_test", module_path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryContainerIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.raw = self.inputs / "checkpoint.pt"
        self.data = random.Random(19).randbytes(8192) * 3
        self.raw.write_bytes(self.data)
        self.empty = self.inputs / "empty.bin"
        self.empty.write_bytes(b"")
        self.tar = self.inputs / "original.tar"
        with tarfile.open(self.tar, "w") as pack:
            pack.add(self.raw, arcname="checkpoint.pt")
            pack.add(self.empty, arcname="empty.bin")
        self.original_tar = self.tar.read_bytes()
        self.plan = {"source_scope": str(self.inputs), "chunk_bytes": 4096,
                     "files": [self.row(p) for p in [self.raw, self.empty, self.tar]]}
        self.archive = self.root / "first.zip"

    def tearDown(self):
        self.temp.cleanup()

    def row(self, path):
        s = path.stat()
        return {"key": path.name, "path": str(path), "stat": [s.st_size, s.st_mtime_ns, s.st_ino],
                "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def build(self):
        recovery.build_container(self.plan, self.archive, recovery.Budget(30))

    def rewrite(self, target, transform):
        with zipfile.ZipFile(self.archive) as src, zipfile.ZipFile(target, "w") as out:
            for name in src.namelist():
                out.writestr(name, transform(name, src.read(name)))

    def test_raw_tar_padding_and_empty_file_survive_without_originals(self):
        self.build()
        second = self.root / "second.zip"
        shutil.copyfile(self.archive, second)
        self.assertNotEqual(self.archive.stat().st_ino, second.stat().st_ino)
        for path in [self.raw, self.empty, self.tar]:
            path.unlink()
        proof = recovery.verify_container(second, recovery.Budget(30))
        self.assertTrue(proof["all_files_exact"])
        self.assertFalse(proof["external_base_required"])
        self.assertLess(proof["unique_uncompressed_bytes"], proof["logical_bytes"])
        for key, expected in [("checkpoint.pt", self.data), ("original.tar", self.original_tar), ("empty.bin", b"")]:
            output = self.root / "restored" / key
            recovery.extract_entry(second, key, output, recovery.Budget(30))
            self.assertEqual(output.read_bytes(), expected)

    def test_corrupt_object_is_rejected(self):
        self.build()
        corrupt = self.root / "corrupt.zip"
        changed = False
        def transform(name, data):
            nonlocal changed
            if name.startswith("objects/") and not changed:
                changed = True
                return bytes([data[0] ^ 1]) + data[1:]
            return data
        self.rewrite(corrupt, transform)
        with self.assertRaisesRegex(ValueError, "object SHA256"):
            recovery.verify_container(corrupt, recovery.Budget(30))

    def test_changed_source_is_rejected_before_output(self):
        self.raw.write_bytes(self.data + b"changed")
        with self.assertRaisesRegex(ValueError, "changed after registration"):
            self.build()
        self.assertFalse(self.archive.exists())

    def test_duplicate_and_escaping_keys_are_rejected(self):
        self.plan["files"].append(dict(self.plan["files"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build()
        self.plan["files"].pop()
        for key in ["../outside", "/absolute", "..\\outside", "a//b"]:
            self.plan["files"][0]["key"] = key
            with self.assertRaises(ValueError):
                self.build()

    def test_existing_extraction_is_never_overwritten(self):
        self.build()
        output = self.root / "existing.pt"
        output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            recovery.extract_entry(self.archive, "checkpoint.pt", output, recovery.Budget(30))
        self.assertEqual(output.read_bytes(), b"keep")

    def test_recipe_order_corruption_is_rejected(self):
        self.build()
        corrupt = self.root / "reordered.zip"
        def transform(name, data):
            if name != "manifest.json": return data
            m = json.loads(data)
            parts = m["files"]["checkpoint.pt"]["parts"]
            parts[0], parts[1] = parts[1], parts[0]
            return json.dumps(m).encode()
        self.rewrite(corrupt, transform)
        with self.assertRaisesRegex(ValueError, "file SHA256"):
            recovery.verify_container(corrupt, recovery.Budget(30))

    def test_invalid_or_exhausted_budget_is_rejected(self):
        for seconds in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError): recovery.Budget(seconds)
        exhausted = recovery.Budget(30)
        exhausted.deadline = 0
        with self.assertRaises(TimeoutError):
            recovery.build_container(self.plan, self.archive, exhausted)
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
