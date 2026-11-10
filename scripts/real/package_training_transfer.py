#!/usr/bin/env python3
"""Build a source-only handoff ZIP; never package weights, raw data or secrets.

Run after committing the selected source. This authoring tool uses local Git
only to enumerate tracked files; it is not called by training or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[2]
PREFIXES = ("src/fastwam/", "scripts/", "configs/", "tests/", "docs/real/", "LICENSES/")
ROOT_FILES = {"pyproject.toml", "requirements-preprocessing.txt", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "README_zh.md"}
EXTENSIONS = {".py", ".sh", ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".csv"}
VERIFY = '''from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for row in m['files']:
    p=(root/row['path']).resolve()
    if not p.is_relative_to(root) or not p.is_file():raise SystemExit('Missing/unsafe: '+row['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Hash mismatch: '+row['path'])
print('Verified',len(m['files']),'files; data and model assets are supplied separately.')
'''


def eligible(name):
    p = Path(name)
    if ".local." in name or any(part in {"__pycache__", "tmp", "outputs", ".git", ".env"} for part in p.parts):
        return False
    return name in ROOT_FILES or (name.startswith(PREFIXES) and p.suffix in EXTENSIONS)


def build(output: Path, repo: Path = REPO):
    if output.exists():
        raise FileExistsError(output)
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    names = sorted(n for n in names if n and eligible(n))
    required = {"scripts/real/configure_transfer_training.py", "configs/real/transfer/preprocess_joint.template.json",
                "docs/real/WAM_TRANSFER_ZH.md", "src/fastwam/real/preprocessing/piper.py", "pyproject.toml"}
    if not required.issubset(names):
        raise ValueError("Commit/stage the handoff source before packaging: " + str(required - set(names)))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    entries = {}
    for name in names:
        p = repo / name
        if p.is_symlink() or not p.resolve().is_relative_to(repo.resolve()):
            raise ValueError("Refusing linked source: " + name)
        if p.stat().st_size > 5_000_000:
            raise ValueError("Unexpected large source file: " + name)
        entries["code/" + name] = p.read_bytes()
    entries["START_HERE.md"] = ("# WAM 真机训练转移包\n\n"
        "先读 `code/docs/real/WAM_TRANSFER_ZH.md`，再按其中命令准备新任务数据。\n"
        "`code/` 是完整参考 Python 源码和配置；不要覆盖目标项目自己的模型实现。\n"
        "本包不含真实数据、权重、模型资产、API token、SSH key、CAN/相机驱动或已验收的现场推理服务。\n"
        "解压后运行 `python verify_bundle.py` 检查传输完整性。\n"
        "本交接路线不保留 retrospection，也不构建 WARM memory。\n").encode("utf-8")
    entries["verify_bundle.py"] = VERIFY.encode("utf-8")
    manifest = {"schema": "warm.training-transfer.v1", "source_commit": revision,
                "snapshot": "tracked working-file bytes; each file has an individual SHA256",
                "weights_included": False, "recorded_data_included": False,
                "files": [{"path": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n,b in sorted(entries.items())]}
    entries["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(entries.items()):
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            z.writestr(zi, data)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        for item in manifest["files"]:
            if hashlib.sha256(z.read(item["path"])).hexdigest() != item["sha256"]:
                raise RuntimeError("ZIP source mismatch")
    receipt = {"zip": str(output.resolve()), "bytes": output.stat().st_size,
               "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_commit": revision,
               "files": len(entries), "crc_and_file_hashes_verified": True}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    build(p.parse_args().output)
