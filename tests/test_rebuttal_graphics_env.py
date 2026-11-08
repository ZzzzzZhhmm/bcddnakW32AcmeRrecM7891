"""Fake library files exercise configuration only; they do not qualify GPUs."""
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    p=Path(__file__).resolve().parents[1]/"scripts/setup_rebuttal_graphics_env.py"
    spec=importlib.util.spec_from_file_location("graphics_setup",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def libraries(root):
    root.mkdir()
    for name in ("libGLX_nvidia.so.1.2.3", "libEGL_nvidia.so.1.2.3"):
        (root/name).write_bytes(b"synthetic test fixture, not a real library")
    (root/"nvidia_icd.json").write_text(json.dumps({"file_format_version":"1.0.0",
                                                  "ICD":{"api_version":"1.3.0","library_path":"old.so"}}))


def test_configs_are_process_local_quoted_and_idempotent(tmp_path,monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH","existing")
    driver=tmp_path/"extracted libraries";libraries(driver)
    out=tmp_path/"process settings"
    m=module();r=m.configure(driver,out,driver_version="1.2.3")
    assert json.loads((out/"vulkan_icd.json").read_text())["ICD"]["library_path"]==str(driver/"libGLX_nvidia.so.1.2.3")
    assert "'" in (out/"graphics_env.sh").read_text()
    assert m.configure(driver,out,driver_version="1.2.3")==r
    import os
    assert os.environ["LD_LIBRARY_PATH"]=="existing"


def test_mismatch_and_unknown_existing_configs_are_preserved(tmp_path):
    driver=tmp_path/"driver";libraries(driver)
    out=tmp_path/"out";m=module()
    with pytest.raises(ValueError,match="match the live driver"):
        m.configure(driver,out,driver_version="2.0")
    assert not out.exists()
    out.mkdir();(out/"graphics_env.sh").write_text("someone else's configuration")
    with pytest.raises(ValueError,match="refusing to overwrite"):
        m.configure(driver,out,driver_version="1.2.3")
    assert not (out/"vulkan_icd.json").exists()
