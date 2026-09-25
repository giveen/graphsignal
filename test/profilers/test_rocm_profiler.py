import contextlib
import ctypes.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from graphsignal.profilers import rocm_profiler as rp


class ResourceStub:
    def __init__(self, value, label=None):
        self.value = value
        self.label = label or value

    def joinpath(self, *parts):
        return ResourceStub(os.path.join(self.value, *parts))

    def __str__(self):
        return self.value

    def exists(self):
        return Path(self.value).is_file()


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for name in (
        "ROCM_VERSION", "ROCM_TOOLKIT_VERSION", "ROCM_PATH", "ROCM_HOME",
        "ROCP_TOOL_LIBRARIES", "LD_LIBRARY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def test_packaged_rocm_path_selects_amd64_version(tmp_path):
    library = tmp_path / "_native" / "amd64-rocm7" / "libgsrocmprof.so"
    library.parent.mkdir(parents=True)
    library.touch()
    resource = ResourceStub(str(tmp_path), "graphsignal")
    resources = types.SimpleNamespace(
        files=lambda package: resource,
        as_file=lambda candidate: contextlib.nullcontext(Path(str(candidate))),
    )
    with patch("importlib.resources.files", resources.files), \
         patch("importlib.resources.as_file", resources.as_file):
        expected = str(tmp_path / "_native" / "amd64-rocm7" / "libgsrocmprof.so")
        assert rp._packaged_rocm_so_path(7) == expected


def test_packaged_rocm_path_returns_none_when_missing_or_resources_fail(tmp_path):
    resource = ResourceStub(str(tmp_path), "graphsignal")
    with patch("importlib.resources.files", return_value=resource), \
         patch("importlib.resources.as_file", lambda candidate: contextlib.nullcontext(Path(str(candidate)))):
        assert rp._packaged_rocm_so_path(6) is None

    with patch("importlib.resources.files", side_effect=ImportError("broken resources")):
        assert rp._packaged_rocm_so_path(6) is None


@pytest.mark.parametrize(
    "env_name,value,expected",
    [("ROCM_VERSION", "7.2.3", 7), ("ROCM_TOOLKIT_VERSION", " 6.4 ", 6)],
)
def test_detect_rocm_major_from_environment(monkeypatch, env_name, value, expected):
    monkeypatch.setenv(env_name, value)
    assert rp._detect_rocm_major() == expected


def test_detect_rocm_major_falls_through_invalid_environment(monkeypatch):
    monkeypatch.setenv("ROCM_VERSION", "unknown")
    monkeypatch.setenv("ROCM_TOOLKIT_VERSION", "7.1")
    assert rp._detect_rocm_major() == 7


def test_detect_rocm_major_from_torch(monkeypatch):
    monkeypatch.setattr(rp.os, "listdir", lambda path: (_ for _ in ()).throw(OSError("no /opt")))
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(hip=" 6.3.41134 "))
    )
    assert rp._detect_rocm_major() == 6


def test_detect_rocm_major_ignores_bad_torch_and_tolerates_access_exception(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(hip=6)))
    with patch.object(rp.os, "listdir", side_effect=OSError("unreadable /opt")):
        assert rp._detect_rocm_major() is None

    class BrokenVersion:
        @property
        def hip(self):
            raise RuntimeError("broken torch")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(version=BrokenVersion()))
    with patch.object(rp.os, "listdir", side_effect=OSError("unreadable /opt")):
        assert rp._detect_rocm_major() is None


@pytest.mark.parametrize("version_file", [".info/version", ".info/version-dev", ".info/version-utils"])
def test_detect_rocm_major_from_install_version_file(monkeypatch, tmp_path, version_file):
    info = tmp_path / ".info"
    info.mkdir()
    (info / Path(version_file).name).write_text(" 7.2.3-123\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert rp._detect_rocm_major() == 7


def test_detect_rocm_major_resolves_versioned_symlink_path(monkeypatch, tmp_path):
    install = tmp_path / "rocm"
    install.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(install))
    monkeypatch.setattr(rp.os.path, "realpath", lambda path: "/opt/rocm-7.3.1")
    assert rp._detect_rocm_major() == 7


def test_detect_rocm_major_prefers_rocm_path_versioned_name(monkeypatch, tmp_path):
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm-6.2"))
    monkeypatch.setenv("ROCM_HOME", str(tmp_path / "rocm-7.2"))
    assert rp._detect_rocm_major() == 6


def test_detect_rocm_major_scans_opt_and_chooses_highest_install(monkeypatch, tmp_path):
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)
    opt = tmp_path / "opt"
    opt.mkdir()
    (opt / "rocm-6.2").mkdir()
    (opt / "rocm-7.2").mkdir()
    (opt / "rocm-10.0").mkdir()
    (opt / "rocm-9.0").touch()  # files do not count
    # The production scan is intentionally absolute; patch only listdir/isdir.
    real_isdir = rp.os.path.isdir
    monkeypatch.setattr(rp.os, "listdir", lambda path: ["rocm-6.2", "rocm-7.2", "rocm-10.0", "rocm-9.0"] if path == "/opt" else real_isdir(path))
    monkeypatch.setattr(rp.os.path, "isdir", lambda path: path in {
        "/opt/rocm-6.2", "/opt/rocm-7.2", "/opt/rocm-10.0", "/opt/rocm-9.0"
    })
    assert rp._detect_rocm_major() == 10


def test_read_rocm_version_file_tries_files_in_order_and_handles_bad_content(tmp_path):
    (tmp_path / ".info").mkdir()
    (tmp_path / ".info" / "version").write_text("not-a-version")
    (tmp_path / ".info" / "version-dev").write_text("6.4.1-99")
    assert rp._read_rocm_version_file(str(tmp_path)) == 6


def test_read_rocm_version_file_returns_none_when_unreadable_or_invalid(tmp_path):
    assert rp._read_rocm_version_file(str(tmp_path / "missing")) is None
    info = tmp_path / ".info"
    info.mkdir()
    (info / "version").write_text("dev")
    assert rp._read_rocm_version_file(str(tmp_path)) is None


def test_ensure_rocp_tool_libraries_prepends_once_and_ignores_empty_entries(monkeypatch):
    monkeypatch.setenv("ROCP_TOOL_LIBRARIES", "::/old/tool.so:")
    rp._ensure_rocp_tool_libraries("/new/libgsrocmprof.so")
    assert os.environ["ROCP_TOOL_LIBRARIES"] == "/new/libgsrocmprof.so:/old/tool.so"

    rp._ensure_rocp_tool_libraries("/new/libgsrocmprof.so")
    assert os.environ["ROCP_TOOL_LIBRARIES"] == "/new/libgsrocmprof.so:/old/tool.so"


def test_ensure_rocp_tool_libraries_sets_single_value_when_unset(monkeypatch):
    rp._ensure_rocp_tool_libraries("/tool/libgsrocmprof.so")
    assert os.environ["ROCP_TOOL_LIBRARIES"] == "/tool/libgsrocmprof.so"


def test_ensure_librocprofiler_finds_loader_path_and_prepends_directory(tmp_path, monkeypatch):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    library = lib_dir / "librocprofiler-sdk.so.1"
    library.touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    with patch.object(ctypes.util, "find_library", return_value=str(library)):
        assert rp._ensure_librocprofiler_ld_library_path()
    assert os.environ["LD_LIBRARY_PATH"] == f"{lib_dir}:/existing"


def test_ensure_librocprofiler_accepts_bare_soname_without_modifying_path(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unchanged")
    with patch.object(ctypes.util, "find_library", return_value="librocprofiler-sdk.so.1"):
        assert rp._ensure_librocprofiler_ld_library_path()
    assert os.environ["LD_LIBRARY_PATH"] == "/unchanged"


@pytest.mark.parametrize("subdir", ["lib", "lib64"])
def test_ensure_librocprofiler_searches_rocm_path_idempotently(tmp_path, monkeypatch, subdir):
    lib_dir = tmp_path / subdir
    lib_dir.mkdir()
    (lib_dir / "librocprofiler-sdk.so.1").touch()
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", f"::/existing:")
    with patch.object(ctypes.util, "find_library", return_value=None):
        assert rp._ensure_librocprofiler_ld_library_path()
        assert rp._ensure_librocprofiler_ld_library_path()
    assert os.environ["LD_LIBRARY_PATH"] == f"{lib_dir}:/existing"


def test_ensure_librocprofiler_falls_back_to_default_opt_paths(monkeypatch, tmp_path):
    lib_dir = "/opt/rocm/lib"
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(rp.os.path, "isdir", side_effect=lambda path: path == lib_dir), \
         patch.object(rp.os, "listdir", return_value=["librocprofiler-sdk.so.2"]):
        assert rp._ensure_librocprofiler_ld_library_path()
    assert os.environ["LD_LIBRARY_PATH"] == "/opt/rocm/lib:/existing"


def test_ensure_librocprofiler_handles_listdir_exception_and_returns_false(monkeypatch, tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(rp.os, "listdir", side_effect=PermissionError("denied")):
        assert rp._ensure_librocprofiler_ld_library_path() is False


def test_setup_ignores_graph_flag_and_rejects_non_linux(monkeypatch):
    monkeypatch.setattr(rp.sys, "platform", "darwin")
    with patch.object(rp, "_detect_rocm_major") as detect:
        assert rp.RocmProfiler.setup_env_vars("graph") is False
    detect.assert_not_called()


def test_setup_reports_missing_rocm_library_and_rocp_runtime(monkeypatch):
    monkeypatch.setattr(rp.sys, "platform", "linux")
    with patch.object(rp, "_detect_rocm_major", return_value=None):
        assert rp.RocmProfiler.setup_env_vars() is False

    with patch.object(rp, "_detect_rocm_major", return_value=7), \
         patch.object(rp, "_packaged_rocm_so_path", return_value=None), \
         patch.object(rp, "_ensure_librocprofiler_ld_library_path") as runtime:
        assert rp.RocmProfiler.setup_env_vars() is False
    runtime.assert_not_called()

    with patch.object(rp, "_detect_rocm_major", return_value=7), \
         patch.object(rp, "_packaged_rocm_so_path", return_value="/pkg/libgsrocmprof.so"), \
         patch.object(rp, "_ensure_librocprofiler_ld_library_path", return_value=False), \
         patch.object(rp, "_ensure_rocp_tool_libraries") as tool:
        assert rp.RocmProfiler.setup_env_vars() is False
    tool.assert_not_called()


def test_setup_success_publishes_tool_library(monkeypatch):
    monkeypatch.setattr(rp.sys, "platform", "linux")
    with patch.object(rp, "_detect_rocm_major", return_value=7), \
         patch.object(rp, "_packaged_rocm_so_path", return_value="/pkg/libgsrocmprof.so"), \
         patch.object(rp, "_ensure_librocprofiler_ld_library_path", return_value=True):
        assert rp.RocmProfiler.setup_env_vars("graph")
    assert os.environ["ROCP_TOOL_LIBRARIES"] == "/pkg/libgsrocmprof.so"
