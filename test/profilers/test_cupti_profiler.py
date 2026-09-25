import contextlib
import ctypes.util
import importlib.metadata
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from graphsignal.profilers import cupti_profiler as cp


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


def clear_cuda_env(monkeypatch):
    for name in (
        "CUDA_VERSION", "CUDA_TOOLKIT_VERSION", "CUDA_HOME", "CUDA_PATH",
        "CUDA_INJECTION64_PATH", "LD_LIBRARY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    clear_cuda_env(monkeypatch)


def test_detect_arch_tag_normalizes_supported_and_defaults(monkeypatch):
    for machine in ("aarch64", "arm64", "ARM64", "AArch64"):
        monkeypatch.setattr(cp.platform, "machine", lambda m=machine: m)
        assert cp._detect_arch_tag() == "arm64"

    for machine in ("x86_64", "AMD64", "riscv64", "", "amd64"):
        monkeypatch.setattr(cp.platform, "machine", lambda m=machine: m)
        assert cp._detect_arch_tag() == "amd64"


@pytest.mark.parametrize(
    "env_name,value,expected",
    [
        ("CUDA_VERSION", "12.4", 12),
        ("CUDA_VERSION", " 13.0.1 ", 13),
        ("CUDA_TOOLKIT_VERSION", " 11.8 ", 11),
    ],
)
def test_detect_cuda_major_from_version_environment(
        monkeypatch, env_name, value, expected):
    clear_cuda_env(monkeypatch)
    monkeypatch.setenv(env_name, value)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(cuda="9.9")))
    with patch.object(ctypes.util, "find_library", return_value=None):
        assert cp._detect_cuda_major() == expected


def test_detect_cuda_major_prefers_first_valid_version_variable(monkeypatch):
    clear_cuda_env(monkeypatch)
    monkeypatch.setenv("CUDA_VERSION", "not-a-version")
    monkeypatch.setenv("CUDA_TOOLKIT_VERSION", "12.9")
    assert cp._detect_cuda_major() == 12


def test_detect_cuda_major_from_torch_after_invalid_environment(monkeypatch):
    clear_cuda_env(monkeypatch)
    monkeypatch.setenv("CUDA_VERSION", "unknown")
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(cuda=" 12.6 "))
    )
    with patch.object(ctypes.util, "find_library", return_value=None):
        assert cp._detect_cuda_major() == 12


def test_detect_cuda_major_ignores_non_string_torch_and_tolerates_broken_version(monkeypatch):
    clear_cuda_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(cuda=12)))
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(cp.os.path, "isfile", return_value=False):
        assert cp._detect_cuda_major() is None

    class BrokenVersion:
        @property
        def cuda(self):
            raise RuntimeError("broken torch")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(version=BrokenVersion()))
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(cp.os.path, "isfile", return_value=False):
        assert cp._detect_cuda_major() is None


def test_detect_cuda_major_from_cudart_soname(monkeypatch):
    clear_cuda_env(monkeypatch)
    with patch.object(ctypes.util, "find_library", return_value="libcudart.so.12.10"):
        assert cp._detect_cuda_major() == 12


def test_detect_cuda_major_filesystem_prefers_newer_major(monkeypatch, tmp_path):
    clear_cuda_env(monkeypatch)
    lib64 = tmp_path / "lib64"
    lib64.mkdir()
    (lib64 / "libcudart.so.11").touch()
    (lib64 / "libcudart.so.13").touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    with patch.object(ctypes.util, "find_library", return_value=None):
        assert cp._detect_cuda_major() == 13


def test_detect_cuda_major_uses_cuda_path_when_home_is_unset(monkeypatch, tmp_path):
    clear_cuda_env(monkeypatch)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libcudart.so.12").touch()
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    with patch.object(ctypes.util, "find_library", return_value=None):
        assert cp._detect_cuda_major() == 12


def test_set_cuda_graph_trace_leaves_inherited_value_without_flag(monkeypatch):
    monkeypatch.setenv(cp.CUDA_GRAPH_TRACE_ENV_VAR, "node")
    for value in (None, "", "node"):
        cp._set_cuda_graph_trace(value)
        assert os.environ[cp.CUDA_GRAPH_TRACE_ENV_VAR] == "node"


def test_set_cuda_graph_trace_explicit_value_overrides_inherited(monkeypatch):
    monkeypatch.setenv(cp.CUDA_GRAPH_TRACE_ENV_VAR, "node")
    cp._set_cuda_graph_trace("graph")
    assert os.environ[cp.CUDA_GRAPH_TRACE_ENV_VAR] == "graph"


def test_packaged_cupti_path_selects_arch_and_cuda_directory(tmp_path):
    library = tmp_path / "_native" / "arm64-cu13" / "libgscuptiprof.so"
    library.parent.mkdir(parents=True)
    library.touch()
    resources = types.SimpleNamespace(
        files=lambda package: ResourceStub(str(tmp_path), package),
        as_file=lambda candidate: contextlib.nullcontext(Path(str(candidate))),
    )
    with patch("importlib.resources.files", resources.files), \
         patch("importlib.resources.as_file", resources.as_file), \
         patch.object(cp, "_detect_cuda_major", return_value=13), \
         patch.object(cp, "_detect_arch_tag", return_value="arm64"):
        assert cp._packaged_cupti_so_path() == str(library)


def test_packaged_cupti_path_returns_none_without_detected_cuda():
    with patch.object(cp, "_detect_cuda_major", return_value=None):
        assert cp._packaged_cupti_so_path() is None


def test_packaged_cupti_path_returns_none_when_resource_missing(tmp_path):
    with patch("importlib.resources.files", return_value=ResourceStub(str(tmp_path), "graphsignal")), \
         patch("importlib.resources.as_file", lambda candidate: contextlib.nullcontext(Path(str(candidate)))), \
         patch.object(cp, "_detect_cuda_major", return_value=12), \
         patch.object(cp, "_detect_arch_tag", return_value="amd64"):
        assert cp._packaged_cupti_so_path() is None


def test_packaged_cupti_path_swallows_resource_errors():
    with patch("importlib.resources.files", side_effect=ImportError("broken resources")):
        assert cp._packaged_cupti_so_path() is None


def test_ensure_injection_path_preserves_existing_value(monkeypatch):
    monkeypatch.setattr(cp.sys, "platform", "linux")
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/inherited/cupti.so")
    with patch.object(cp, "_packaged_cupti_so_path") as packaged:
        assert cp._ensure_cuda_injection64_path() == "/inherited/cupti.so"
    packaged.assert_not_called()


def test_ensure_injection_path_publishes_packaged_value(monkeypatch):
    monkeypatch.setattr(cp.sys, "platform", "linux")
    with patch.object(cp, "_packaged_cupti_so_path", return_value="/pkg/libgscuptiprof.so"):
        assert cp._ensure_cuda_injection64_path() == "/pkg/libgscuptiprof.so"
    assert os.environ["CUDA_INJECTION64_PATH"] == "/pkg/libgscuptiprof.so"


def test_ensure_injection_path_rejects_non_linux_and_missing_package(monkeypatch):
    monkeypatch.setattr(cp.sys, "platform", "darwin")
    assert cp._ensure_cuda_injection64_path() is None

    monkeypatch.setattr(cp.sys, "platform", "linux")
    with patch.object(cp, "_packaged_cupti_so_path", return_value=None):
        assert cp._ensure_cuda_injection64_path() is None
    assert "CUDA_INJECTION64_PATH" not in os.environ


def test_libcupti_prefers_exact_major_soname_and_prepends_idempotently(tmp_path, monkeypatch):
    lib_dir = tmp_path / "extras" / "CUPTI" / "lib64"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libcupti.so.12").touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", "::/existing/lib:")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(importlib.util, "find_spec", return_value=None), \
         patch.object(importlib.metadata, "distribution", side_effect=importlib.metadata.PackageNotFoundError):
        assert cp._ensure_libcupti_ld_library_path(prefer_major=12)
    assert os.environ["LD_LIBRARY_PATH"] == f"{lib_dir}:/existing/lib"

    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(importlib.util, "find_spec", return_value=None), \
         patch.object(importlib.metadata, "distribution", side_effect=importlib.metadata.PackageNotFoundError), \
         patch.object(cp.os.path, "isdir", side_effect=lambda path: path == str(lib_dir)), \
         patch.object(cp.os, "listdir", return_value=["libcupti.so.12"]):
        assert cp._ensure_libcupti_ld_library_path(prefer_major=12)
    assert os.environ["LD_LIBRARY_PATH"] == f"{lib_dir}:/existing/lib"


def test_libcupti_python_package_directory(tmp_path, monkeypatch):
    package_dir = tmp_path / "site-packages" / "nvidia" / "cuda_cupti"
    lib_dir = package_dir / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libcupti.so.12").touch()
    spec = types.SimpleNamespace(submodule_search_locations={str(package_dir)})
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(importlib.util, "find_spec", return_value=spec):
        assert cp._ensure_libcupti_ld_library_path(prefer_major=12)
    assert os.environ["LD_LIBRARY_PATH"] == f"{lib_dir}:/existing"


def test_libcupti_metadata_path_prefers_matching_distribution(tmp_path, monkeypatch):
    matching_lib = tmp_path / "cu12" / "libcupti.so.12"
    generic_lib = tmp_path / "generic" / "libcupti.so.13"
    matching_lib.parent.mkdir()
    generic_lib.parent.mkdir()
    matching_lib.touch()
    requested = []

    def distribution(name):
        requested.append(name)
        if name == "nvidia-cuda-cupti-cu12":
            dist = Mock(files=["nvidia/cuda_cupti/lib/libcupti.so.12"])
            dist.locate_file.side_effect = lambda file: matching_lib
            return dist
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(importlib.util, "find_spec", return_value=None), \
         patch.object(importlib.metadata, "distribution", side_effect=distribution):
        assert cp._ensure_libcupti_ld_library_path(prefer_major=12)
    assert requested
    assert os.environ["LD_LIBRARY_PATH"] == f"{matching_lib.parent}:/existing"


def test_libcupti_sdk_search_order_and_unversioned_acceptance(tmp_path, monkeypatch):
    extras = tmp_path / "extras" / "CUPTI" / "lib64"
    extras.mkdir(parents=True)
    (extras / "libcupti.so").touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(importlib.util, "find_spec", return_value=None), \
         patch.object(importlib.metadata, "distribution", side_effect=importlib.metadata.PackageNotFoundError):
        assert cp._ensure_libcupti_ld_library_path(prefer_major=11)
    assert os.environ["LD_LIBRARY_PATH"] == str(extras)


def test_libcupti_returns_true_when_loader_already_finds_library(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unchanged")
    with patch.object(ctypes.util, "find_library", return_value="libcupti.so.12"):
        assert cp._ensure_libcupti_ld_library_path(prefer_major=12)
    assert os.environ["LD_LIBRARY_PATH"] == "/unchanged"


def test_libcupti_returns_false_when_nothing_is_found(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    with patch.object(ctypes.util, "find_library", return_value=None), \
         patch.object(importlib.util, "find_spec", return_value=None), \
         patch.object(importlib.metadata, "distribution", side_effect=importlib.metadata.PackageNotFoundError):
        assert cp._ensure_libcupti_ld_library_path(prefer_major=12) is False


def test_setup_sets_graph_trace_before_platform_check(monkeypatch):
    monkeypatch.setenv(cp.CUDA_GRAPH_TRACE_ENV_VAR, "node")
    monkeypatch.setattr(cp.sys, "platform", "darwin")
    with patch.object(cp, "_detect_cuda_major") as detect:
        assert cp.CuptiProfiler.setup_env_vars("graph") is False
    detect.assert_not_called()
    assert os.environ[cp.CUDA_GRAPH_TRACE_ENV_VAR] == "graph"


def test_setup_reports_each_setup_failure(monkeypatch):
    monkeypatch.setattr(cp.sys, "platform", "linux")
    with patch.object(cp, "_detect_cuda_major", return_value=None):
        assert cp.CuptiProfiler.setup_env_vars() is False

    with patch.object(cp, "_detect_cuda_major", return_value=12), \
         patch.object(cp, "_ensure_cuda_injection64_path", return_value=None), \
         patch.object(cp, "_ensure_libcupti_ld_library_path") as libcupti:
        assert cp.CuptiProfiler.setup_env_vars() is False
    libcupti.assert_not_called()

    with patch.object(cp, "_detect_cuda_major", return_value=13), \
         patch.object(cp, "_ensure_cuda_injection64_path", return_value="/injection.so"), \
         patch.object(cp, "_ensure_libcupti_ld_library_path", return_value=False):
        assert cp.CuptiProfiler.setup_env_vars() is False

    with patch.object(cp, "_detect_cuda_major", return_value=13) as detect, \
         patch.object(cp, "_ensure_cuda_injection64_path", return_value="/injection.so") as injection, \
         patch.object(cp, "_ensure_libcupti_ld_library_path", return_value=True) as libcupti:
        assert cp.CuptiProfiler.setup_env_vars() is True
    detect.assert_called_once_with()
    injection.assert_called_once_with()
    libcupti.assert_called_once_with(prefer_major=13)
