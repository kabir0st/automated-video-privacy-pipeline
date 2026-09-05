"""Execution-provider selection.

The rule that matters here: ONNX Runtime *advertises* an execution provider
whenever its plugin ships in the wheel, without checking that the plugin's own
dependencies resolve. Trusting that list is actively harmful — `fp16_model_path`
converts the graph on the strength of a GPU being available, so a provider that
lists but fails to load leaves an fp16 graph running on the CPU, which measured
71 ms/pass against 26 ms for plain fp32. A broken GPU is slower than no GPU.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libs import utils                                          # noqa: E402


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    utils._provider_cache.clear()
    yield
    utils._provider_cache.clear()


def _fake(monkeypatch, available, usable=()):
    """Pretend ORT advertises `available` and that only `usable` really load."""
    class _ORT:
        @staticmethod
        def get_available_providers():
            return list(available)
    monkeypatch.setitem(sys.modules, "onnxruntime", _ORT)
    monkeypatch.setattr(utils, "_provider_usable", lambda n: n in usable)


ALL_GPU = ("DmlExecutionProvider", "CUDAExecutionProvider",
           "MIGraphXExecutionProvider", "ROCMExecutionProvider")


class TestSelection:
    def test_cpu_only_box(self, monkeypatch):
        _fake(monkeypatch, ["CPUExecutionProvider"], ["CPUExecutionProvider"])
        assert utils.best_onnx_providers() == ["CPUExecutionProvider"]

    def test_azure_is_not_a_gpu(self, monkeypatch):
        """ORT's stock CPU wheel advertises AzureExecutionProvider. Treating
        'not CPU' as 'GPU' would enable fp16 on a CPU-only machine."""
        _fake(monkeypatch, ["AzureExecutionProvider", "CPUExecutionProvider"],
              ["CPUExecutionProvider"])
        assert utils.best_onnx_providers() == ["CPUExecutionProvider"]
        assert not utils.has_gpu_provider()

    def test_unloadable_gpu_provider_is_dropped(self, monkeypatch):
        _fake(monkeypatch,
              ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
              ["CPUExecutionProvider"])          # advertised but won't load
        assert utils.best_onnx_providers() == ["CPUExecutionProvider"]
        assert not utils.has_gpu_provider(), \
            "an unloadable GPU must not enable the fp16 path"

    def test_working_gpu_provider_is_used(self, monkeypatch):
        _fake(monkeypatch,
              ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
              ["MIGraphXExecutionProvider", "CPUExecutionProvider"])
        assert utils.best_onnx_providers()[0] == "MIGraphXExecutionProvider"
        assert utils.has_gpu_provider()

    @pytest.mark.parametrize("expected", ALL_GPU)
    def test_every_gpu_provider_is_recognised(self, monkeypatch, expected):
        _fake(monkeypatch, [expected, "CPUExecutionProvider"],
              [expected, "CPUExecutionProvider"])
        assert utils.best_onnx_providers()[0] == expected
        assert utils.has_gpu_provider()
        assert expected in utils.GPU_PROVIDERS

    def test_preference_order(self, monkeypatch):
        avail = list(ALL_GPU) + ["CPUExecutionProvider"]
        _fake(monkeypatch, avail, avail)
        assert utils.best_onnx_providers() == avail

    def test_migraphx_preferred_over_plain_rocm(self, monkeypatch):
        """AMD ships onnxruntime_migraphx rather than onnxruntime_rocm from
        ROCm 7.1 on, and MIGraphX compiles the graph ahead of time."""
        avail = ["ROCMExecutionProvider", "MIGraphXExecutionProvider",
                 "CPUExecutionProvider"]
        _fake(monkeypatch, avail, avail)
        assert utils.best_onnx_providers()[0] == "MIGraphXExecutionProvider"

    def test_never_returns_empty(self, monkeypatch):
        _fake(monkeypatch, ["SomethingExoticExecutionProvider"], [])
        assert utils.best_onnx_providers() == ["CPUExecutionProvider"]


class TestMIGraphXCache:
    def test_cache_configured_when_migraphx_wins(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ORT_MIGRAPHX_MODEL_CACHE_PATH", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        avail = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
        _fake(monkeypatch, avail, avail)
        utils.best_onnx_providers()
        import os
        cache = os.environ.get("ORT_MIGRAPHX_MODEL_CACHE_PATH")
        assert cache and Path(cache).is_dir()

    def test_user_choice_is_not_overridden(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORT_MIGRAPHX_MODEL_CACHE_PATH", "/my/own/cache")
        avail = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
        _fake(monkeypatch, avail, avail)
        utils.best_onnx_providers()
        import os
        assert os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] == "/my/own/cache"

    def test_not_configured_when_migraphx_not_selected(self, monkeypatch):
        monkeypatch.delenv("ORT_MIGRAPHX_MODEL_CACHE_PATH", raising=False)
        _fake(monkeypatch, ["CPUExecutionProvider"], ["CPUExecutionProvider"])
        utils.best_onnx_providers()
        import os
        assert "ORT_MIGRAPHX_MODEL_CACHE_PATH" not in os.environ


class TestCpuThreads:
    def test_positive_and_bounded(self):
        import os
        n = utils._cpu_threads()
        assert 1 <= n <= (os.cpu_count() or 2)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AVPP_CPU_THREADS", "3")
        assert utils._cpu_threads() == 3

    @pytest.mark.parametrize("bad", ["0", "-4", "banana", ""])
    def test_bad_override_ignored(self, monkeypatch, bad):
        monkeypatch.setenv("AVPP_CPU_THREADS", bad)
        assert utils._cpu_threads() >= 1
