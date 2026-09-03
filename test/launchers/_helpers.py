"""Shared test helpers for per-launcher test files.

`LaunchFixture` mocks the side effects that every launcher's `launch()` method
triggers (CUPTI/ROCm env setup, target resolution, and the final
`launch_supervised` handoff) so a test can assert on the launcher's argv
handling without touching the OS.
"""

from unittest.mock import patch


class LaunchFixture:
    """Shared mocks for launch() smoke tests across launchers.

    Pass the launcher module (e.g. ``graphsignal.launchers.vllm_launcher``)
    so the patches land on the symbols actually referenced by that module
    (each launcher imports `launch_supervised` / `_resolve` by name).
    """

    def __init__(self, module):
        self.module = module
        self.cupti_env = patch.object(
            module.CuptiProfiler, 'setup_env_vars', return_value=True)
        self.rocm_env = patch.object(
            module.RocmProfiler, 'setup_env_vars', return_value=True)
        self.launch_supervised = patch.object(module, 'launch_supervised')
        self.resolve = patch.object(module, '_resolve', return_value='/abs/exec')

    def __enter__(self):
        self.cupti_env_m = self.cupti_env.start()
        self.rocm_env_m = self.rocm_env.start()
        self.launch_supervised_m = self.launch_supervised.start()
        self.resolve_m = self.resolve.start()
        return self

    def __exit__(self, *exc):
        self.resolve.stop()
        self.launch_supervised.stop()
        self.rocm_env.stop()
        self.cupti_env.stop()

    @property
    def launched_argv(self):
        """The argv handed to launch_supervised (resolved executable first)."""
        return self.launch_supervised_m.call_args[0][0]
