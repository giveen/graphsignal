import logging
import os
import tempfile

from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import resolve_executable as _resolve
from graphsignal.launchers.supervisor import launch_supervised
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')

_NINFER_NAMES = {'ninfer', 'ninfer-serve', 'ninfer-perplexity'}
_REQUEST_LOG_FLAG = '--request-log-jsonl'
_REQUEST_LOG_ENV = 'GRAPHSIGNAL_NINFER_JSONL'
_NVTX_ENV = 'GRAPHSIGNAL_NINFER_NVTX'


class NinferLauncher(BaseLauncher):
    def match(self) -> bool:
        return self.executable_name() in _NINFER_NAMES

    def launch(self) -> None:
        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)

        args = list(self.args)
        # NInfer creates its static NVTX domain before CUDA loads the injected
        # library, so CUPTI cannot observe domain creation. This launcher-owned
        # opt-in tells the native adapter that domain handles may be adopted;
        # generic workloads remain strict and only accept an observed domain.
        os.environ[_NVTX_ENV] = '1'
        # NInfer uses header-only NVTX v3. CUDA injection alone initializes
        # CUPTI, but NVTX v3 has a separate per-library discovery path. Point
        # it at the same profiler library; its exported InitializeInjectionNvtx
        # forwards the export-table callback to CUPTI.
        cuda_injection = os.environ.get('CUDA_INJECTION64_PATH')
        if cuda_injection:
            os.environ.setdefault('NVTX_INJECTION64_PATH', cuda_injection)
        temporary_log = None
        try:
            if self.executable_name() == 'ninfer-serve':
                args, request_log, request_log_supplied = _ensure_request_log(args)
                if not request_log_supplied:
                    fd, temporary_log = tempfile.mkstemp(
                        prefix='graphsignal-ninfer-', suffix='.jsonl')
                    os.close(fd)
                    request_log = temporary_log
                    args.extend((_REQUEST_LOG_FLAG, request_log))
                if request_log is not None:
                    os.environ[_REQUEST_LOG_ENV] = request_log

            executable = _resolve(args[0])
            if not executable:
                raise FileNotFoundError(f'executable not found: {args[0]}')

            logger.debug('NinferLauncher launch: %s %s', executable, args)
            launch_supervised(
                [executable] + args[1:],
                metrics_port=None,
                listen_host=self.listen_host,
                listen_port=self.listen_port)
        finally:
            if temporary_log is not None:
                try:
                    os.remove(temporary_log)
                except FileNotFoundError:
                    pass


def _ensure_request_log(args):
    """Return argv, its request-log path, and whether the user supplied one."""
    args = list(args)
    for i, arg in enumerate(args):
        if arg == _REQUEST_LOG_FLAG:
            if i + 1 < len(args):
                return args, args[i + 1], True
            return args, None, True
        if arg.startswith(_REQUEST_LOG_FLAG + '='):
            return args, arg.split('=', 1)[1], True
    return args, None, False
