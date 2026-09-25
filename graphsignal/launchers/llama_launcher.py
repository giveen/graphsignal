import logging

from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    resolve_executable as _resolve, resolve_metrics_host, resolve_metrics_port)
from graphsignal.launchers.supervisor import launch_supervised
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')

_LLAMA_NAMES = {'llama-server'}
DEFAULT_SERVE_PORT = 8080
DEFAULT_SERVE_HOST = '127.0.0.1'
DEFAULT_METRICS_PATH = '/metrics'


class LlamaLauncher(BaseLauncher):
    """Configure llama.cpp's server for Graphsignal's Prometheus reader.

    llama-server already exposes the useful engine metrics at ``/metrics`` when
    ``--metrics`` is enabled. Keep the workload argv otherwise byte-for-byte
    intact; the flag is only added when the user did not provide it.
    """

    def match(self) -> bool:
        return self.executable_name() in _LLAMA_NAMES

    def launch(self) -> None:
        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)

        args = list(self.args)
        _warn_unsupported_metrics_modes(args)
        if not _has_flag(args, '--metrics'):
            args.append('--metrics')

        metrics_port = resolve_metrics_port(
            self.metrics_port, args, default=DEFAULT_SERVE_PORT)
        metrics_host = _scrapable_host(resolve_metrics_host(
            None, args, default=DEFAULT_SERVE_HOST))
        metrics_path = _metrics_path(args)

        executable = _resolve(args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {args[0]}')

        logger.debug('LlamaLauncher launch: %s %s', executable, args)
        launch_supervised(
            [executable] + args[1:],
            metrics_port=metrics_port,
            metrics_path=metrics_path,
            metrics_host=metrics_host,
            listen_host=self.listen_host,
            listen_port=self.listen_port)


def _scrapable_host(host: str) -> str:
    # Wildcard binds describe the server socket, not a usable destination.
    if host == '0.0.0.0':
        return '127.0.0.1'
    if host == '::':
        return '::1'
    return host


def _metrics_path(args) -> str:
    prefix = None
    for i, arg in enumerate(args):
        if arg == '--api-prefix' and i + 1 < len(args):
            prefix = args[i + 1]
        elif arg.startswith('--api-prefix='):
            prefix = arg.split('=', 1)[1]
    prefix = (prefix or '').strip()
    if prefix in ('', '/'):
        return DEFAULT_METRICS_PATH
    return '/' + prefix.strip('/') + DEFAULT_METRICS_PATH


def _warn_unsupported_metrics_modes(args) -> None:
    for flag, message in (
            ('--api-key', 'llama.cpp requires authentication for /metrics; '
                          'Graphsignal cannot scrape it without an API key'),
            ('--api-keys-file', 'llama.cpp may require authentication for /metrics; '
                                'Graphsignal cannot scrape it without credentials'),
            ('--ssl-key-file', 'llama.cpp TLS cannot be scraped by the current '
                               'HTTP Prometheus reader')):
        if _has_flag(args, flag):
            logger.warning('%s; engine metrics may be unavailable', message)


def _has_flag(args, flag) -> bool:
    for arg in args:
        if arg == flag or arg.startswith(flag + '='):
            return True
    return False
