import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import List, Optional

logger = logging.getLogger('graphsignal')


def resolve_executable(name: str) -> Optional[str]:
    """Resolve a command name to an executable path (absolute path as-is,
    otherwise PATH lookup). Returns None when unresolvable."""
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return shutil.which(name)


def extract_host(args: List[str], default: Optional[str] = None) -> Optional[str]:
    """Extract the HTTP bind host from engine CLI args (`--host H` / `--host=H`)."""
    for i, arg in enumerate(args):
        if arg == '--host':
            if i + 1 < len(args):
                return args[i + 1]
            return default
        if arg.startswith('--host='):
            return arg.split('=', 1)[1] or default
    return default


def extract_port(args: List[str], default: Optional[int] = None) -> Optional[int]:
    """Extract the serving port from engine CLI args (`--port N` / `--port=N`).

    Returns ``default`` when no `--port` is present or the value isn't an int.
    """
    for i, arg in enumerate(args):
        if arg == '--port':
            if i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except (TypeError, ValueError):
                    return default
        elif arg.startswith('--port='):
            try:
                return int(arg.split('=', 1)[1])
            except (TypeError, ValueError):
                return default
    return default


def resolve_metrics_port(metrics_port: Optional[int], args: List[str],
                         default: Optional[int] = None) -> Optional[int]:
    """Pick the Prometheus scrape port for a launcher.

    An explicit `--metrics-port` (forwarded from `graphsignal-run`) always wins;
    otherwise fall back to the engine's serving port (`--port`) or its default.
    """
    if metrics_port is not None:
        return metrics_port
    return extract_port(args, default=default)


def resolve_metrics_host(metrics_host: Optional[str], args: List[str],
                         default: Optional[str] = None) -> Optional[str]:
    """Pick the Prometheus scrape host for a launcher.

    An explicit scrape host always wins; otherwise use the engine's ``--host``
    or its default.
    """
    if metrics_host is not None:
        return metrics_host
    return extract_host(args, default=default)


def start_watcher(pid: int,
                  metrics_port: Optional[int] = None,
                  metrics_path: Optional[str] = None,
                  metrics_host: Optional[str] = None,
                  listen_host: Optional[str] = None,
                  listen_port: Optional[int] = None) -> Optional[subprocess.Popen]:
    """Spawn the graphsignal-watch subprocess to observe the given pid."""
    cmd = [
        sys.executable,
        '-m', 'graphsignal.commands.graphsignal_watch',
        '--pid', str(int(pid)),
    ]
    if metrics_port is not None:
        cmd.extend(['--metrics-port', str(int(metrics_port))])
    if metrics_path is not None:
        cmd.extend(['--metrics-path', str(metrics_path)])
    if metrics_host is not None:
        cmd.extend(['--metrics-host', str(metrics_host)])
    if listen_host is not None:
        cmd.extend(['--listen-host', str(listen_host)])
    if listen_port is not None:
        cmd.extend(['--listen-port', str(int(listen_port))])

    # The child runs with stderr discarded, so a failure inside it — a port
    # already in use, say — is invisible from out here. Give it a pipe to
    # report the /signals endpoint's bind state on, and log what it says.
    read_fd, write_fd = os.pipe()
    cmd.extend(['--status-fd', str(write_fd)])
    logger.debug('Starting graphsignal-watch: %s', ' '.join(cmd))
    try:
        process = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(write_fd,),
        )
    except Exception as exc:
        logger.error('Failed to start graphsignal-watch: %s', exc, exc_info=True)
        os.close(read_fd)
        return None
    finally:
        # The child owns the write end now; dropping ours lets the reader
        # thread see EOF when it exits.
        os.close(write_fd)

    threading.Thread(
        target=_log_status_lines, args=(read_fd,),
        name='graphsignal-watch-status', daemon=True).start()
    return process


def _log_status_lines(read_fd: int) -> None:
    """Relay the watcher's status lines to this process's log.

    A bind failure is a warning: the user has no /signals endpoint and the
    reason is otherwise not in the workload's log. Everything else is debug.
    """
    try:
        stream = os.fdopen(read_fd, 'r')
    except OSError:
        # The pipe is gone (interpreter teardown, an unusual close_fds setup).
        # Losing status relay is not worth an unhandled thread exception.
        logger.debug('graphsignal-watch status pipe unavailable', exc_info=True)
        return
    with stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith('signals-endpoint bind_failed'):
                logger.warning('graphsignal-watch: %s', line)
            else:
                logger.debug('graphsignal-watch: %s', line)
