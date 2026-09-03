"""The one thing the profiler says to Graphsignal without an API key.

Once per run, the watcher asks whether the version it is running is still
current and logs a line when it is not. The request carries the version and
nothing else, the answer carries a version and nothing else, and
`GRAPHSIGNAL_DISABLE_VERSION_CHECK=1` turns it off.

## The notice is loud; a failure is silent

The notice is a warning, so it reaches stderr at the watcher's default level and
— because every `graphsignal` record is forwarded into the `LogStore` — also
lands in the `errors` array of `GET /signals`. That is deliberate: an agent
reading the payload can act on a stale profiler.

Everything that can go wrong instead says nothing above `debug`. A version check
is not part of anybody's workload, and a run that works must not look broken
because a network it never needed was unreachable.
"""

from typing import Optional
import logging
import re
import threading

from graphsignal.version import __version__
from graphsignal.watcher.env_vars import read_config_param, DEFAULT_API_BASE

logger = logging.getLogger('graphsignal')

# Short, because nothing waits on this. The uploader's 10s budget belongs to
# data somebody would miss; a version check that is slow is a version check not
# worth doing.
REQUEST_TIMEOUT_SEC = 3

_MAJOR_MINOR = re.compile(r'^(\d+)\.(\d+)(?:\.|$)')


def _major_minor(version) -> Optional[tuple]:
    """`1.2.3` -> `(1, 2)`, or None when it is not a version at all.

    Matches a prefix so suffixed releases (`1.2.0rc1`, `1.2.0+cu12`) read as the
    releases they are.
    """
    if not isinstance(version, str):
        return None
    match = _MAJOR_MINOR.match(version.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def newer_version(current: str, latest: Optional[str]) -> Optional[str]:
    """The version worth telling somebody about, or None.

    Compares `(major, minor)` only: a patch release is not worth interrupting a
    run for, and a notice that fires on every patch is one people learn to
    ignore.

    Applied to the server's answer rather than trusting it, so a server that
    answers with something older, equal or unreadable produces no notice.
    """
    from_version = _major_minor(current)
    to_version = _major_minor(latest)
    if not from_version or not to_version:
        return None
    return latest if to_version > from_version else None


def _fetch_latest_version(api_base: Optional[str]) -> Optional[str]:
    import requests

    base = (api_base or DEFAULT_API_BASE).rstrip('/')
    response = requests.get(
        f'{base}/api/v1/version_check',
        params={'version': __version__},
        timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    latest = response.json().get('latest_version')
    return latest if isinstance(latest, str) else None


def _run_version_check(api_base: Optional[str], stop_event: Optional[threading.Event]) -> None:
    # One bare catch around the whole body, deliberately: there is no failure
    # mode here that anybody should hear about, so there is nothing to tell
    # apart. Connection refused, a timeout, a 500, a body that is not JSON and a
    # version string that does not parse are all "say nothing".
    try:
        latest = _fetch_latest_version(api_base)

        # Checked after the request, not before: a check still in flight when
        # the target dies has nothing useful to say to a process on its way out.
        if stop_event is not None and stop_event.is_set():
            return

        newer = newer_version(__version__, latest)
        if newer:
            logger.warning(
                'A newer version of graphsignal is available: %s (running %s). '
                'Upgrade with: pip install --upgrade graphsignal',
                newer, __version__)
    except Exception:
        logger.debug('Version check failed', exc_info=True)


def start_version_check(
        api_base: Optional[str] = None,
        stop_event: Optional[threading.Event] = None) -> Optional[threading.Thread]:
    """Check for a newer release, on a thread of its own.

    Its own thread because the caller is the pid monitor's poll loop, where a
    blocking HTTP call would stall child-process discovery for as long as it
    takes. Daemon, so it can never hold the process open, and the 3s timeout
    bounds it anyway.

    Returns the thread, or None when the check is disabled.
    """
    if read_config_param('disable_version_check', bool, default_value=False):
        logger.debug('Version check disabled')
        return None

    thread = threading.Thread(
        target=_run_version_check,
        args=(api_base, stop_event),
        name='graphsignal-version-check',
        daemon=True)
    thread.start()
    return thread
