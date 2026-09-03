import collections
import logging
import os
import threading
import time

from graphsignal import version

logger = logging.getLogger('graphsignal')

LEVELS = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3, 'critical': 4}


class LogStore:
    MAX_ENTRIES = 200
    MESSAGE_SIZE_LIMIT = 1024
    STACK_TRACE_SIZE_LIMIT = 4 * 1024

    def __init__(self):
        self._lock = threading.Lock()
        self._entries = collections.deque(maxlen=LogStore.MAX_ENTRIES)

    def log_message(
            self,
            *,
            tags=None,
            level='info',
            message=None,
            exception=None,
            timestamp_ns=None):
        # no logging in this function!
        if message is None:
            return
        if message and len(message) > self.MESSAGE_SIZE_LIMIT:
            return
        if exception and len(exception) > self.STACK_TRACE_SIZE_LIMIT:
            return

        entry = {
            'level': level.lower() if level else 'info',
            'ts': timestamp_ns if timestamp_ns else time.time_ns(),
            'message': message,
            'exception': exception if exception else None,
            'tags': dict(tags) if tags else {},
        }

        with self._lock:
            self._entries.append(entry)

    def log_watcher_message(
            self,
            tags=None,
            level=None,
            message=None,
            exception=None,
            timestamp_ns=None):
        all_tags = {}
        if tags is not None:
            all_tags.update(tags)
        all_tags['scope.name'] = 'watcher'
        all_tags['logger'] = 'graphsignal'
        all_tags['watcher.pid'] = os.getpid()

        self.log_message(
            tags=all_tags,
            level=level,
            message=f'Graphsignal {version.__version__}: {message}',
            exception=exception,
            timestamp_ns=timestamp_ns)

    def last_entries(self, min_level='warning', limit=10):
        min_rank = LEVELS.get(min_level, 2)
        with self._lock:
            matched = [e for e in self._entries
                       if LEVELS.get(e['level'], 1) >= min_rank]
        return matched[-limit:]

    def export(self):
        """Snapshot of all retained entries; never resets the store."""
        with self._lock:
            return [dict(e) for e in self._entries]

    def clear(self):
        with self._lock:
            self._entries.clear()
