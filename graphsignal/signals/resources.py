import logging
import threading
import time

logger = logging.getLogger('graphsignal')

MAX_TAG_KEY_LEN = 50
MAX_TAG_VALUE_LEN = 250
MAX_ATTRIBUTE_VALUE_LEN = 2500
MAX_RESOURCES = 5000
RESOURCE_EXPIRY_NS = 600 * 1_000_000_000
CLEANUP_INTERVAL_NS = 60 * 1_000_000_000


class ResourceStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._resources = {}
        self._last_cleanup_ts = 0
        self._last_cap_log_ts = 0

    def update_resource(
            self,
            kind,
            tags=None,
            attributes=None,
            first_seen_ts=None,
            last_seen_ts=None):
        if kind is None:
            return

        now_ns = time.time_ns()
        if first_seen_ts is None:
            first_seen_ts = now_ns
        if last_seen_ts is None:
            last_seen_ts = now_ns

        norm_tags = {}
        if tags:
            for key, value in tags.items():
                norm_tags[str(key)[:MAX_TAG_KEY_LEN]] = str(value)[:MAX_TAG_VALUE_LEN]

        norm_attributes = {}
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                norm_attributes[str(key)[:MAX_TAG_KEY_LEN]] = str(value)[:MAX_ATTRIBUTE_VALUE_LEN]

        resource_key = (kind, frozenset(norm_tags.items()))

        with self._lock:
            self._maybe_cleanup(now_ns)
            existing = self._resources.get(resource_key)
            if existing is not None:
                first_seen_ts = min(existing['first_seen_ts'], first_seen_ts)
                last_seen_ts = max(existing['last_seen_ts'], last_seen_ts)
            elif len(self._resources) >= MAX_RESOURCES:
                if now_ns - self._last_cap_log_ts >= CLEANUP_INTERVAL_NS:
                    self._last_cap_log_ts = now_ns
                    logger.debug('Max resources reached (%d), dropping new resource: %s',
                                 MAX_RESOURCES, kind)
                return
            self._resources[resource_key] = {
                'kind': kind,
                'tags': norm_tags,
                'attributes': norm_attributes,
                'first_seen_ts': first_seen_ts,
                'last_seen_ts': last_seen_ts,
            }

    def export(self):
        """Snapshot of all resources; never resets the store."""
        with self._lock:
            return [
                {**resource,
                 'tags': dict(resource['tags']),
                 'attributes': dict(resource['attributes'])}
                for resource in self._resources.values()]

    def clear(self):
        with self._lock:
            self._resources.clear()

    def _maybe_cleanup(self, now_ns):
        if now_ns - self._last_cleanup_ts < CLEANUP_INTERVAL_NS:
            return
        self._last_cleanup_ts = now_ns
        cutoff = now_ns - RESOURCE_EXPIRY_NS
        expired = [
            key for key, resource in self._resources.items()
            if resource['last_seen_ts'] < cutoff
        ]
        for key in expired:
            del self._resources[key]
