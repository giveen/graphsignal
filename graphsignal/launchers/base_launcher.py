import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger('graphsignal')


class BaseLauncher(ABC):
    def __init__(self, args: List[str],
                 metrics_port: Optional[int] = None,
                 listen_host: Optional[str] = None,
                 listen_port: Optional[int] = None):
        self.args: List[str] = list(args)
        # Explicit Prometheus scrape port from `graphsignal-run --metrics-port`.
        # When None, each launcher derives it from the engine's --port/default.
        self.metrics_port: Optional[int] = metrics_port
        # Bind address for the watcher's /signals endpoint, from
        # `graphsignal-run --listen-host`/`--listen-port`. When None, the
        # watcher defaults apply (127.0.0.1:18259).
        self.listen_host: Optional[str] = listen_host
        self.listen_port: Optional[int] = listen_port

    @abstractmethod
    def match(self) -> bool:
        ...

    @abstractmethod
    def launch(self) -> None:
        ...

    def executable_name(self) -> str:
        if not self.args:
            return ''
        return os.path.basename(self.args[0])
