from __future__ import annotations

from typing import Any, Dict, Optional


class WandBLogger:
    """Optional, failure-safe wrapper around wandb.

    The project keeps running if wandb is unavailable, not logged in,
    or disabled by config. All methods become no-ops in that case.
    """

    def __init__(self, enabled: bool = False, config: Optional[Dict[str, Any]] = None):
        self.enabled = enabled
        self.config = config or {}
        self._run = None
        self._available = False
        self._wandb = None

        if not enabled:
            return

        try:
            import wandb  # type: ignore
        except Exception:
            return

        self._wandb = wandb
        self._available = True

    def init(self, **kwargs):
        if not self.enabled or not self._available:
            return None
        if self._run is not None:
            return self._run

        init_kwargs = dict(kwargs)
        init_kwargs.setdefault("config", self.config)
        self._run = self._wandb.init(**init_kwargs)
        return self._run

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled or not self._available or self._run is None:
            return
        if step is None:
            self._wandb.log(data)
        else:
            self._wandb.log(data, step=step)

    def finish(self):
        if not self.enabled or not self._available or self._run is None:
            return
        try:
            self._wandb.finish()
        finally:
            self._run = None
