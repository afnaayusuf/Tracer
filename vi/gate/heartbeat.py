from __future__ import annotations


class HeartbeatScheduler:
    """R10 / E-GATE-01 / E-GATE-05: the motion gate is blind to whatever is not moving, so a
    full-frame detection runs on a clock the gate cannot suppress: at episode open, every
    `active_ms` while the tile has live tubes, every `quiet_ms` otherwise. Asset-home zones are
    checked on every heartbeat. The scheduler is pure so it can be tested without video."""

    def __init__(self, active_ms: int = 1000, quiet_ms: int = 10_000, fire_at_open: bool = True):
        self.active_ms = active_ms
        self.quiet_ms = quiet_ms
        self.fire_at_open = fire_at_open
        self._last_ms: int | None = None
        self.fired = 0

    def due(self, t_ms: int, tile_active: bool) -> bool:
        if self._last_ms is None:
            if self.fire_at_open:
                self._last_ms = t_ms
                self.fired += 1
                return True
            self._last_ms = t_ms
            return False
        period = self.active_ms if tile_active else self.quiet_ms
        if t_ms - self._last_ms >= period:
            self._last_ms = t_ms
            self.fired += 1
            return True
        return False
