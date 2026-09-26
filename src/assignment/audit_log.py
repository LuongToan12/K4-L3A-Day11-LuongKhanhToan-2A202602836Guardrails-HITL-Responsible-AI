"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        import time
        key = request_id or user_id
        self._open[key] = {
            "start_time": time.time(),
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "input": text,
            "request_id": request_id,
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        import time
        key = request_id or user_id
        open_entry = self._open.pop(key, None)
        latency_ms = 0.0
        timestamp = utc_now_iso()
        input_text = ""
        if open_entry:
            latency_ms = round((time.time() - open_entry.get("start_time", time.time())) * 1000, 2)
            timestamp = open_entry.get("timestamp", timestamp)
            input_text = open_entry.get("input", "")

        log_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "request_id": request_id,
            "input": input_text,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(log_entry)
        return log_entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path_str = filepath or default_audit_log_path()
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
