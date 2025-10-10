from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent


def _timestamp() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class TrainingManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._wait_thread: Optional[threading.Thread] = None
        self._on_complete: Optional[Callable[[], None]] = None
        self._logs: List[Dict[str, Any]] = []
        self._seq: int = 0
        self._status: Dict[str, Any] = {
            "state": "idle",
            "message": "Idle",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "return_code": None,
        }
        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_status(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)

    def _append_log(self, line: str) -> None:
        clean = line.rstrip("\n")
        with self._lock:
            entry = {
                "seq": self._seq,
                "timestamp": _timestamp(),
                "line": clean,
            }
            self._logs.append(entry)
            self._seq += 1
            # Keep log size reasonable (retain last 10k lines)
            if len(self._logs) > 10000:
                trim = len(self._logs) - 10000
                self._logs = self._logs[trim:]

    def _reader(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._append_log(line)
        finally:
            try:
                proc.stdout.close()
            except Exception:  # noqa: BLE001
                pass

    def _waiter(self, proc: subprocess.Popen[str]) -> None:
        return_code: Optional[int] = None
        try:
            return_code = proc.wait()
        finally:
            on_complete: Optional[Callable[[], None]] = None
            with self._lock:
                self._process = None
                self._stdout_thread = None
                self._wait_thread = None
                on_complete = self._on_complete
                self._on_complete = None
                finished_at = _timestamp()
                if return_code == 0:
                    state = "completed"
                    message = "Training completed"
                    error = None
                elif self._stop_requested:
                    state = "stopped"
                    message = "Training stopped by user"
                    error = None
                else:
                    state = "error"
                    message = f"Training failed (code {return_code})"
                    error = "See logs for details."
                self._status.update(
                    {
                        "state": state,
                        "message": message,
                        "finished_at": finished_at,
                        "error": error,
                        "return_code": return_code,
                    }
                )
                self._stop_requested = False
            if return_code == 0 and on_complete is not None:
                try:
                    on_complete()
                except Exception as exc:  # noqa: BLE001
                    self._append_log(f"[manager] on_complete callback failed: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_training(
        self,
        config: Dict[str, Any],
        on_complete: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._process and self._process.poll() is None:
                raise RuntimeError("Training already in progress")

            self._logs.clear()
            self._seq = 0
            self._status = {
                "state": "running",
                "message": "Training in progress",
                "started_at": _timestamp(),
                "finished_at": None,
                "error": None,
                "return_code": None,
            }
            self._stop_requested = False
            self._on_complete = on_complete

            cmd = [sys.executable, "-u", "train.py"]
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            proc = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._process = proc

            reader_thread = threading.Thread(target=self._reader, args=(proc,), daemon=True)
            waiter_thread = threading.Thread(target=self._waiter, args=(proc,), daemon=True)
            self._stdout_thread = reader_thread
            self._wait_thread = waiter_thread

            reader_thread.start()
            waiter_thread.start()

            return dict(self._status)

    def stop_training(self) -> Dict[str, Any]:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                raise RuntimeError("Training is not running")
            self._stop_requested = True
            proc = self._process
            self._status.update({"message": "Stopping training…"})

        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[manager] Failed to terminate process: {exc}")
        return self.status()

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def get_logs(self, since: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            if since < 0:
                since = 0
            entries = [entry for entry in self._logs if entry["seq"] >= since]
            next_seq = self._seq
            return entries, next_seq

    def reset(self) -> None:
        with self._lock:
            self._process = None
            self._stdout_thread = None
            self._wait_thread = None
            self._on_complete = None
            self._logs.clear()
            self._seq = 0
            self._status = {
                "state": "idle",
                "message": "Idle",
                "started_at": None,
                "finished_at": None,
                "error": None,
                "return_code": None,
            }
            self._stop_requested = False


manager = TrainingManager()
