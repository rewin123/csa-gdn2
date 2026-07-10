"""Tiny append-only run logger for the live dashboard.

Training calls `lg.point(...)` at each eval and `lg.status(...)` on phase
changes. Both write to a directory the local `scripts/dash.sh` poller mirrors to
the Mac, where `dashboard.html` renders it live. No deps, line-buffered, safe to
call from a tight loop.
"""
from __future__ import annotations

import json
import os
import time


class RunLogger:
    def __init__(self, resdir: str, run_name: str = "rank7"):
        os.makedirs(resdir, exist_ok=True)
        self.resdir = resdir
        self.run_name = run_name
        self.progress_path = os.path.join(resdir, "progress.jsonl")
        self.status_path = os.path.join(resdir, "status.json")
        self._t0 = time.time()
        self._f = open(self.progress_path, "a", buffering=1)   # line-buffered

    def point(self, *, cand, seed, step, total_steps, recall, loss=None, gamma=None, **extra):
        """One evaluation point for a (candidate, seed) training curve."""
        rec = {"kind": "point", "cand": cand, "seed": seed, "step": step,
               "total_steps": total_steps, "recall": recall, "loss": loss,
               "gamma": gamma, "ts": time.time(), **extra}
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def status(self, *, phase, cand=None, seed=None, done=0, total=0, note="", **extra):
        """Overall progress snapshot (which candidate/seed is running, counts)."""
        rec = {"kind": "status", "run": self.run_name, "phase": phase, "cand": cand,
               "seed": seed, "done": done, "total": total, "note": note,
               "elapsed_s": round(time.time() - self._t0, 1), "ts": time.time(), **extra}
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as g:
            json.dump(rec, g)
        os.replace(tmp, self.status_path)         # atomic; poller never sees a half file

    def event(self, msg, **extra):
        rec = {"kind": "event", "msg": msg, "ts": time.time(), **extra}
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def reset(self):
        """Truncate the progress log (call at the very start of a fresh run)."""
        self._f.close()
        self._f = open(self.progress_path, "w", buffering=1)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
