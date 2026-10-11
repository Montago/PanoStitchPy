"""Runs stitching jobs in a background thread and reports back to the GUI through Qt signals."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from panostitch.pipeline import Cancelled, Options, stitch


@dataclass
class Job:
    folder: Path
    output: Path


class StitchWorker(QObject):
    progress = Signal(float, str)      # overall fraction 0..1, message
    log = Signal(str)
    job_started = Signal(int)
    job_done = Signal(int, str)        # job index, output path
    job_failed = Signal(int, str)      # job index, error message
    finished = Signal(bool)            # True if cancelled

    def __init__(self, jobs: list[Job], options: Options):
        super().__init__()
        self._jobs = jobs
        self._opt = options
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        n = len(self._jobs)
        cancelled = False
        for idx, job in enumerate(self._jobs):
            if self._cancel:
                cancelled = True
                break
            self.job_started.emit(idx)
            t0 = time.time()
            prefix = f"[{idx + 1}/{n}] " if n > 1 else ""

            def progress(f: float, msg: str, idx=idx, prefix=prefix) -> None:
                self.progress.emit((idx + f) / n, prefix + msg)

            try:
                out = stitch(job.folder, job.output, self._opt, progress, lambda: self._cancel, self.log.emit)
                self.log.emit(f"Færdig på {time.time() - t0:.0f} sekunder")
                self.job_done.emit(idx, str(out))
            except Cancelled:
                self.log.emit("Annulleret")
                cancelled = True
                break
            except MemoryError:
                self.job_failed.emit(idx, "Ikke nok hukommelse – prøv en lavere kvalitet")
            except Exception as e:  # report and carry on with the next panorama
                self.log.emit(traceback.format_exc())
                self.job_failed.emit(idx, str(e) or e.__class__.__name__)
        self.finished.emit(cancelled)
