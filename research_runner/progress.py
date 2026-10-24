"""Dependency-free progress, persistent state, and live subprocess log forwarding."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

_current = ContextVar('interactionguard_progress', default=None)
_fallback = None


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


class Reporter:
    def __init__(self, directory=None, *, stream=None, interval=1., clock=time.monotonic):
        self.directory = Path(directory) if directory else None
        self.stream = stream
        self.interval = interval
        self.clock = clock
        self.phases = {}
        self.last = {}
        self.lock = threading.RLock()
        self.last_activity = self.clock()

    def update(self, phase, current=None, total=None, *, message='', unit='items', status='running', force=False, display=True):
        if current is not None and (isinstance(current, bool) or current < 0):
            raise ValueError('Progress current must be nonnegative')
        if total is not None and (isinstance(total, bool) or total < 0 or (current is not None and current > total)):
            raise ValueError('Progress total must be >= current')
        if status not in {'running', 'completed', 'failed', 'cancelled', 'waiting', 'skipped'}:
            raise ValueError('Invalid progress status')
        with self.lock:
            now = self.clock()
            previous = self.phases.get(phase)
            if previous is None or (current is not None and previous['current'] is not None and current < previous['current']):
                previous = {'start': now, 'current': None, 'printed': -float('inf'), 'status': None}
                self.phases[phase] = previous
            changed = previous['status'] != status
            previous['current'], previous['status'] = current, status
            finished_count = current is not None and total is not None and current == total
            if not force and not changed and status == 'running' and not finished_count and now - previous['printed'] < self.interval:
                return None
            previous['printed'] = now
            self.last_activity = now
            elapsed = now - previous['start']
            fraction = current / total if current is not None and total else None
            eta = elapsed * (total-current) / current if current and total and status == 'running' else None
            record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                      'phase': str(phase), 'status': status, 'current': current, 'total': total,
                      'unit': unit, 'percent': fraction * 100 if fraction is not None else None,
                      'elapsed_seconds': elapsed, 'eta_seconds': eta, 'message': str(message)}
            self.last = record
            bar = ''
            if fraction is not None:
                fill = min(20, int(fraction*20))
                bar = f' [{"#"*fill}{"-"*(20-fill)}] {current}/{total} {unit} ({fraction*100:.1f}%)'
            elif current is not None:
                bar = f' {current} {unit}'
            suffix = f' | {message}' if message else ''
            eta_text = f' ETA {duration(eta)}' if eta is not None else ''
            text = f'[IG] {phase} | {status}{bar} | elapsed {duration(elapsed)}{eta_text}{suffix}'
            if display:
                print(text, file=self.stream or sys.stdout, flush=True)
            if self.directory:
                self.directory.mkdir(parents=True, exist_ok=True)
                encoded = json.dumps(record, ensure_ascii=True)
                temporary = self.directory / f'.progress-{os.getpid()}-{uuid.uuid4().hex}.tmp'
                try:
                    temporary.write_text(encoded + '\n')
                    os.replace(temporary, self.directory / 'progress.json')
                finally:
                    temporary.unlink(missing_ok=True)
                with (self.directory / 'progress.jsonl').open('a') as history:
                    history.write(encoded + '\n')
            return record


def reporter():
    global _fallback
    active = _current.get()
    if active is not None:
        return active
    directory = os.environ.get('IG_PROGRESS_DIR')
    if _fallback is None or str(_fallback.directory or '') != str(directory or ''):
        _fallback = Reporter(directory)
    return _fallback


def emit(phase, current=None, total=None, **kwargs):
    return reporter().update(phase, current, total, **kwargs)


@contextmanager
def scope(directory, phase, *, heartbeat_seconds=5.):
    active = Reporter(directory)
    token = _current.set(active)
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            with active.lock:
                if active.clock() - active.last_activity < heartbeat_seconds:
                    continue
                last = dict(active.last)
                active.update(last.get('phase', phase), last.get('current'), last.get('total'),
                              unit=last.get('unit', 'items'), status='waiting',
                              message='Still active; waiting for the current operation to finish.', force=True)

    active.update(phase, message='Starting; validating prerequisites and configuration.', force=True)
    watcher = threading.Thread(target=heartbeat, name='interactionguard-heartbeat', daemon=True)
    watcher.start()
    try:
        yield active
    except KeyboardInterrupt:
        stopped.set()
        watcher.join()
        active.update(phase, status='cancelled', message='Interrupted by user.', force=True)
        raise
    except FileExistsError:
        stopped.set()
        watcher.join()
        active.update(phase, status='waiting', message='Existing output requires explicit overwrite or cancellation.', force=True)
        raise
    except BaseException as error:
        stopped.set()
        watcher.join()
        active.update(phase, status='failed', message=type(error).__name__, force=True)
        raise
    else:
        stopped.set()
        watcher.join()
        active.update(phase, status='completed', force=True)
    finally:
        stopped.set()
        watcher.join()
        _current.reset(token)


def iterable(items, phase, total=None, unit='items'):
    if total is None:
        try:
            total = len(items)
        except TypeError:
            pass
    emit(phase, 0, total, unit=unit, force=True)
    count = 0
    for item in items:
        yield item
        count += 1
        emit(phase, count, total, unit=unit)
    emit(phase, count, total, unit=unit, status='completed', force=True)


def trainer_callback(base):
    """Build a Transformers callback without importing ML libraries in the parent."""
    class TrainingProgress(base):
        def __init__(self):
            self.eval_batches = 0

        def on_train_begin(self, args, state, control, **kwargs):
            emit('encoder.train', state.global_step, state.max_steps, unit='optimizer steps', force=True)

        def on_step_end(self, args, state, control, **kwargs):
            emit('encoder.train', state.global_step, state.max_steps, unit='optimizer steps',
                 message=f'Epoch={state.epoch}')

        def on_substep_end(self, args, state, control, **kwargs):
            emit('encoder.train', state.global_step, state.max_steps, unit='optimizer steps',
                 message=f'Epoch={state.epoch}; accumulating gradients')

        def on_prediction_step(self, args, state, control, **kwargs):
            self.eval_batches += 1
            loader = kwargs.get('eval_dataloader')
            total = len(loader) if loader is not None and hasattr(loader, '__len__') else None
            emit('encoder.validation', self.eval_batches, total, unit='batches')

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            emit('encoder.validation', self.eval_batches, unit='batches',
                 status='completed', message=str(metrics or {}), force=True)
            self.eval_batches = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            emit('encoder.train', state.global_step, state.max_steps, unit='optimizer steps',
                 message=str(logs or {}), force=True)

        def on_train_end(self, args, state, control, **kwargs):
            emit('encoder.train', state.global_step, state.max_steps, unit='optimizer steps', status='completed',
                 message='Training ended; early stopping may end before the planned step budget.', force=True)

    return TrainingProgress()


def logged_process(command, logfile, *, heartbeat_seconds=5., poll_seconds=.2, phase='Worker', **kwargs):
    """Tail a blocking subprocess's log while preserving subprocess.run semantics.

    A watcher provides a heartbeat even during model loading or a long batch.
    It never signals or terminates the worker, and is joined before returning.
    """
    path = Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    active = reporter()
    stopped = threading.Event()
    activity = {'offset': 0, 'last': time.monotonic()}

    def tail():
        pending = ''
        while True:
            if path.exists():
                with path.open(errors='replace') as stream:
                    stream.seek(activity['offset'])
                    data = stream.read()
                    activity['offset'] = stream.tell()
                pending += data
                lines = pending.split('\n')
                pending = lines.pop()
                for line in lines:
                    line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).strip()
                    if not line:
                        continue
                    activity['last'] = time.monotonic()
                    # Model outputs and full trace text must not flood the console.
                    if '[IG]' in line:
                        line = line[line.rfind('[IG]'):]
                        print(line[:1000], file=active.stream or sys.stdout, flush=True)
                        count = re.search(r'\] (\d+)/(\d+) (.*?) \(', line)
                        current, total, unit = (int(count[1]), int(count[2]), count[3]) if count else (None, None, 'items')
                        active.update(phase, current, total, unit=unit, message=line[:500], force=True, display=False)
                    elif any(marker in line for marker in ('epoch', 'loss', 'EMBEDDINGS', 'JUDGE ', 'ENCODER COMPLETE', 'AE SELECTION')):
                        active.update(phase, message=line[:500])
            if stopped.is_set():
                break
            if time.monotonic() - activity['last'] >= heartbeat_seconds:
                active.update(phase, status='waiting', message=f'Process is active; waiting for next batch/update. Log: {path}', force=True)
                activity['last'] = time.monotonic()
            stopped.wait(poll_seconds)

    active.update(phase, message=f'Launching process. Log: {path}', force=True)
    with path.open('w') as log:
        watcher = threading.Thread(target=tail, name='interactionguard-log-watch', daemon=True)
        watcher.start()
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        except BaseException:
            stopped.set()
            watcher.join()
            active.update(phase, status='failed', message=f'Process failed; see {path}', force=True)
            raise
        finally:
            stopped.set()
            watcher.join()
    returncode = getattr(result, 'returncode', 0)
    status = 'completed' if returncode == 0 else 'failed'
    active.update(phase, status=status, message=f'Process exited ({returncode}). Log: {path}', force=True)
    return result
