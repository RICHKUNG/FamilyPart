"""多場景工作流程聚合執行。"""

from __future__ import annotations

import contextlib
import io
from collections import deque
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import WorkflowConfigError, WorkflowRuntimeError
from .scene_workflow import run_scene_workflow
from .scenes import expand_output_path_template, normalize_scene_list
from .utils import now_local_iso, now_local_stamp

from my3dis.common_utils import ENTRY_LOG_FORMAT, configure_entry_log_format

# Optional OOM monitoring (gracefully handle if not installed)
try:
    from oom_monitor.memory_events import MemoryEventsReader, OOMEvent
    HAS_OOM_MONITOR = True
except ImportError:
    HAS_OOM_MONITOR = False
    MemoryEventsReader = None
    OOMEvent = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SceneJob:
    index: int
    scene: str
    kwargs: Dict[str, Any]


@dataclass(frozen=True)
class _SceneProcessHandle:
    job: _SceneJob
    process: mp.Process
    conn: mp.connection.Connection


class _TeeStream(io.TextIOBase):
    """Write data to the scene log file and mirror to the original stream."""

    def __init__(self, primary: io.TextIOBase, mirror: io.TextIOBase | None) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:  # type: ignore[override]
        if self._primary:
            self._primary.write(data)
        if self._mirror:
            self._mirror.write(data)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if self._primary:
            self._primary.flush()
        if self._mirror:
            self._mirror.flush()

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def isatty(self) -> bool:  # type: ignore[override]
        if self._mirror and hasattr(self._mirror, 'isatty'):
            return self._mirror.isatty()
        return False

    @property
    def encoding(self) -> str | None:  # type: ignore[override]
        primary_encoding = getattr(self._primary, 'encoding', None)
        if primary_encoding:
            return primary_encoding
        if self._mirror:
            return getattr(self._mirror, 'encoding', None)
        return None


def _sanitise_scene_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name) or 'scene'


def _prepare_scene_log(job: _SceneJob) -> tuple[Optional[logging.Handler], Optional[contextlib.ExitStack], Optional[Path]]:
    """Return logging handler + stdout/stderr redirect stack pointing to a scene log."""

    config_path = job.kwargs.get('config_path')
    config_stem = Path(config_path).stem if config_path else 'adhoc'
    experiment_cfg = job.kwargs.get('experiment_cfg') or {}
    run_stamp_raw = experiment_cfg.get('run_timestamp')
    run_stamp = str(run_stamp_raw) if run_stamp_raw else now_local_stamp()

    log_dir = Path('logs') / 'scene_workers' / f'{config_stem}_{run_stamp}'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Unable to prepare log directory %s for scene %s: %s", log_dir, job.scene, exc)
        return None, None, None

    sanitised_scene = _sanitise_scene_name(job.scene)
    log_path = log_dir / f'{job.index:03d}_{sanitised_scene}_pid{os.getpid()}.log'

    handler: Optional[logging.Handler] = None
    stack: Optional[contextlib.ExitStack] = None
    try:
        handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter(ENTRY_LOG_FORMAT))
        logging.getLogger().addHandler(handler)

        stack = contextlib.ExitStack()
        log_stream = stack.enter_context(open(log_path, 'a', encoding='utf-8', buffering=1))
        stdout = _TeeStream(log_stream, sys.stdout)
        stderr = _TeeStream(log_stream, sys.stderr)
        stack.enter_context(contextlib.redirect_stdout(stdout))
        stack.enter_context(contextlib.redirect_stderr(stderr))
    except OSError as exc:
        LOGGER.warning("Unable to open log file %s for scene %s: %s", log_path, job.scene, exc)
        if stack:
            stack.close()
        if handler:
            logging.getLogger().removeHandler(handler)
            handler.close()
        return None, None, log_path

    return handler, stack, log_path


def _cleanup_scene_log(handler: Optional[logging.Handler], stack: Optional[contextlib.ExitStack]) -> None:
    if handler:
        logging.getLogger().removeHandler(handler)
        handler.close()
    if stack:
        stack.close()


def _run_scene_job(job: _SceneJob) -> Dict[str, Any]:
    level = configure_entry_log_format()
    LOGGER.setLevel(level)
    handler: Optional[logging.Handler] = None
    stack: Optional[contextlib.ExitStack] = None
    log_path: Optional[Path] = None
    try:
        handler, stack, log_path = _prepare_scene_log(job)
        if log_path:
            LOGGER.info("Scene %s PID=%d logging to %s", job.scene, os.getpid(), log_path)
        summary = run_scene_workflow(**job.kwargs)
        if log_path:
            summary.setdefault('experiment', {}).setdefault('scene_log', str(log_path))
        return summary
    finally:
        _cleanup_scene_log(handler, stack)


def _scene_job_worker(conn: mp.connection.Connection, job: _SceneJob) -> None:
    """Execute a scene workflow inside an isolated process and report back the outcome."""
    try:
        summary = _run_scene_job(job)
    except BaseException as exc:  # pragma: no cover - propagate detailed diagnostics
        try:
            conn.send(
                {
                    'status': 'error',
                    'exc_type': f'{type(exc).__module__}.{type(exc).__name__}',
                    'exc_message': str(exc),
                    'traceback': traceback.format_exc(),
                }
            )
        except OSError:
            pass
    else:
        try:
            conn.send({'status': 'ok', 'summary': summary})
        except OSError:
            pass
    finally:
        conn.close()


def _launch_scene_process(job: _SceneJob, ctx: mp.context.BaseContext) -> _SceneProcessHandle:
    """Spawn a new process to execute a scene job and return a handle for communication."""
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_scene_job_worker, args=(child_conn, job), daemon=False)
    try:
        process.start()
    except BaseException:
        parent_conn.close()
        raise
    finally:
        child_conn.close()
    return _SceneProcessHandle(job=job, process=process, conn=parent_conn)


def _finalise_scene_process(handle: _SceneProcessHandle) -> Tuple[bool, Dict[str, Any]]:
    """Collect the outcome from an isolated scene process."""
    payload: Dict[str, Any] = {}
    try:
        payload = handle.conn.recv()
    except EOFError:
        payload = {}
    finally:
        handle.conn.close()
        handle.process.join()

    if payload.get('status') == 'ok':
        return True, payload['summary']

    error_payload = dict(payload)
    error_payload['exitcode'] = handle.process.exitcode
    if not error_payload.get('exc_type'):
        error_payload.setdefault('exc_type', 'RuntimeError')
        error_payload.setdefault(
            'exc_message',
            f'Isolated scene process exited with code {handle.process.exitcode}',
        )
    return False, error_payload


def _run_scene_job_isolated(job: _SceneJob) -> Tuple[bool, Dict[str, Any]]:
    """Launch a dedicated child process for a scene job and return success flag with payload."""
    ctx = mp.get_context('spawn')
    handle = _launch_scene_process(job, ctx)
    return _finalise_scene_process(handle)


def _resolve_path_override(env_var: str, configured: Optional[str]) -> Optional[str]:
    """Return the configured path, allowing the environment to override it."""

    override = os.environ.get(env_var)
    return override if override else configured


def _prepare_memory_event_readers(
    paths: Optional[Sequence[Path | str]],
) -> Tuple[List, List[Path]]:
    readers: List = []
    missing: List[Path] = []
    if not paths or not HAS_OOM_MONITOR or MemoryEventsReader is None:
        return readers, missing
    for raw in paths:
        path = Path(raw)
        reader = MemoryEventsReader(path)
        try:
            snapshot = reader.read()
        except OSError:
            missing.append(reader.path)
            continue
        if 'oom' in snapshot or 'oom_kill' in snapshot:
            readers.append(reader)
        else:
            missing.append(reader.path)
    return readers, missing


def _read_memory_snapshots(readers: Sequence) -> Dict[Path, Dict[str, int]]:
    snapshot: Dict[Path, Dict[str, int]] = {}
    if not HAS_OOM_MONITOR:
        return snapshot
    for reader in readers:
        try:
            snapshot[reader.path] = reader.read()
        except OSError:
            snapshot[reader.path] = {}
    return snapshot


def _detect_memory_events(
    readers: Sequence,
    previous: Dict[Path, Dict[str, int]],
    current: Dict[Path, Dict[str, int]],
) -> List:
    events: List = []
    if not HAS_OOM_MONITOR:
        return events
    for reader in readers:
        events.extend(reader.detect_oom_events(previous.get(reader.path), current.get(reader.path, {})))
    return events


def _format_oom_events(events: Sequence) -> str:
    parts = []
    if not HAS_OOM_MONITOR or not events:
        return ""
    for event in events:
        parts.append(f"{event.path.name}:{event.field}+{event.delta}")
    return ", ".join(parts)


def _format_job_error(job: _SceneJob, payload: Dict[str, Any]) -> str:
    """Format error message from failed isolated subprocess (DRY helper)."""
    message = (
        f"Scene {job.scene} failed in isolated subprocess "
        f"{payload.get('exc_type')}: {payload.get('exc_message')}"
    ).strip()
    exitcode = payload.get('exitcode')
    if exitcode is not None:
        message = f"{message} (exitcode={exitcode})"
    if payload.get('traceback'):
        message = f"{message}\n{payload['traceback']}"
    return message


def execute_workflow(
    config: Dict[str, Any],
    *,
    override_output: Optional[str] = None,
    config_path: Optional[Path] = None,
    memory_event_paths: Optional[Sequence[Path | str]] = None,
) -> List[Dict[str, Any]]:
    """根據設定檔執行場景或多場景工作流程。"""
    experiment_cfg = config.get('experiment', {})
    if not isinstance(experiment_cfg, dict):
        raise WorkflowConfigError('`experiment` section must be a mapping')

    stages_cfg = config.get('stages', {})
    if not isinstance(stages_cfg, dict):
        raise WorkflowConfigError('`stages` section must be a mapping')
    default_stage_gpu = stages_cfg.get('gpu')

    memory_readers, missing_memory_paths = _prepare_memory_event_readers(memory_event_paths)
    if memory_event_paths is not None:
        if missing_memory_paths and not memory_readers:
            LOGGER.warning(
                "memory.events monitoring unavailable (%s); parallel scene execution will be disabled",
                ", ".join(str(path) for path in missing_memory_paths),
            )
        elif missing_memory_paths:
            LOGGER.warning(
                "Some memory.events files lack OOM counters: %s",
                ", ".join(str(path) for path in missing_memory_paths),
            )

    scenes_cfg = experiment_cfg.get('scenes')
    scene_start_cfg = experiment_cfg.get('scene_start')
    scene_end_cfg = experiment_cfg.get('scene_end')
    scenes_list: Optional[List[str]] = None
    dataset_root_raw = _resolve_path_override('MY3DIS_DATASET_ROOT', experiment_cfg.get('dataset_root'))
    if scenes_cfg is not None or scene_start_cfg is not None or scene_end_cfg is not None:
        if not dataset_root_raw:
            raise WorkflowConfigError('experiment.dataset_root is required when selecting scenes')
        dataset_root = Path(dataset_root_raw).expanduser().absolute()
        scenes_list = normalize_scene_list(
            scenes_cfg,
            dataset_root,
            scene_start=scene_start_cfg,
            scene_end=scene_end_cfg,
        )
    else:
        dataset_root = Path(dataset_root_raw).expanduser().absolute() if dataset_root_raw else None

    if scenes_list:
        output_root_base_raw = override_output or _resolve_path_override(
            'MY3DIS_OUTPUT_ROOT',
            experiment_cfg.get('output_root'),
        )
        output_root_base_resolved = expand_output_path_template(output_root_base_raw, experiment_cfg)
        output_root_base = Path(output_root_base_resolved).expanduser().absolute()

        aggregate_output = bool(experiment_cfg.get('aggregate_output'))
        run_timestamp = experiment_cfg.get('run_timestamp')
        if aggregate_output:
            experiment_stamp = str(run_timestamp) if run_timestamp else now_local_stamp()
            experiment_root = output_root_base / experiment_stamp
        else:
            experiment_root = output_root_base
            experiment_stamp = str(run_timestamp) if run_timestamp else None

        experiment_root.mkdir(parents=True, exist_ok=True)

        parent_meta = {
            'name': experiment_cfg.get('name'),
            'scenes': scenes_list,
            'experiment_root': str(experiment_root),
            'timestamp': experiment_stamp,
            'aggregate_output': aggregate_output,
            'scene_start': scene_start_cfg,
            'scene_end': scene_end_cfg,
        }

        parallel_cfg = experiment_cfg.get('parallel_scenes')
        try:
            parallel_scenes = int(parallel_cfg) if parallel_cfg is not None else 1
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid experiment.parallel_scenes=%r, falling back to sequential execution",
                parallel_cfg,
            )
            parallel_scenes = 1
        parallel_scenes = max(1, parallel_scenes)

        # Check force_parallel flag to allow parallel execution without OOM monitoring
        force_parallel = bool(experiment_cfg.get('force_parallel', False))

        if parallel_scenes > 1 and memory_event_paths is not None and not memory_readers:
            if force_parallel:
                LOGGER.warning(
                    "Parallel execution enabled without OOM monitoring (force_parallel=true). "
                    "Note: Out-of-memory events will not be detected."
                )
            else:
                LOGGER.warning(
                    "Parallel execution requested but OOM monitoring is unavailable; forcing sequential scene scheduling. "
                    "Set experiment.force_parallel=true to override this safety check."
                )
                parallel_scenes = 1

        jobs: List[_SceneJob] = []
        for index, scene_name in enumerate(scenes_list):
            if dataset_root is None:
                raise WorkflowConfigError('dataset_root must be provided when scenes are enumerated')
            scene_data_path = dataset_root / scene_name / 'outputs' / 'color'
            if not scene_data_path.exists():
                raise WorkflowConfigError(f'data path not found for scene {scene_name}: {scene_data_path}')
            if aggregate_output:
                scene_output_root = experiment_root / scene_name
            else:
                scene_output_root = output_root_base / scene_name
            scene_output_root.mkdir(parents=True, exist_ok=True)

            scene_experiment_cfg = dict(experiment_cfg)
            scene_experiment_cfg['name'] = experiment_cfg.get('name')
            scene_experiment_cfg['data_path'] = str(scene_data_path)
            scene_experiment_cfg['output_root'] = str(scene_output_root)
            scene_experiment_cfg['aggregate_output'] = aggregate_output
            scene_experiment_cfg['run_timestamp'] = experiment_stamp
            scene_experiment_cfg.pop('scenes', None)

            parent_meta_scene = dict(parent_meta)
            parent_meta_scene['index'] = index
            parent_meta_scene['scene'] = scene_name

            job_kwargs = {
                'config': config,
                'experiment_cfg': scene_experiment_cfg,
                'stages_cfg': stages_cfg,
                'default_stage_gpu': default_stage_gpu,
                'data_path': str(scene_data_path),
                'output_root': str(scene_output_root),
                'config_path': config_path,
                'parent_meta': parent_meta_scene,
            }
            jobs.append(_SceneJob(index=index, scene=scene_name, kwargs=job_kwargs))

        summaries_ordered: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
        scene_errors: List[Tuple[_SceneJob, BaseException]] = []

        isolate_cfg = experiment_cfg.get('isolate_scenes')
        if isolate_cfg is None:
            isolate_scenes = True
        else:
            isolate_scenes = bool(isolate_cfg)
            if not isolate_scenes:
                LOGGER.warning(
                    "experiment.isolate_scenes=%r disables per-scene isolation; memory usage may accumulate across scenes",
                    isolate_cfg,
                )

        if parallel_scenes > 1 and len(jobs) > 1:
            LOGGER.info(
                "Executing %d scenes with up to %d parallel workers (isolated per scene)",
                len(jobs),
                parallel_scenes,
            )
            ctx = mp.get_context('spawn')
            pending_jobs = deque(jobs)
            active_handles: List[_SceneProcessHandle] = []
            executor_backoff = False
            previous_snapshot = _read_memory_snapshots(memory_readers) if memory_readers else {}

            def fill_pool() -> None:
                while (
                    not executor_backoff
                    and pending_jobs
                    and len(active_handles) < parallel_scenes
                ):
                    next_job = pending_jobs.popleft()
                    try:
                        handle = _launch_scene_process(next_job, ctx)
                    except BaseException as exc:  # pragma: no cover - propagate diagnostics
                        LOGGER.error(
                            "Failed to start isolated process for scene %s",
                            next_job.scene,
                            exc_info=True,
                        )
                        scene_errors.append((next_job, exc))
                        continue
                    active_handles.append(handle)

            fill_pool()

            while active_handles:
                ready_conns = mp.connection.wait([handle.conn for handle in active_handles])
                for conn in ready_conns:
                    idx = next(
                        (i for i, handle in enumerate(active_handles) if handle.conn is conn),
                        None,
                    )
                    if idx is None:
                        continue
                    handle = active_handles.pop(idx)
                    job = handle.job
                    before_snapshot = previous_snapshot if memory_readers else {}
                    try:
                        ok, payload = _finalise_scene_process(handle)
                    except BaseException as exc:  # pragma: no cover - propagate diagnostics
                        LOGGER.error("Scene %s failed during execution", job.scene, exc_info=True)
                        scene_errors.append((job, exc))
                        continue

                    if ok:
                        summaries_ordered[job.index] = payload
                    else:
                        error_msg = _format_job_error(job, payload)
                        scene_errors.append((job, WorkflowRuntimeError(error_msg)))

                    if memory_readers:
                        current_snapshot = _read_memory_snapshots(memory_readers)
                        events = _detect_memory_events(memory_readers, previous_snapshot, current_snapshot)
                        previous_snapshot = current_snapshot
                        if events and not executor_backoff:
                            executor_backoff = True
                            LOGGER.warning(
                                "OOM counters increased while running in parallel (%s); "
                                "continuing remaining %d scene(s) sequentially",
                                _format_oom_events(events),
                                len(pending_jobs),
                            )

                fill_pool()

            if executor_backoff and pending_jobs:
                LOGGER.info(
                    "Falling back to sequential execution for remaining %d scene(s)",
                    len(pending_jobs),
                )
                while pending_jobs:
                    job = pending_jobs.popleft()
                    try:
                        before_snapshot = previous_snapshot if memory_readers else {}
                        if isolate_scenes:
                            ok, payload = _run_scene_job_isolated(job)
                            if ok:
                                summaries_ordered[job.index] = payload
                            else:
                                raise WorkflowRuntimeError(_format_job_error(job, payload))
                        else:
                            summaries_ordered[job.index] = _run_scene_job(job)
                        if memory_readers:
                            current_snapshot = _read_memory_snapshots(memory_readers)
                            events = _detect_memory_events(memory_readers, before_snapshot, current_snapshot)
                            previous_snapshot = current_snapshot
                            if events:
                                LOGGER.warning(
                                    "OOM counters increased during scene %s (%s)",
                                    job.scene,
                                    _format_oom_events(events),
                                )
                    except BaseException as exc:
                        LOGGER.error("Scene %s failed during execution", job.scene, exc_info=True)
                        raise
        else:
            previous_snapshot = _read_memory_snapshots(memory_readers) if memory_readers else {}
            for job in jobs:
                try:
                    before_snapshot = previous_snapshot if memory_readers else {}
                    if isolate_scenes:
                        ok, payload = _run_scene_job_isolated(job)
                        if ok:
                            summaries_ordered[job.index] = payload
                        else:
                            raise WorkflowRuntimeError(_format_job_error(job, payload))
                    else:
                        summaries_ordered[job.index] = _run_scene_job(job)
                    if memory_readers:
                        current_snapshot = _read_memory_snapshots(memory_readers)
                        events = _detect_memory_events(memory_readers, before_snapshot, current_snapshot)
                        previous_snapshot = current_snapshot
                        if events:
                            LOGGER.warning(
                                "OOM counters increased during scene %s (%s)",
                                job.scene,
                                _format_oom_events(events),
                            )
                except BaseException as exc:
                    LOGGER.error("Scene %s failed during execution", job.scene, exc_info=True)
                    raise

        if scene_errors:
            failed_names = ', '.join(job.scene for job, _ in scene_errors)
            primary_exc = scene_errors[0][1]
            raise WorkflowRuntimeError(
                f'One or more scene workflows failed: {failed_names}'
            ) from primary_exc

        summaries: List[Dict[str, Any]] = [summary for summary in summaries_ordered if summary is not None]
        experiment_scene_records: List[Dict[str, Any]] = []

        for job in jobs:
            summary = summaries_ordered[job.index]
            if summary is None:
                continue

            summary_json_path = summary.get('summary_json')
            summary_json_rel = None
            if summary_json_path:
                try:
                    summary_json_rel = str(Path(summary_json_path).relative_to(experiment_root))
                except ValueError:
                    summary_json_rel = summary_json_path

            run_dir_path = summary.get('run_dir')
            run_dir_rel = None
            if run_dir_path:
                try:
                    run_dir_rel = str(Path(run_dir_path).relative_to(experiment_root))
                except ValueError:
                    run_dir_rel = run_dir_path

            scene_record = {
                'scene': job.scene,
                'run_dir': run_dir_rel,
                'summary_json': summary_json_rel,
                'levels': summary.get('scene_level_summary', {}).get('levels')
                if summary.get('scene_level_summary')
                else None,
            }
            experiment_scene_records.append(scene_record)

        if aggregate_output:
            experiment_summary = {
                'generated_at': now_local_iso(),
                'experiment': {
                    'name': experiment_cfg.get('name'),
                    'timestamp': experiment_stamp,
                    'scenes': scenes_list,
                    'output_root': str(experiment_root),
                },
                'runs': experiment_scene_records,
            }
            experiment_summary_path = experiment_root / 'experiment_summary.json'
            try:
                with experiment_summary_path.open('w') as handle:
                    json.dump(experiment_summary, handle, indent=2)
            except OSError as exc:
                LOGGER.warning(
                    "Failed to write experiment summary to %s: %s",
                    experiment_summary_path,
                    exc,
                    exc_info=True,
                )

        return summaries

    data_path_raw = experiment_cfg.get('data_path')
    if not data_path_raw:
        raise WorkflowConfigError('experiment.data_path is required')
    output_root_raw = expand_output_path_template(override_output or experiment_cfg.get('output_root'), experiment_cfg)

    summary = run_scene_workflow(
        config=config,
        experiment_cfg=experiment_cfg,
        stages_cfg=stages_cfg,
        default_stage_gpu=default_stage_gpu,
        data_path=data_path_raw,
        output_root=output_root_raw,
        config_path=config_path,
        parent_meta=None,
    )
    return [summary]


__all__ = ['execute_workflow']
