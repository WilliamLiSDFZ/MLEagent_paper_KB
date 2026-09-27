"""Run a snapshotted train.py with a persistent per-run attempt limit (Linux/macOS).

Initialize once, then use the existing snapshot command (and write adoption.json
if needed) before each run. Successful training is finalized with the existing
artifact helper. The limit counts reserved starts, including failures; epochs
and fits inside one train.py are not counted separately. This is a protocol
helper, not a sandbox preventing direct execution of train.py.

    python run_training.py init --run-dir "$RUN_DIR" --max-train-calls 10
    python run_training.py run --run-dir "$RUN_DIR" --trial-id trial0001
    python run_training.py status --run-dir "$RUN_DIR"

Initialize with the shared environment's Python; every run uses that interpreter.
An optional --deadline-utc at
init stops training at a known deadline; otherwise Kubernetes controls lifetime.
Exit codes: 0 success, 1 training/finalization failure, 2 invalid input or busy,
3 budget exhausted. This helper never edits best.json or stops the Pod.
"""

import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import experiment_artifacts as artifacts


def _now():
    return datetime.now(timezone.utc)


def _deadline(value):
    if value is None:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("--deadline-utc requires a timezone")
    return result.astimezone(timezone.utc)


@contextmanager
def _lock(root):
    with (root / ".train.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another training command is active for this run") from None
        yield stream


def _layout(root):
    if not root.is_dir() or root.parent.name != "results":
        raise ValueError("--run-dir must be an existing worktree/results/<run_id> directory")
    metadata = artifacts._read_json(root / "run.json")
    if (metadata.get("run_tag") or metadata.get("run_id")) != root.name:
        raise ValueError("run.json identity differs from the run directory")
    return root.parent.parent


def _budget(root):
    worktree = _layout(root)
    budget = artifacts._read_json(root / "train_budget.json")
    if budget.get("run_id") != root.name or budget.get("worktree") != str(worktree):
        raise ValueError("Training budget belongs to another run/worktree")
    if type(budget.get("max_train_calls")) is not int or budget["max_train_calls"] < 1:
        raise ValueError("Invalid training budget")
    return budget


def _usage(root, budget):
    records = [artifacts._read_json(p) for p in sorted(root.glob("trials/*/execution.json"))]
    return {"max_train_calls": budget["max_train_calls"], "used": len(records),
            "remaining": max(0, budget["max_train_calls"] - len(records)),
            "states": dict(Counter(r["status"] for r in records))}


def _init(args, root):
    worktree = _layout(root)
    if args.max_train_calls < 1:
        raise ValueError("--max-train-calls must be positive")
    deadline = _deadline(args.deadline_utc)
    artifacts._bytes(worktree / "train.py")
    with _lock(root):
        for trial in root.glob("trials/*"):
            if any((trial / name).exists() for name in ("execution.json", "run.log", "metrics.json")):
                raise ValueError("Initialize the budget before any training; existing attempts cannot be reset")
            if (trial / "source.json").exists() and artifacts._read_json(trial / "source.json").get("execution_status") != "pending":
                raise ValueError("Initialize the budget before any completed trials")
        artifacts._write_json(root / "train_budget.json", {
            "run_id": root.name, "worktree": str(worktree),
            # Do not resolve the venv symlink: its bin/python selects the environment.
            "python": str(Path(sys.executable).absolute()), "max_train_calls": args.max_train_calls,
            "deadline_utc": deadline.isoformat() if deadline else None,
        })
    print(json.dumps({"status": "initialized", **_usage(root, _budget(root))}))
    return 0


def _update(path, record):
    temporary = path.with_suffix(".json.tmp")
    if temporary.exists():
        raise ValueError(f"Unfinished state update: {temporary}")
    try:
        artifacts._write_json(temporary, record)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _Interrupted(Exception):
    pass


def _interrupt(number, frame):
    raise _Interrupted(f"signal {number}")


def _terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Also stop remaining workers if the training parent already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run(args, root):
    with _lock(root) as lock:
        budget = _budget(root)
        usage = _usage(root, budget)
        if usage["remaining"] == 0:
            print(json.dumps({"status": "budget_exhausted", **usage}))
            return 3
        deadline = _deadline(budget["deadline_utc"])
        if deadline is not None and _now() >= deadline:
            raise ValueError("Training deadline has passed")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.trial_id):
            raise ValueError("Invalid trial ID")
        trial = root / "trials" / args.trial_id
        if trial.resolve() != trial or (root / "trials").is_symlink():
            raise ValueError("Trial directory must belong to this run without symlinks")
        receipt = artifacts._read_json(trial / "source.json")
        source = Path(budget["worktree"]) / "train.py"
        if (receipt.get("protocol") != artifacts.PROTOCOL or receipt.get("execution_status") != "pending"
                or receipt.get("run_id") != root.name or receipt.get("experiment_id") != args.trial_id
                or receipt.get("source_path") != str(source)):
            raise ValueError("Expected a pending snapshot of this worktree's train.py for this trial")
        artifacts._identity(argparse.Namespace(run_id=root.name, prepared_id=None), receipt)
        if any(artifacts._hash(artifacts._bytes(p)) != receipt.get("sha256") for p in (source, trial / "source.py")):
            raise ValueError("Source changed after snapshot")
        if any((trial / name).exists() or (trial / name).is_symlink()
               for name in ("execution.json", "run.log", "metrics.json", "validation_predictions.npy")):
            raise ValueError("Trial already started or contains results; create a new trial")
        execution = {"attempt_no": usage["used"] + 1, "run_id": root.name, "trial_id": args.trial_id,
                     "source_sha256": receipt["sha256"], "status": "starting", "returncode": None,
                     "started_at_utc": _now().isoformat(), "finished_at_utc": None}
        record_path = trial / "execution.json"
        # Reserve before spawning. Even an interrupted or failed launch consumes one slot.
        artifacts._write_json(record_path, execution)
        process = None
        handlers = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            env = dict(os.environ, AUTORESEARCH_ARTIFACT_DIR=str(trial))
            with (trial / "run.log").open("xb") as log:
                process = subprocess.Popen([budget["python"], str(source)], cwd=budget["worktree"], env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    # Keep the lock alive in the child even if the launcher is SIGKILLed.
                    pass_fds=(lock.fileno(),))
                execution.update(status="running", pid=process.pid)
                _update(record_path, execution)
                remaining = max(0, (deadline - _now()).total_seconds()) if deadline else None
                execution["returncode"] = process.wait(timeout=remaining)
            if execution["returncode"] != 0:
                raise ValueError(f"Training exited with code {execution['returncode']}")
            if deadline is not None and _now() >= deadline:
                raise _Interrupted("Training deadline reached; results were not finalized")
            artifacts._bytes(trial / "validation_predictions.npy")
            with redirect_stdout(io.StringIO()):
                artifacts.complete_command(argparse.Namespace(
                    artifact_dir=trial, run_id=root.name, prepared_id=receipt["prepared_id"], log_file=None))
            execution["status"] = "completed"
        except (_Interrupted, KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
            # The source receipt is authoritative if finalization already committed.
            if (process is not None and process.returncode == 0
                    and artifacts._read_json(trial / "source.json").get("execution_status") == "completed"):
                execution["status"] = "completed"
            else:
                execution.update(status="interrupted", reason=str(exc))
        except (OSError, ValueError, UnicodeError) as exc:
            execution.update(status="failed", reason=str(exc))
        finally:
            try:
                if process is not None and execution["status"] != "completed":
                    _terminate(process)
                    execution["returncode"] = process.returncode
                execution["finished_at_utc"] = _now().isoformat()
                _update(record_path, execution)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        print(json.dumps({**execution, **_usage(root, budget)}))
        return 0 if execution["status"] == "completed" else 1


# 初始化预算、启动一次候选训练，或只读查询剩余次数。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a fixed budget before training")
    init.add_argument("--max-train-calls", type=int, required=True)
    init.add_argument("--deadline-utc", help="Optional absolute ISO timestamp with timezone")
    run = commands.add_parser("run", help="Run and finalize one existing source snapshot")
    run.add_argument("--trial-id", required=True)
    status = commands.add_parser("status", help="Read persistent attempt counts")
    for command in (init, run, status):
        command.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_dir.expanduser().resolve()
    try:
        if args.command == "init":
            return _init(args, root)
        if args.command == "run":
            _layout(root)
            return _run(args, root)
        print(json.dumps(_usage(root, _budget(root))))
        return 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
