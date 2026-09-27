"""Offline contracts for the fixed per-run training-attempt budget."""

import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import experiment_artifacts as artifacts
import run_training


class RunTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="run-training-")
        self.addCleanup(temporary.cleanup)
        self.worktree = Path(temporary.name).resolve() / "worktree"
        self.worktree.mkdir()
        self.source = self.worktree / "train.py"
        self.source.write_text(self._training_source(), encoding="utf-8")
        self.run = self._run_directory("fixture-run")
        self.prepared_id = "a" * 64

    def _run_directory(self, name):
        run = self.worktree / "results" / name
        run.mkdir(parents=True)
        self._write_json(run / "run.json", {"run_tag": name})
        return run

    def _write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def _read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _call(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = run_training.main([str(value) for value in args])
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def _init(self, limit=2, *extra, run=None):
        result = self._call("init", "--run-dir", run or self.run,
                            "--max-train-calls", limit, *extra)
        self.assertEqual(result[0], 0, result)
        return self._read_json((run or self.run) / "train_budget.json")

    def _training_source(self, behavior="success"):
        if behavior == "waiting":
            return (
                "import os, time\nfrom pathlib import Path\n"
                "Path(os.environ['AUTORESEARCH_ARTIFACT_DIR'], 'ready').touch()\n"
                "time.sleep(30)\n"
            )
        if behavior == "failed":
            return "import sys\nprint('fixture training failed', flush=True)\nsys.exit(7)\n"
        if behavior == "missing_outputs":
            return "print('fixture exited without metrics', flush=True)\n"
        return (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "directory = Path(os.environ['AUTORESEARCH_ARTIFACT_DIR'])\n"
            "receipt = json.loads((directory / 'source.json').read_text())\n"
            "metrics = {'score': 0.8, 'maximize': True, "
            f"'metric_version': {artifacts.METRIC_VERSION!r}, "
            "'prepared_id': receipt['prepared_id']}\n"
            "config = {'seed': 1337, 'cwd': str(Path.cwd()), 'python': sys.executable, "
            "'artifact_dir': str(directory)}\n"
            "(directory / 'metrics.json').write_text(json.dumps(metrics))\n"
            "(directory / 'config.json').write_text(json.dumps(config))\n"
            "(directory / 'validation_predictions.npy').write_bytes(b'fixture-predictions')\n"
            "print('fixture training completed', flush=True)\n"
            "print('fixture stderr', file=sys.stderr, flush=True)\n"
        )

    def _snapshot(self, trial_id="trial0001", behavior="success", *, run=None,
                  receipt_run_id=None, source=None):
        run = run or self.run
        source = source or self.source
        source.write_text(self._training_source(behavior), encoding="utf-8")
        trial = run / "trials" / trial_id
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = artifacts.main(["snapshot", "--source", str(source), "--artifact-dir", str(trial),
                "--experiment-id", trial_id, "--run-id", receipt_run_id or run.name,
                "--prepared-id", self.prepared_id])
        self.assertEqual(code, 0)
        return trial

    def _run_trial(self, trial):
        return self._call("run", "--run-dir", trial.parent.parent, "--trial-id", trial.name)

    def _status(self, used, remaining):
        result = self._call("status", "--run-dir", self.run)
        self.assertEqual(result[0], 0, result)
        status = json.loads(result[1])
        self.assertEqual(status["used"], used)
        self.assertEqual(status["remaining"], remaining)
        self.assertEqual(status["max_train_calls"], used + remaining)
        self.assertIn("states", status)
        return status

    def _assert_not_started(self, trial):
        self.assertFalse((trial / "execution.json").exists())
        self.assertFalse((trial / "run.log").exists())
        self.assertEqual(self._read_json(trial / "source.json")["execution_status"], "pending")

    def _assert_execution(self, trial, attempt_no, status, returncode):
        execution = self._read_json(trial / "execution.json")
        self.assertEqual(execution["attempt_no"], attempt_no)
        self.assertEqual(execution["run_id"], self.run.name)
        self.assertEqual(execution["trial_id"], trial.name)
        self.assertEqual(execution["source_sha256"], hashlib.sha256((trial / "source.py").read_bytes()).hexdigest())
        self.assertEqual(execution["status"], status)
        self.assertEqual(execution["returncode"], returncode)
        started = datetime.fromisoformat(execution["started_at_utc"])
        finished = datetime.fromisoformat(execution["finished_at_utc"])
        self.assertIsNotNone(started.utcoffset())
        self.assertIsNotNone(finished.utcoffset())
        self.assertLessEqual(started, finished)

    def test_init_binds_run_worktree_and_interpreter_without_overwriting(self):
        before = {path: path.read_bytes() for path in (self.source, self.run / "run.json")}
        budget = self._init(2)
        self.assertEqual(budget["run_id"], self.run.name)
        self.assertEqual(Path(budget["worktree"]), self.worktree)
        self.assertTrue(Path(budget["python"]).is_absolute())
        self.assertEqual(Path(budget["python"]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(budget["max_train_calls"], 2)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        saved_budget = (self.run / "train_budget.json").read_bytes()
        for limit in (2, 9):
            with self.subTest(limit=limit):
                result = self._call("init", "--run-dir", self.run, "--max-train-calls", limit)
                self.assertEqual(result[0], 2, result)
                self.assertEqual((self.run / "train_budget.json").read_bytes(), saved_budget)
        self._status(0, 2)

    def test_init_accepts_run_id_metadata_alias(self):
        self._write_json(self.run / "run.json", {"run_id": self.run.name})
        self._init(1)
        self._status(0, 1)

    def test_init_rejects_missing_or_mismatched_run_and_invalid_limit(self):
        missing = self.worktree / "results" / "missing"
        result = self._call("init", "--run-dir", missing, "--max-train-calls", 1)
        self.assertEqual(result[0], 2, result)
        self.assertFalse(missing.exists())
        self._write_json(self.run / "run.json", {"run_tag": "other-run"})
        self.assertEqual(self._call("init", "--run-dir", self.run, "--max-train-calls", 1)[0], 2)
        self.assertFalse((self.run / "train_budget.json").exists())
        self._write_json(self.run / "run.json", {"run_tag": self.run.name})
        for limit in (0, -1, "not-an-integer"):
            with self.subTest(limit=limit):
                self.assertEqual(self._call("init", "--run-dir", self.run, "--max-train-calls", limit)[0], 2)
                self.assertFalse((self.run / "train_budget.json").exists())

    def test_init_does_not_adopt_preexisting_training(self):
        for marker in ("run.log", "execution.json", "completed"):
            with self.subTest(marker=marker):
                run = self._run_directory("old-" + marker.replace(".", "-"))
                trial = self._snapshot(run=run)
                if marker == "completed":
                    self._write_json(trial / "metrics.json", {"score": 0.8, "maximize": True,
                        "metric_version": artifacts.METRIC_VERSION, "prepared_id": self.prepared_id})
                    self._write_json(trial / "config.json", {"seed": 1337})
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(artifacts.main(["complete", "--artifact-dir", str(trial)]), 0)
                else:
                    (trial / marker).write_text("existing training evidence\n", encoding="utf-8")
                before = {path: path.read_bytes() for path in trial.iterdir()}
                result = self._call("init", "--run-dir", run, "--max-train-calls", 2)
                self.assertEqual(result[0], 2, result)
                self.assertFalse((run / "train_budget.json").exists())
                self.assertEqual({path: path.read_bytes() for path in trial.iterdir()}, before)

    def test_failure_consumes_one_call_and_last_allowed_success_can_complete(self):
        self._init(2)
        failed = self._snapshot(behavior="failed")
        result = self._run_trial(failed)
        self.assertEqual(result[0], 1, result)
        self._assert_execution(failed, 1, "failed", 7)
        self.assertEqual(self._read_json(failed / "source.json")["execution_status"], "pending")
        self._status(1, 1)

        completed = self._snapshot("trial0002")
        result = self._run_trial(completed)
        self.assertEqual(result[0], 0, result)
        self._assert_execution(completed, 2, "completed", 0)
        receipt = self._read_json(completed / "source.json")
        self.assertEqual(receipt["execution_status"], "completed")
        for filename, field in (("metrics.json", "metrics_sha256"), ("config.json", "config_sha256"),
                                ("run.log", "log_sha256")):
            self.assertEqual(receipt[field], hashlib.sha256((completed / filename).read_bytes()).hexdigest())
        config = self._read_json(completed / "config.json")
        self.assertEqual(Path(config["cwd"]), self.worktree)
        self.assertEqual(Path(config["artifact_dir"]), completed)
        self.assertEqual(Path(config["python"]).resolve(), Path(sys.executable).resolve())
        log = (completed / "run.log").read_text(encoding="utf-8")
        self.assertIn("fixture training completed", log)
        self.assertIn("fixture stderr", log)
        self._status(2, 0)

        blocked = self._snapshot("trial0003")
        result = self._run_trial(blocked)
        self.assertEqual(result[0], 3, result)
        self._assert_not_started(blocked)
        self._status(2, 0)

    def test_used_budget_cannot_be_reset_by_initializing_again(self):
        self._init(2)
        trial = self._snapshot()
        self.assertEqual(self._run_trial(trial)[0], 0)
        before = (self.run / "train_budget.json").read_bytes()
        result = self._call("init", "--run-dir", self.run, "--max-train-calls", 100)
        self.assertEqual(result[0], 2, result)
        self.assertEqual((self.run / "train_budget.json").read_bytes(), before)
        self._status(1, 1)

    def test_same_trial_cannot_restart_or_overwrite_training_evidence(self):
        self._init(3)
        for index, behavior in enumerate(("success", "failed"), 1):
            with self.subTest(behavior=behavior):
                trial = self._snapshot("trial000" + str(index), behavior)
                self.assertEqual(self._run_trial(trial)[0], 0 if behavior == "success" else 1)
                paths = [trial / name for name in ("run.log", "execution.json", "source.json")]
                before = {path: path.read_bytes() for path in paths}
                result = self._run_trial(trial)
                self.assertEqual(result[0], 2, result)
                self.assertEqual({path: path.read_bytes() for path in paths}, before)
                self._status(index, 3 - index)

    def test_changed_original_or_snapshot_source_is_rejected_without_counting(self):
        self._init(1)
        trial = self._snapshot()
        for path in (self.source, trial / "source.py"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_text("raise RuntimeError('modified source must not execute')\n", encoding="utf-8")
                result = self._run_trial(trial)
                self.assertEqual(result[0], 2, result)
                self._assert_not_started(trial)
                self._status(0, 1)
                path.write_bytes(original)

    def test_cross_run_receipt_or_foreign_source_path_is_rejected_without_counting(self):
        self._init(2)
        wrong_run = self._snapshot(receipt_run_id="other-run")
        outside = self.worktree.parent / "other-train.py"
        wrong_source = self._snapshot("trial0002", source=outside)
        for trial in (wrong_run, wrong_source):
            with self.subTest(trial=trial.name):
                result = self._run_trial(trial)
                self.assertEqual(result[0], 2, result)
                self._assert_not_started(trial)
                self._status(0, 2)

    def test_run_requires_initialized_budget(self):
        trial = self._snapshot()
        result = self._run_trial(trial)
        self.assertEqual(result[0], 2, result)
        self.assertFalse((self.run / "train_budget.json").exists())
        self._assert_not_started(trial)

    def test_exit_zero_without_metrics_counts_but_does_not_complete(self):
        self._init(1)
        trial = self._snapshot(behavior="missing_outputs")
        result = self._run_trial(trial)
        self.assertEqual(result[0], 1, result)
        self._assert_execution(trial, 1, "failed", 0)
        self.assertEqual(self._read_json(trial / "source.json")["execution_status"], "pending")
        self.assertFalse((trial / "metrics.json").exists())
        self._status(1, 0)

    def test_status_and_count_persist_across_fresh_processes(self):
        self._init(3)
        first = self._snapshot()
        self.assertEqual(self._run_trial(first)[0], 0)
        second = self._snapshot("trial0002", "failed")
        script = Path(run_training.__file__).resolve()
        result = subprocess.run([sys.executable, "-S", str(script), "run", "--run-dir", str(self.run),
                                 "--trial-id", second.name], capture_output=True, text=True,
                                encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 1, result.stderr)
        self._assert_execution(second, 2, "failed", 7)
        result = subprocess.run([sys.executable, "-S", str(script), "status", "--run-dir", str(self.run)],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertEqual(status["used"], 2)
        self.assertEqual(status["remaining"], 1)
        self.assertEqual(status["max_train_calls"], 3)
        self.assertIn("states", status)
        self._status(2, 1)

    def test_expired_deadline_prevents_training_without_counting(self):
        expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        budget = self._init(1, "--deadline-utc", expired)
        self.assertEqual(datetime.fromisoformat(budget["deadline_utc"]), datetime.fromisoformat(expired))
        trial = self._snapshot()
        result = self._run_trial(trial)
        self.assertEqual(result[0], 2, result)
        self._assert_not_started(trial)
        self._status(0, 1)

    def test_deadline_requires_an_explicit_timezone(self):
        result = self._call("init", "--run-dir", self.run, "--max-train-calls", 1,
                            "--deadline-utc", "2099-01-01T00:00:00")
        self.assertEqual(result[0], 2, result)
        self.assertFalse((self.run / "train_budget.json").exists())

    def _waiting_launcher(self, trial):
        process = subprocess.Popen([sys.executable, str(Path(run_training.__file__).resolve()),
            "run", "--run-dir", str(self.run), "--trial-id", trial.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)
            record = trial / "execution.json"
            if record.exists():
                pid = self._read_json(record).get("pid")
                if pid is not None:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        self.addCleanup(cleanup)
        until = time.monotonic() + 5
        while time.monotonic() < until:
            if (trial / "ready").exists():
                return process
            if process.poll() is not None:
                self.fail(f"Launcher exited before training: {process.communicate()}")
            time.sleep(0.02)
        self.fail("Training did not start in time")

    def test_concurrent_launch_rejected_and_sigterm_stops_training(self):
        self._init(2)
        first = self._snapshot(behavior="waiting")
        second = self._snapshot("trial0002", "waiting")
        process = self._waiting_launcher(first)
        pid = self._read_json(first / "execution.json")["pid"]
        result = self._run_trial(second)
        self.assertEqual(result[0], 2, result)
        self.assertIn("active", result[2])
        self._assert_not_started(second)
        self._status(1, 1)
        process.terminate()
        output = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 1, output)
        self._assert_execution(first, 1, "interrupted", -signal.SIGTERM)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self._status(1, 1)

    def test_killed_launcher_keeps_budget_reserved_and_child_holds_lock(self):
        self._init(2)
        first = self._snapshot(behavior="waiting")
        second = self._snapshot("trial0002", "waiting")
        process = self._waiting_launcher(first)
        process.kill()
        process.communicate(timeout=10)
        result = self._run_trial(second)
        self.assertEqual(result[0], 2, result)
        self.assertIn("active", result[2])
        self._assert_not_started(second)
        self._status(1, 1)

    def test_signal_after_finalization_preserves_completed_status(self):
        self._init(1)
        trial = self._snapshot()
        complete = artifacts.complete_command

        def complete_then_interrupt(args):
            complete(args)
            signal.raise_signal(signal.SIGTERM)

        with mock.patch.object(artifacts, "complete_command", side_effect=complete_then_interrupt):
            result = self._run_trial(trial)
        self.assertEqual(result[0], 0, result)
        self._assert_execution(trial, 1, "completed", 0)
        self.assertEqual(self._read_json(trial / "source.json")["execution_status"], "completed")

    def test_signal_during_state_update_still_records_interruption(self):
        self._init(1)
        trial = self._snapshot(behavior="waiting")
        write = artifacts._write_json
        interrupted = False

        def write_then_interrupt(path, record):
            nonlocal interrupted
            write(path, record)
            if Path(path).name == "execution.json.tmp" and not interrupted:
                interrupted = True
                signal.raise_signal(signal.SIGTERM)

        with mock.patch.object(artifacts, "_write_json", side_effect=write_then_interrupt):
            result = self._run_trial(trial)
        self.assertEqual(result[0], 1, result)
        self._assert_execution(trial, 1, "interrupted", -signal.SIGTERM)
        self.assertFalse((trial / "execution.json.tmp").exists())
        self._status(1, 0)


if __name__ == "__main__":
    unittest.main()
