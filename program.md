# autoresearch

This is an experiment to have the LLM do its own research on Jigsaw Unintended Bias in Toxicity Classification.

## Setup

To set up a new experiment, verify the user's existing setup. At the start of this prompt, record the actual UTC time as `agent_started_at_utc`. If the prompt gives a research duration (for example, finish within six hours), derive its absolute deadline once from that start, unless the user specifies another start point. An explicit user deadline takes precedence. Never restart this clock at `init`, after retrieval, or on reconnection:

1. **Create this arm's worktree before any edits**: `EXPERIMENT_ARM` must be `baseline`. Claude starts in the shared main repository at `$AUTORESEARCH_SOURCE_DIR` (`/workspace/autoresearch`). The Job includes `POD_UID` in `RUN_TAG`, so every new Pod gets a fresh `$AUTORESEARCH_REPO_DIR` at `$AUTORESEARCH_SOURCE_DIR/worktrees/$RUN_TAG`; never replace these values with a previous run's paths. Read `BASE_COMMIT` from the Job's `/tmp/autoresearch-start-commit` and use read-only Git checks to confirm that the main checkout is clean and its HEAD still matches that recorded commit. Both arms must use the same starting SHA and upstream `train.py`. Stop on a mismatch. Create a new branch and worktree using the self-contained Bash subprocess below; if the user specified a new branch name, use that exact name instead of the default `BRANCH_NAME`. Both the branch and worktree path must be new: never force, overwrite, delete, or reuse an existing one. Only the worktree's parent directory may be created before `git worktree add`. Proceed to later setup steps only if this entire subprocess exits successfully; on failure, do not enter an existing worktree or initialize any results. Do not rely on the calling tool's `set -e` behavior or remove failure guards. A transient Git lock error may be retried a few times; never delete a lock file. This initial `worktree add -b` is the only permitted Git mutation: do not commit, push, reset, stage files, switch branches, prune, or remove worktrees. Do not edit the main checkout or the other arm's worktree. `RUN_TAG` is the separate Pod run identifier, not the branch name.

   ```bash
   bash -euo pipefail -c '
   : "${POD_UID:?}" "${RUN_TAG:?}" "${AUTORESEARCH_SOURCE_DIR:?}" "${AUTORESEARCH_REPO_DIR:?}"
   test "$EXPERIMENT_ARM" = "baseline" || exit 1
   case "$RUN_TAG" in *-"$POD_UID") ;; *) exit 1 ;; esac
   test "$AUTORESEARCH_REPO_DIR" = "$AUTORESEARCH_SOURCE_DIR/worktrees/$RUN_TAG" || exit 1
   BASE_COMMIT=$(cat /tmp/autoresearch-start-commit) || exit 1
   test -n "$BASE_COMMIT" || exit 1
   SOURCE_HEAD=$(git -C "$AUTORESEARCH_SOURCE_DIR" rev-parse HEAD) || exit 1
   SOURCE_STATUS=$(git --no-optional-locks -C "$AUTORESEARCH_SOURCE_DIR" status --porcelain) || exit 1
   test "$SOURCE_HEAD" = "$BASE_COMMIT" || exit 1
   test -z "$SOURCE_STATUS" || exit 1
   test ! -e "$AUTORESEARCH_REPO_DIR" || exit 1
   test ! -L "$AUTORESEARCH_REPO_DIR" || exit 1
   BRANCH_NAME="run/$(date -u +%Y%m%d_%H%M%S)-jigsaw-unintended-bias-in-toxicity-classification-baseline"
   mkdir -p "$(dirname "$AUTORESEARCH_REPO_DIR")" || exit 1
   git -C "$AUTORESEARCH_SOURCE_DIR" worktree add -b "$BRANCH_NAME" "$AUTORESEARCH_REPO_DIR" "$BASE_COMMIT" || exit 1
   cd "$AUTORESEARCH_REPO_DIR" || exit 1
   WORKTREE_ROOT=$(git rev-parse --show-toplevel) || exit 1
   test "$WORKTREE_ROOT" = "$AUTORESEARCH_REPO_DIR" || exit 1
   '
   ```
2. **Check the runtime context**: The Job reuses the user's preinstalled environment at `AUTORESEARCH_VENV=/workspace/autoresearch/.venv`, mounted read-only along with its Python interpreter. Before becoming ready, the Job checks actual CUDA forward and backward computation with this environment. The user enters the ready Pod and starts `claude` directly. This environment must already match the task's locked dependencies; if it is missing or unusable, stop and report the problem. Preserve `AUTORESEARCH_VENV` and `VIRTUAL_ENV`; do not install or sync packages, change the shared environment, or create a worktree `.venv`. Invoke `"$AUTORESEARCH_VENV/bin/python"` explicitly for every Python command: activation in the Job's startup shell does not carry into a new `kubectl exec` or shell tool call. The baseline Job has `activeDeadlineSeconds: 21600` (six hours). The Job's hard limit is measured from its `.status.startTime`; queueing, startup, and waiting for manual entry consume that time. Kubernetes terminates the Pod at that limit. The user's prompt may require earlier completion, such as six hours from the start of agent work; obey that research deadline even if the Pod can live longer. Record the Job deadline separately if its actual start time is known; otherwise leave it unknown rather than inventing it. Use the earliest known deadline for research and training, and reserve time to save results and signal shutdown. Do not add a separate timer script.
3. **Read the in-scope files**: Read `$AUTORESEARCH_TASK_FILE`, `$AUTORESEARCH_REPO_DIR/prepare.py`, and `$AUTORESEARCH_REPO_DIR/train.py` for the task, fixed data/evaluation, and editable implementation. All repository files and helpers mentioned below refer to this worktree. The main repository's `AGENTS.md` still applies; do not copy or edit it. `README.md` may be consulted for upstream/Jigsaw context only; the arm-specific scope below takes precedence over its links and instructions. This is the baseline arm: never read or call `analogy_agent.py`, `autoresearch_analogy/`, analogy configuration, the knowledge base, full-text cache, reports, or `program-analogy.md`. Do not use analogy snapshot/complete commands. Sharing the installed dependency lock does not authorize invoking analogy.
4. **Verify data exists**: Run `cd "$AUTORESEARCH_REPO_DIR" && "$AUTORESEARCH_VENV/bin/python" prepare.py verify --prepared-dir "$AUTORESEARCH_JIGSAW_DIR"`. Both arms use the same read-only prepared split at `/data/jigsaw-prepared/seed-42`. If it is missing or fails verification, stop and report the problem; do not prepare or repair it. Record `prepared_id` from its verified `manifest.json`.
5. **Initialize results.tsv once**: Only after successful worktree creation and data verification, run the subprocess below to create a new `$RUN_DIR` and its five-column TSV header. An existing directory or TSV is an error: stop without truncating, deleting, or reinitializing it. Record the actual branch, recorded starting Git SHA, main-repository path, worktree path, shared `AUTORESEARCH_VENV` path and resolved Python interpreter path, task-file SHA-256, verified prepared ID, initial seed, fixed resources, Job `RUN_TAG`, `MAX_TRAIN_CALLS`, `agent_started_at_utc`, the prompt-derived `research_deadline_utc`, the known `job_deadline_utc`, and their earliest known value as `deadline_utc` in `$RUN_DIR/run.json`. Use JSON `null` for an unknown deadline; include `run_tag: RUN_TAG` for the budget helper. Keep all new experiment artifacts under this run directory. Shared directories may be accessible; isolation relies on following this protocol. Do not read other arms' or previous runs' code, logs, models, reports, or histories.
6. **Initialize the training budget yourself**: After writing `run.json`, execute the budget initialization below using the Job's `MAX_TRAIN_CALLS`. The Job does not initialize it for you.
7. **Go**: Once these checks and initialization pass, start drafting. Do not wait for another confirmation.

```bash
bash -euo pipefail <<'BASH'
: "${POD_UID:?}" "${RUN_TAG:?}" "${AUTORESEARCH_SOURCE_DIR:?}" "${AUTORESEARCH_REPO_DIR:?}"
test "$EXPERIMENT_ARM" = "baseline" || exit 1
case "$RUN_TAG" in *-"$POD_UID") ;; *) exit 1 ;; esac
test "$AUTORESEARCH_REPO_DIR" = "$AUTORESEARCH_SOURCE_DIR/worktrees/$RUN_TAG" || exit 1
test -f "$AUTORESEARCH_REPO_DIR/.git" || exit 1
cd "$AUTORESEARCH_REPO_DIR" || exit 1
WORKTREE_ROOT=$(git rev-parse --show-toplevel) || exit 1
test "$WORKTREE_ROOT" = "$AUTORESEARCH_REPO_DIR" || exit 1
RUN_DIR="$AUTORESEARCH_REPO_DIR/results/$RUN_TAG"
PREPARED_ID=$("$AUTORESEARCH_VENV/bin/python" -c 'import json,os; print(json.load(open(os.path.join(os.environ["AUTORESEARCH_JIGSAW_DIR"], "manifest.json")))["prepared_id"])') || exit 1
mkdir -p "$AUTORESEARCH_REPO_DIR/results" || exit 1
mkdir "$RUN_DIR" || exit 1  # Exclusive creation; never use mkdir -p for RUN_DIR.
(set -o noclobber; printf 'commit\tval_score\tmemory_gb\tstatus\tdescription\n' > "$RUN_DIR/results.tsv") || exit 1
BASH
```

**Training budget initialization (agent action)**: After writing `run.json`, run this once in Bash before drafting or retrieval. The Job only supplies `MAX_TRAIN_CALLS` (default `10`); it does not run `init`. Use that positive integer unchanged in both arms. Record it in `run.json`. `deadline_utc` must be the earliest known research/Job deadline as an ISO timestamp with timezone, or JSON `null` if neither is known. Pass the recorded deadline to `init` when available; do not substitute the eight-hour Pod limit for a six-hour research deadline.

```bash
cd "$AUTORESEARCH_REPO_DIR" || exit 1
set -u
RUN_DIR="$AUTORESEARCH_REPO_DIR/results/$RUN_TAG"
: "${MAX_TRAIN_CALLS:?Job must supply the training-attempt limit}"
DEADLINE_UTC=$("$AUTORESEARCH_VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deadline_utc"] or "")' "$RUN_DIR/run.json") || exit 1
DEADLINE_ARGS=()
if [ -n "$DEADLINE_UTC" ]; then
  DEADLINE_ARGS=(--deadline-utc "$DEADLINE_UTC")
fi
"$AUTORESEARCH_VENV/bin/python" run_training.py init \
  --run-dir "$RUN_DIR" --max-train-calls "$MAX_TRAIN_CALLS" "${DEADLINE_ARGS[@]}"
```

Require successful initialization before continuing. On session reconnection, reuse the existing budget and query `run_training.py status --run-dir "$RUN_DIR"`; never repeat setup or `init`. Once initialized, `train_budget.json` and the attempt records are fixed bookkeeping: do not edit, delete, or reset them. One reserved `train.py` launch consumes one attempt, including OOM, code errors, interruption, or launch failure. A rejected pre-launch check consumes none. Epochs inside one candidate are not separate attempts; do not pack independent candidate searches into one invocation to evade the limit. The first draft and every debug retry count. Stop proposing new candidates when the budget is exhausted even if time remains. A zero `remaining` count can mean the final allowed training is still running: let that launch finish and record its result while time remains; do not interrupt it merely because its slot was consumed.

**Stay in this worktree**: After creation, explicitly start every shell tool call with `cd "$AUTORESEARCH_REPO_DIR" || exit 1`. One shell's `cd` does not change Claude's launch directory or later tool calls. Use absolute paths under `$AUTORESEARCH_REPO_DIR` with every file-editing tool; never edit a relative `train.py` from Claude's main-repository directory. Recreate these variables and the current trial paths inside each call, and fail on unset variables (`set -u`). Do not assume an earlier shell's assignments survive. Persist the best completed artifact path in `$RUN_DIR/best.json` and reload it before using `BEST_ARTIFACT_DIR`.

## Experimentation

Each experiment runs on a single GPU until the training-attempt budget, the research deadline, or the Job's hard deadline stops it, whichever comes first. Reading, reasoning, retrieval where enabled, editing, training, evaluation, debugging, and bookkeeping all consume research time. For pair007, baseline has a six-hour Job cap and analogy an eight-hour cap; a prompt requiring completion within six hours still applies to both. The extra Pod lifetime does not authorize extra research. Record actual times and attempt counts rather than claiming equal Job time, GPU compute, or LLM cost. Save results promptly. Check the current UTC time before retrieval, training, and finalization; do not start work that cannot finish in the remaining research time. `run_training.py` enforces the attempt limit and the deadline supplied at `init`; Kubernetes enforces the Job deadline independently.

**What you CAN do:**
- Modify `$AUTORESEARCH_REPO_DIR/train.py` — this is the only source file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Write run artifacts such as source snapshots, logs, metrics, configurations, predictions, and checkpoints under `$RUN_DIR`. Use the fixed helpers for snapshots and budgeted training with completion receipts.
- Signal final shutdown using the two `/tmp` marker files described below; these are lifecycle signals, not source edits.

**What you CANNOT do:**
- Modify `prepare.py` or the prepared data. They contain the fixed data split, labels, and evaluation. Fit learned preprocessing only on training rows; do not train on validation or private test labels.
- Modify the main checkout, another worktree, the program files, helpers, analogy implementation, or dependency files; install new packages; or change the environment. Use the shared, preinstalled dependencies only through `"$AUTORESEARCH_VENV/bin/python"`.
- Modify the evaluation harness. The `evaluate_predictions` function in `prepare.py` is the ground truth metric: the Jigsaw composite AUC using continuous probabilities.
- Extend or restart the Job, continue after a known deadline, start further candidates after the attempt budget is exhausted, or count incomplete, late, or failed candidates as successful results.
- Modify `run_training.py`, `train_budget.json`, or `execution.json`; bypass the launcher with direct `python train.py`, imported candidate execution, or an alternate training command. Every candidate training/debug attempt must go through `run_training.py run`.

**The goal is simple: get the highest val_score.** Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The code must run successfully and use the fixed Jigsaw evaluation.

**GPU and VRAM**: No GPU model or minimum VRAM is guaranteed. Record the actual model and VRAM in `run.json`, and choose model size, batch size, precision, and kernels supported by that GPU and the fixed environment. Do not assume BF16 or FlashAttention support. Some VRAM increase is acceptable for meaningful val_score gains, but stay within the assigned card's capacity.

Never silently fall back to CPU training. If CUDA initialization or a real CUDA operation fails because the assigned hardware or driver is incompatible with the fixed PyTorch build, record an infrastructure/environment failure and stop this arm. Preserve the error, valid results, and summary as described under **Finish and release the Pod**, then signal `/tmp/autoresearch-failed`; do not write a CPU replacement candidate or change the shared environment to continue.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_score improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_score improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: First draft a Jigsaw classifier from the upstream `train.py`, using `load_data()` and `evaluate_predictions()` from `prepare.py`, and remove the upstream language-model time-based stopping logic. `load_data()` returns training, validation, and test DataFrames; pass continuous probabilities in validation row order to `evaluate_predictions()` and print its returned `score` as `val_score`. This first completed candidate establishes this arm's initial result; do not import a previously optimized implementation.

**Randomness**: Initialize Python, NumPy, and Torch with seed `1337` in both arms. Record the seed in every candidate's configuration. A later seed change must be an explicitly described experimental variable.

**Candidate artifacts**: `train.py` must use `AUTORESEARCH_ARTIFACT_DIR` as its output directory and write `metrics.json` containing the complete dictionary returned by `evaluate_predictions()`, `config.json` describing the actual configuration, and `validation_predictions.npy` containing the same continuous probabilities in validation row order. Keep `run.log` and any checkpoints in that directory. Never edit a source snapshot, completion receipt, or finalized result. Every attempt, including a debug retry, gets a new ID and directory:

```bash
cd "$AUTORESEARCH_REPO_DIR" || exit 1
set -u
RUN_DIR="$AUTORESEARCH_REPO_DIR/results/$RUN_TAG"
PREPARED_ID=$("$AUTORESEARCH_VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prepared_id"])' "$RUN_DIR/run.json") || exit 1
TRIAL_ID=trial0001  # Increment for every attempt; never reuse a directory.
ARTIFACT_DIR="$RUN_DIR/trials/$TRIAL_ID"
"$AUTORESEARCH_VENV/bin/python" experiment_artifacts.py snapshot --source train.py \
  --artifact-dir "$ARTIFACT_DIR" --experiment-id "$TRIAL_ID" \
  --run-id "$RUN_TAG" --prepared-id "$PREPARED_ID" || exit 1
"$AUTORESEARCH_VENV/bin/python" run_training.py run \
  --run-dir "$RUN_DIR" --trial-id "$TRIAL_ID"
```

The snapshot must succeed before training starts. The launcher sets `AUTORESEARCH_ARTIFACT_DIR`, saves stdout/stderr to `run.log`, records `execution.json`, and automatically validates/finalizes a successful result. Do not redirect over `run.log` or run `complete` separately. Exit 0 means completed; 1 means a counted training/finalization failure; 2 means invalid input or another active launch; 3 means the budget is exhausted. Inspect the record/status for exit 2 rather than assuming another attempt was consumed. Do not change `train.py` until the launcher has finished finalizing it. A printed score alone is not a completed result. Keep the path of the best eligible candidate as `BEST_ARTIFACT_DIR`; restore its implementation with `cp "$BEST_ARTIFACT_DIR/source.py" "$AUTORESEARCH_REPO_DIR/train.py"` after rejecting or abandoning a later candidate. If the Job terminates during a candidate, its pending result is ineligible and the last completed best snapshot remains authoritative. Do not finalize an interrupted candidate later or after a known deadline.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_score:        0.900000
training_seconds: 842.3
total_seconds:    891.5
peak_vram_mb:     45060.2
```

Training duration depends on the model and training schedule. You can extract the key metric from the candidate's log file:

```bash
cd "$AUTORESEARCH_REPO_DIR" || exit 1
grep "^val_score:" "$ARTIFACT_DIR/run.log"
```

## Logging results

When an experiment is done, append it to `$RUN_DIR/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions). Never truncate this file or write its header again.

The TSV has a header row and 5 columns:

```
commit	val_score	memory_gb	status	description
```

1. First 7 characters of the snapshotted source SHA-256. The column name `commit` is retained for compatibility; this is not a Git commit.
2. val_score achieved (e.g. 0.900000), only from an eligible completed result; leave blank for crashes, incomplete runs, or deadline expiry.
3. Peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024); leave blank if unavailable.
4. Status: `keep`, `discard`, or `crash` (including incomplete or timed-out attempts).
5. Short text description of what this experiment tried, including its trial ID and failure reason if applicable.

Example:

```
commit	val_score	memory_gb	status	description
a1b2c3d	0.900000	44.0	keep	trial0001 initial Jigsaw classifier
b2c3d4e	0.905000	44.2	keep	trial0002 increase LR to 0.04
c3d4e5f	0.897000	44.0	discard	trial0003 switch to GeLU activation
d4e5f6a			crash	trial0004 double model width (OOM)
```

## The experiment loop

The experiment runs only in the worktree and branch you created during Setup and recorded in `$RUN_DIR/run.json`. After that initial creation, Git branches, the index, and history stay unchanged; source snapshots identify candidates.

LOOP UNTIL THE TRAINING BUDGET IS EXHAUSTED, A DEADLINE IS REACHED, JOB TERMINATION, OR HUMAN INTERRUPTION:

1. Explicitly enter `$AUTORESEARCH_REPO_DIR`, check its branch/starting commit with read-only commands and the actual UTC time remaining against the recorded deadline. Query `"$AUTORESEARCH_VENV/bin/python" run_training.py status --run-dir "$RUN_DIR"`. If `remaining` is zero, wait for any already-running final attempt and record its result within the deadline, then go to **Finish and release the Pod**. If time is up, stop active work and finish; do not start another idea or retrieval call. Identify the current best completed candidate, if any.
2. Use your own reasoning from the allowed task, code, and this run's results; do not use analogy. Tune `train.py` with one experimental idea by directly hacking the code. Start improvements from the best snapshot.
3. Allocate a new trial ID and take the pre-execution source snapshot with the neutral `experiment_artifacts.py` helper above.
4. Run the experiment only through `"$AUTORESEARCH_VENV/bin/python" run_training.py run --run-dir "$RUN_DIR" --trial-id "$TRIAL_ID"`. The launcher saves training output to the trial log and prints a small execution summary. Exit 3 goes directly to final bookkeeping and shutdown.
5. Read out the results: `grep "^val_score:\|^peak_vram_mb:" "$ARTIFACT_DIR/run.log"`. Check the launcher exit status, `execution.json`, and the completed `source.json` receipt. No separate completion command is needed. Finish recording the last allowed trial even when `remaining` has just reached zero.
6. If training or completion fails, inspect `tail -n 50 "$ARTIFACT_DIR/run.log"` and the helper error. Log the failed attempt. An easy fix may be tried only with remaining attempts and time, but it needs a new trial ID and snapshot; never finalize the old attempt with new code.
7. Record the result in the TSV. Only a valid completed receipt makes a candidate eligible for `keep` or `discard`.
8. If val_score improved (higher), keep this candidate and update `BEST_ARTIFACT_DIR` and `$RUN_DIR/best.json` with its artifact path, source SHA, and score. The first eligible result becomes best; an equal score may be kept only for a clear simplification.
9. If the candidate is rejected or crashes, restore `$AUTORESEARCH_REPO_DIR/train.py` from `BEST_ARTIFACT_DIR/source.py` when a best exists. Preserve all trial artifacts. If no candidate has completed yet, continue the draft/debug process only with remaining attempts and time.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. Advance the best source snapshot so that you can iterate, without changing Git branches, the index, or history.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: if it's something easy to fix, fix it in a new attempt. If the idea itself is fundamentally broken, skip it, log `crash`, restore the best implementation, and move on.

**Continue autonomously while the Job runs**: Do not pause to ask the human whether to continue. If you run out of ideas, re-read the in-scope files, combine previous near-misses, or try another architectural change. Keep logs, completed receipts, and `best.json` up to date; do not defer saving everything until the end. Stop when the training budget is exhausted, the recorded research/Job deadline is reached, the Job terminates, or the human interrupts you. Never restart the Job to continue or finalize results after a known deadline. Report the best eligible result and its artifact path when possible.

**Finish and release the Pod**: If the entire research task finishes before Kubernetes stops the Job (for example, a user-specified stopping condition is met), for budget exhaustion, first let the already-running final allowed attempt finish within the deadline. For deadline expiry, human interruption, or an unrecoverable failure, stop active training. Stop any outstanding retrieval before shutdown. Save eligible results and the best artifact path, and write `$RUN_DIR/summary.md` with the outcome, stopping reason, and final `run_training.py status` counts. Budget exhaustion is a normal stopping condition; preserve and report the best eligible result even if fewer than `MAX_TRAIN_CALLS` attempts succeeded. Then run `touch /tmp/autoresearch-finished` as your final shell command. For an unrecoverable failure that prevents continuing, save the error and any valid results, then run `touch /tmp/autoresearch-failed` instead. If setup failed before a run directory could be created, report the error in the conversation before signaling failure. The Job exits within a few seconds and disconnects Claude, so do not leave summaries to a later reply. Do not stop after just one candidate, one reply, or a recoverable training error while attempts and research time remain. Once the attempt budget or research deadline is reached, finish bookkeeping and signal shutdown; do not idle until the Job's hard limit. Kubernetes remains the fallback if you cannot write a marker.
