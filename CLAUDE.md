# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An umbrella repo for one research project: **do research papers, retrieved at run time, make an ML-engineering agent better?** It holds three independent projects, each vendored as a **git subtree** with its own remote, dependencies, and docs. There is no top-level build, test suite, or Python environment — work happens inside one subproject at a time, run from that subproject's directory.

| dir | role | subtree remote / branch |
|---|---|---|
| `Agentic_Knowledge_Base/` | Offline half: scrapes conference papers, builds the paper corpus / KB / retrieval indexes, and the **analysis tier** that judges whether agent runs support any claim | `Agentic_Knowledge_Base` / `feature/analogy-retrieval` |
| `MLEvolve/` | Online half: MLE-bench agent doing Monte Carlo Graph Search over candidate solutions; consumes the KB (analogy retrieval in `engine/analogy/`) | `MLEvolve` (`MLEvolve-externalKB`) / `main` |
| `autoresearch/` | Fork of karpathy/autoresearch retargeted at Jigsaw Unintended Bias; a Claude-in-a-pod experiment comparing a **baseline** arm (`program.md`) with an **analogy** arm (`program-analogy.md`) | `autoresearch` / `codex/jigsaw-unintended-bias` |

Each subproject has its own guidance — read it before working there:
- `MLEvolve/CLAUDE.md` (detailed architecture, config, arms A–E, k8s launch). `MLEvolve/AGENTS.md` is a near-verbatim copy for Codex; **change both together**.
- `Agentic_Knowledge_Base/README.md`, `UPDATELOG.md` (change record, newest first — read the relevant entry before "fixing" something that looks odd), and `docs/` (design docs, many in EN + `.zh.md` pairs).
- `autoresearch/docs/` (Chinese; `analogy_agent.md`, `jigsaw_comparison.md`, `run_analysis.md`). Its `README.md` is mostly upstream karpathy text. `autoresearch/CLAUDE.md` / `AGENTS.md` are gitignored — generated per session by the k8s launchers, so don't create or commit them.

## Subtree sync

Commits are made in this umbrella repo; subtrees are synced with the root scripts, which refuse to run on a dirty tree (commit or stash first):

```bash
./pull-subtrees.sh   # git subtree pull for all three prefixes
./push-subtrees.sh   # git subtree push for all three prefixes
```

Keep a commit's changes within one subtree prefix where possible so subtree push splits cleanly. The branch each subtree tracks is hardcoded in both scripts — change both if it changes.

## How the pieces connect

- **Paper corpus contract.** `Agentic_Knowledge_Base/scripts/6_build_paper_corpus.py` writes `output/paper_corpus/records.jsonl` (title + tldr + abstract). MLEvolve's `engine/analogy/corpus.py` builds BM25 over it (`analogy.corpus_path`); in pods it lives at `/workspace/Agentic_Knowledge_Base/output/paper_corpus`. Older embedding-index modes use `manifest.json` as the contract — the consumer must load the same embedding model (default `BAAI/bge-m3`).
- **Analogy agent is duplicated, not shared.** `autoresearch/autoresearch_analogy/` is a standalone port of `MLEvolve/engine/analogy/` (same module names: `agent`, `corpus`, `context`, `fulltext`, `observed_loop`, `report_v2`, `code_tools`). autoresearch never imports MLEvolve. A fix in one usually needs porting to the other.
- **Analysis lives in the KB repo.** Grading happens on the cluster (`MLEvolve/utils/grade_all.py`); `Agentic_Knowledge_Base/scripts/analyze_runs.py`, `measure_adoption.py`, `show_models.py`, `plot_effects.py`, and the `vendi_*` scripts read raw MLEvolve run dirs (local-only, not in git) and write derived tables/figures to `Agentic_Knowledge_Base/results/<date>/`. autoresearch has its own `analyze_runs.py` / `compare_solution_vendi.py` writing to `autoresearch/results/analysis/`.
- **Analysis vocabulary is load-bearing** (see KB README "Analysis"): arms are recovered from each run's `logs/config.yaml`, never the directory name; `agent.seed` is **not** a draw id; comparisons are only at matched ensemble size K; thresholds live in the `Thresholds` dataclass in `analyze_runs.py`. Characterize what a run explored via `show_models.py` (all nodes), not `logs/best_solution.py`.

## Commands

**Agentic_Knowledge_Base** (`pip install -r requirements.txt`; configure `.env` from `.env.example`; all chat completions go through `scripts/llm.py` — use `from llm import client, MODEL`, never construct a client or hardcode a model):

```bash
bash run_all.sh icml 2024                                  # steps 1–4 + paper corpus (FULL_METHODOLOGY=1 adds step 5)
python scripts/6_build_paper_corpus.py --venues all        # corpus only; no LLM, seconds
python -m unittest discover -s tests                       # run from Agentic_Knowledge_Base/ (tests/ is not a package)
python -m unittest discover -s tests -p test_vendi_metrics.py   # single test module
```

`methodology_kb/paperinsight/` is its own nested git repo committed to by the plugins; never commit to it concurrently.

**autoresearch** (uv; `uv sync --only-group analogy` for CPU-only retrieval work):

```bash
uv run python -m unittest discover -s tests                # run from autoresearch/; needs the uv env (analogy group)
uv run python -m unittest discover -s tests -p test_run_training.py   # single test module
```

`train.py` is the only file the experiment agent edits; `prepare.py` is fixed. `run_training.py` enforces a per-run training-attempt budget and deadline; `experiment_artifacts.py` is the arm-neutral artifact helper. The `program*.md` protocols are strict and paired — changes to shared setup/budget/artifact steps must be mirrored in both arms so the comparison stays fair. `k8s/job-jigsaw-pairNNN-{baseline,analogy}.yaml` launch the paired runs.

**MLEvolve**: no test suite; see `MLEvolve/CLAUDE.md` for the layered `--no-deps` install (Python ≥ 3.11) and `bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]`.
