#!/usr/bin/env bash

set -euo pipefail

# 切换到大仓库根目录
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# -----------------------------
# 检查 working tree 是否干净
# -----------------------------
check_clean() {
    echo "Refreshing Git index..."

    git update-index --really-refresh >/dev/null 2>&1 || true

    if ! git diff-index --quiet HEAD --; then
        echo "ERROR: Working tree has modifications."
        echo
        git diff-index --name-status HEAD -- | head -50
        echo
        echo "Please commit or stash your changes before pushing subtrees."
        exit 1
    fi

    if ! git diff-index --cached --quiet HEAD --; then
        echo "ERROR: There are staged changes."
        echo
        git diff --cached --name-status | head -50
        echo
        echo "Please commit your staged changes before pushing subtrees."
        exit 1
    fi
}

# -----------------------------
# Push 一个 subtree
#
# 参数:
#   $1 = prefix
#   $2 = remote
#   $3 = branch
# -----------------------------
push_subtree() {
    PREFIX="$1"
    REMOTE="$2"
    BRANCH="$3"

    echo
    echo "=================================================="
    echo "Pushing subtree:"
    echo "  prefix : $PREFIX"
    echo "  remote : $REMOTE"
    echo "  branch : $BRANCH"
    echo "=================================================="

    check_clean

    # 先刷新远程状态
    echo "Fetching $REMOTE..."
    git fetch "$REMOTE"

    # Push subtree
    git subtree push \
        --prefix="$PREFIX" \
        "$REMOTE" "$BRANCH"

    echo "✓ Finished: $PREFIX"
}


echo "Starting subtree push..."
echo "Repository: $ROOT"


# 1. Agentic_Knowledge_Base
push_subtree \
    "Agentic_Knowledge_Base" \
    "Agentic_Knowledge_Base" \
    "feature/analogy-retrieval"


# 2. autoresearch
push_subtree \
    "autoresearch" \
    "autoresearch" \
    "codex/jigsaw-unintended-bias"


# 3. MLEvolve
push_subtree \
    "MLEvolve" \
    "MLEvolve" \
    "main"


echo
echo "=================================================="
echo "✓ All subtrees pushed successfully."
echo "=================================================="

git status