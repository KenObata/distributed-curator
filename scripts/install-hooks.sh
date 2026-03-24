#!/usr/bin/env bash
# ==============================================================================
# This installer is the one-time setup step only.
# Run from repo root:  ./scripts/install-hooks.sh
# Git hooks live in .git/hooks/, but .gitignore exists, 
# so this script creates a symlink pre-commit to .git/hooks/
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# makes the script work correctly regardless of which subdirectory you run it from 
REPO_ROOT="$(git rev-parse --show-toplevel)" 
HOOK_SRC="${REPO_ROOT}/scripts/pre-commit"
HOOK_DST="${REPO_ROOT}/.git/hooks/pre-commit" # where we create symlink

echo "Installing pre-commit hook..."

# ── Install hook ──────────────────────────────────────────────────────────────

# -f: file exists at that path ?
# -L: symlink alreadt exists with that path? 
if [[ -f "$HOOK_DST" ]] || [[ -L "$HOOK_DST" ]]; then
    printf "${YELLOW}Existing pre-commit hook found. Overwrite? [y/N] ${NC}"
    read -r answer # -r flag prevents backslash interpretation
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -f "$HOOK_DST"
fi

ln -s "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_SRC"
printf "${GREEN}✓${NC} Hook installed: .git/hooks/pre-commit → scripts/pre-commit\n\n"

# ── Verify tooling ───────────────────────────────────────────────────────────
echo "Checking required tools..."

check_tool() {
    local name="$1"
    local install_hint="$2"
    if command -v "$name" &>/dev/null; then
        printf "  ${GREEN}✓${NC} %-12s %s\n" "$name" "$(command -v "$name")"
    else
        printf "  ${RED}✗${NC} %-12s ${YELLOW}%s${NC}\n" "$name" "$install_hint"
    fi
}

check_tool pytest     "pip install pytest" # name, install_hint
check_tool ruff       "pip install ruff"
check_tool sbt        "brew install sbt"

# Check scalafmt sbt plugin
# unlike pytest and ruff which are standalone binaries, scalafmt is an sbt plugin declared in a config file, so command -v wouldn't find it.
if [[ -f "${REPO_ROOT}/project/plugins.sbt" ]]; then
    # -q = quiet mode 
    if grep -q "scalafmt" "${REPO_ROOT}/project/plugins.sbt" 2>/dev/null; then
        printf "  ${GREEN}✓${NC} %-12s %s\n" "scalafmt" "project/plugins.sbt"
    else
        printf "  ${RED}✗${NC} %-12s ${YELLOW}Add to project/plugins.sbt${NC}\n" "scalafmt"
    fi
else
    printf "  ${RED}✗${NC} %-12s ${YELLOW}Create project/plugins.sbt${NC}\n" "scalafmt"
fi

echo ""
echo "Done. Pre-commit hook will run on your next git commit."
echo "Bypass with:  git commit --no-verify"