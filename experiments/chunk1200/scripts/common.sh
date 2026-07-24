#!/usr/bin/env bash

# Shared paths for the versioned chunk1200 workflow. Runtime artifacts stay
# outside Git, while all executable code comes from the current repository.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ROOT=${CHUNK1200_ROOT:-${ROOT:-$HOME/harness_g_chunk1200_experiment}}
