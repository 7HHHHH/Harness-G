#!/usr/bin/env bash

# Shared paths for the versioned experiment workflow. Runtime artifacts stay
# outside Git, while all executable code comes from the current repository.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
ROOT=${EXPERIMENT_ROOT:-${ROOT:-$HOME/harness_g_experiments}}
