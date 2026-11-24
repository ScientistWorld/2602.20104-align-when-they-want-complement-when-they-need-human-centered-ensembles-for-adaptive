#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script is idempotent: every fetch is skipped when its target exists.
#
# Estimated total download size: < 1 MB
# Estimated disk usage after extraction: < 1 MB
#
# The gym's data is procedural: data/<experiment>/ is tracked in git and the
# hidden test (eval/test/) is rebuilt by scripts/build_world.sh, so nothing
# here is needed to score a method. The only external artifact is the
# authors' public code release, kept as a read-only reference next to the
# reimplementation (method/college_admission_method.py) of the paper's own
# College Admission sweep. No code imports it.

set -e

cd "$(dirname "$0")/.."

mkdir -p data/downloads

# Authors' release (https://github.com/shasanamin/aaai26-adaptive-ai), pinned.
UPSTREAM_DIR=method/aaai26-adaptive-ai
UPSTREAM_COMMIT=28892a0797b71aaaa5f4ddc2f1cada3315492814
if [ ! -f "$UPSTREAM_DIR/college_admission.py" ]; then
  rm -rf "$UPSTREAM_DIR"
  git clone https://github.com/shasanamin/aaai26-adaptive-ai "$UPSTREAM_DIR"
  git -C "$UPSTREAM_DIR" checkout --quiet "$UPSTREAM_COMMIT"
fi
