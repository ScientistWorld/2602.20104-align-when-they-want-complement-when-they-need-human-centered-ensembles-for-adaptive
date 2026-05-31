#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This reproduction uses procedural data, so the script only prepares the
# downloads directory expected by the workspace layout.

set -e

cd /home/user

# The College Admission benchmark is generated procedurally from tracked code.
mkdir -p data/downloads
