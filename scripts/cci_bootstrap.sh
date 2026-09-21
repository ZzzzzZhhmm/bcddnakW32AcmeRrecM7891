#!/usr/bin/env bash
# CCI/k8s GPU servers in this environment have no outbound Git/GitHub.
# Launchers must never fetch, pull, push, ls-remote, or fail on git status.
set -euo pipefail
echo "cci_bootstrap: skip (offline server; no git/network operations)"
exit 0
