# Repository collaboration rules

- Keep real-robot integration changes under `src/fastwam/real`,
  `scripts/real`, `configs/real`, `docs/real`, and their tests where possible.
  Do not change the simulation/training main path just to scaffold hardware.
- Unknown hardware, calibration, action semantics, and checkpoint properties
  must stay explicitly unverified. Synthetic checks do not qualify real motion.
- For every commit, use an English subject in this format:
  `Feat(module): Summarize the change` (or Fix, Chore, Refactor, Docs, Test).
  Follow it with a blank line and English bullet points describing the changes
  and relevant validation.
- After every successful local commit, push it to the current branch's
  configured upstream and verify the remote contains it. Never force-push.
  If the push fails (this CCI/k8s node often has no GitHub route), keep the
  local commit and report the actual failure.
- Training, ACP, and eval must never fetch, pull, ls-remote, probe origin, or
  fail on a dirty worktree. Do not run `scripts/cci_bootstrap.sh` network
  checks. Jobs run from the local checkout as-is.
- Keep datasets, checkpoints, recorded media, credentials, and generated runtime
  outputs out of Git. Commit only source, small examples, tests, and handoff docs.
- As soon as experiment values or existing result artifacts are available,
  preserve the raw evidence outside Git and update `docs/nonreal72h/EXPERIMENT_RECORD_ZH.md`.
  Record counts, failures, scope, configuration/checkpoint/source identities,
  archive locations and hashes, and the exact paper claim the evidence supports.
  Keep engineering checks, historical results, and paper-ready results distinct;
  never replace an unresolved paper number with a different run silently.
