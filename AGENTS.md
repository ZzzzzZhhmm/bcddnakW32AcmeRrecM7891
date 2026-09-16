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
- After every successful local commit, push it to the current branch's configured
  upstream and verify the remote contains it. The user has authorized this
  workflow. Never force-push or discard others' work. If synchronization fails,
  preserve the commit and report the actual failure; do not claim it was synced.
- Keep datasets, checkpoints, recorded media, credentials, and generated runtime
  outputs out of Git. Commit only source, small examples, tests, and handoff docs.
