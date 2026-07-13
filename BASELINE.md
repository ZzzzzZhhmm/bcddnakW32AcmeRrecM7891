# WARM baseline provenance

This private repository starts from a clean source snapshot of FastWAM rather
than a GitHub fork.

- Upstream project: `yuantianyuan01/FastWAM`
- Imported revision: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- Revision date: 2026-04-03
- Upstream license: MIT
- Import method: `git archive`; upstream Git history and remotes are not kept
- WARM remote: `ZzzzzZhhmm/WARM` (private)

The upstream MIT copyright and license text must remain in every copy or
substantial derivative. Keeping that attribution does not create an upstream
GitHub fork, pull request, or public development record.

WARM-specific memory and offline-pipeline modules live under
`src/fastwam/memory` so they can share the baseline package's action/data
contracts without a second shadow package. Model integrations will remain
explicitly named WARM components inside `src/fastwam`; the imported baseline is
retained as an attributed compatibility layer until checkpoint and evaluation
parity are established.
