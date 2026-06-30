# v1.3.0 — Changelog Reports

Detailed implementation notes for all changes in this release.

| File | Topic |
|------|-------|
| `noise_settings_refactor.md` | Kernel noise refactor (v1→v2 schema), `noise_settings` TypedDict, schema versioning & migration system (including v3 fix) |
| `noise_v1_implementation.md` | V1 observation noise feature — normaliser scale transform, `noise_std` on `Objective`, GP model pinning, noisy acquisition functions, Pareto front uncertainty, interface exposure |
| `rollback_implementation.md` | Experiment rollback utility — rolls back JSONs and simulation folders to a previous batch boundary |
| `bug_fixes.md` | NumPy 2.4 fix (issue #22), noisy multi-objective reload crash fix, `continue_with_new_version` phantom-points + double-load fix |

