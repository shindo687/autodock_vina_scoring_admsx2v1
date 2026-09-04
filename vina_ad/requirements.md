# vina-ad requirements and provenance

- Official source URL: `https://github.com/ccsb-scripps/AutoDock-Vina.git`
- Fixed upstream commit: `3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645`
- Snapshot imported at: `2026-09-04T00:00:00Z` (UTC; repository task run)
- The source was cloned from the official URL, copied into `upstream/`, and its
  clone-local `.git` directory was removed before the import commit. No
  upstream source file is modified by this sidecar.
- AD protocol: tested with `chainrules==0.1.0` (the documented
  JVP/VJP/grad/value_and_grad protocol). ChainRules is optional at install time:
  when absent, the sidecar's small protocol-compatible fallback keeps the
  public API usable; when present, the real ChainRules registry is used.
  `numpy` is optional; pure Python sequences are supported.
- Supported runtime: CPython 3.10-3.12, Linux/macOS/Windows, IEEE-754 real
  floating point. The restricted sidecar kernel does not require the compiled
  upstream `vina_wrapper`; the sourced official workflow additionally requires
  an installed `vina` binding and explicit PDBQT paths when run outside this
  checkout. It invokes real Vina scoring/one-step optimization plus public
  sidecar `value_and_grad`/`jvp`; unavailable bindings or inputs are explicitly
  reported as deferred.
- Sidecar release: `vina-ad 0.2.0`.
- Evidence bindings: implementation `vina_ad/core.py` (three scoring families,
  public term decomposition, and precomputed-term recombination); public exports
  `vina_ad/__init__.py`; workflow `vina_ad/workflow.py`; tests `tests/`;
  command totals and environment are in
  `/root/ad_xjtan_v4pro/tasks/task-1/artifacts/fix_receipt_round1.json`.
