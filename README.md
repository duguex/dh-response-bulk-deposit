# Data and software for "Calibrating Machine-Learned Electronic Hamiltonians against Derivatives"

Deposit accompanying manuscript **ct-2026-02052b** (Journal of Chemical Theory and Computation),
Mingzhe Liu and Chang-Kui Duan, University of Science and Technology of China.

## What is here

| Path | Contents |
|---|---|
| `data/labels/` | Finite-difference derivative label samples, 240 files (20 geometries × 4 probe atoms × 3 Cartesian axes, δ = 0.001 Å). Each `.npz` stores the QC-gated midpoint Hamiltonian block `h_mid` and the derivative block `dh_dr` (per-Bohr, float32) for one displacement, in the packed graph-block layout of the training code. `qc.csv`, `audit.json`, `manifest.json` carry the per-sample QC gates; `RESCALE_NOTE.json` documents the 2026-09-13 unit-incident correction (per-Å → per-Bohr, factor 1.889716) applied to this cache. |
| `data/geometries/` | The complete 20-cell input set (one perfect-cell stem `Perf_chg0` + 19 displaced stems `Bulk01–19`, neutral cells) as `.npz` graphs plus `bulk_cloud_meta.json`. These are the only geometries used for training, validation, and test. |
| `data/tables/bulk_band_response/` | Band-edge projection evaluations: `eval_canonical.json` holds the records behind the band-edge table (Table 2) — per-component rows (`g_ref_win`/`g_mod_win` are the 3×3 reference and model response windows, from which the Table 3 level shifts follow once multiplied by the zero-point amplitude) plus per-arm summary metrics including `noise_floor_median`; `eval.json`, `eval_phase3*.json`, `eval_with_w.json` and the run log are the superseded earlier passes, kept for provenance. |
| `data/tables/train/` | Training-run records: `fold_eval_baselines.json` (test-set matrix-level errors for the frozen baseline and the H-only fine-tune), `fold_eval_percomp.json` (test-fold matrix and derivative errors for the main λ arms), `histories/*/` (per-epoch training/validation curves for every run), and `run_logs/` (per-run console records including the test-fold lines quoted in Table 1, and the matched-step λ-vs-learning-rate control in `run_logs/probe_lam_vs_lr/`). |
| `data/tables/richardson/` | δ-Richardson label-truncation audit (`delta_floor_report.json`; the 2026-09-13 pass kept as the legacy file): the label-side floor row of the error-budget table (Table 4). |
| `data/tables/label_audit/` | Bulk side of the finite-difference label audit (`bulk.csv`). The audit's charged-defect rows belong to a separate study and are excluded. |
| `data/provenance/` | Canonical-run log (seeds and stage budgets) for the derivative-supervised training chain. |
| `config/` | `dh_bulk_manifest.yaml`: geometry list, 12/4/4 geometry folds, probe-atom indices, displacement step. |
| `scripts/diff_hamgnn/` | The reproduction chain: displacement-cloud generation (`make_bulk_displacement_cloud.py`, `gen_delta_inputs.py`), label building (`build_dh_label_cache.py`, `openmx_fd.py`, `assemble.py`), the λ-joint-loss fine-tuner (`derivative_training.py`, `joint_loss.py`, `dh_labels.py`), and the evaluators and audits (`bulk_fold_eval.py`, `bulk_band_response_eval.py`, `evaluate_dh_response.py`, `audit_dh_labels.py`, `delta_floor_analysis.py`, `scenario_epc.py`). |
| `figures/` | Generator and underlying curve data for Fig. 1 (`make_fig_results.py`, `make_fig_train_curves.py`, `fig_bulk_train_curves_data.csv`). |

## Manuscript numbers → files

| Manuscript item | Machine-readable source |
|---|---|
| Table 1 (static + derivative errors per λ; baseline; H-only) | `data/tables/train/fold_eval_baselines.json` (baseline, H-only) + `data/tables/train/fold_eval_percomp.json` (λ = 0.5, 1.0 arms) + test-fold lines in `data/tables/train/run_logs/` (λ = 0.03, 0.1, 0.3, 2.0 runs) |
| Matched-step λ-vs-learning-rate control | `data/tables/train/run_logs/probe_lam_vs_lr/*.log` |
| Table 2 (band-edge W errors; model-over-reference ratios) | `data/tables/bulk_band_response/eval_canonical.json` (`arms.*.metrics` and per-component `rows`) |
| Table 3 (coupling under zero-point LO displacement) | `scripts/diff_hamgnn/scenario_epc.py` (prints the table from the label cache and checkpoints); the underlying per-component response windows are in the `eval_canonical.json` rows (`g_ref_win`, `g_mod_win`) |
| Table 4, model levels | as Table 1 (λ = 0.5, 1.0 entries) |
| Table 4, label-truncation floor | `data/tables/richardson/delta_floor_report.json` (median pair-based floor) |
| Table 4, projection noise floor | per-arm `noise_floor_median` in `eval_canonical.json`; computed by `bulk_band_response_eval.py` and stored per component |
| Fig. 1 (a,b) | `figures/fig_bulk_train_curves_data.csv` + `data/tables/train/histories/*/` |
| Fig. 1 (c,d) | `data/tables/train/fold_eval_percomp.json` and the run logs above |

## What is deliberately not here

- **HamGNN-family checkpoint and OpenMX basis/pseudopotential files** — third-party
  assets; see `data/THIRD_PARTY_NOT_SHIPPED.md` for their published sources.
- **Raw OpenMX `*.scfout` outputs** (~110 GB) — not required to reproduce any reported
  number; the QC-gated packed label samples are the record the manuscript consumes.
- **Fine-tuned checkpoints** of the derivative-supervised and H-only models — they are
  regenerated by `scripts/diff_hamgnn/derivative_training.py` from the published base
  checkpoint with the seeds and stage budgets in `data/provenance/` and
  `config/dh_bulk_manifest.yaml`.

## Reproducing the headline numbers

1. Install PyTorch + the published HamGNN code base (see scripts' import comments) and
   fetch the base checkpoint listed in `data/THIRD_PARTY_NOT_SHIPPED.md`.
2. Train: `scripts/diff_hamgnn/derivative_training.py` with `--cache-root data/labels`
   and the fold/weight schedule of `config/dh_bulk_manifest.yaml` and the seed/budget
   log in `data/provenance/`.
3. Evaluate: `bulk_fold_eval.py` (Table 1), `bulk_band_response_eval.py` (Table 2,
   Fig. 1 arms, noise floors), `scenario_epc.py` (Table 3), `delta_floor_analysis.py`
   (Table 4 label floor; it documents the two smaller-step label caches it reads).

## Licenses

- Software (all files under `scripts/`, `figures/`, `config/`): **MIT** — see `LICENSE.code`.
- Data (all files under `data/`): **CC-BY-4.0** — see `LICENSE.data`.

`MANIFEST.sha256` lists SHA-256 digests of every file in this deposit.
