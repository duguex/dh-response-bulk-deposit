# Data and software for "Calibrating Machine-Learned Electronic Hamiltonians against Derivatives"

Deposit accompanying manuscript **ct-2026-02052b** (Journal of Chemical Theory and Computation),
Mingzhe Liu and Chang-Kui Duan, University of Science and Technology of China.

The deposit is held private during editorial review (reviewers only) and will be made
public upon acceptance, at which point this record receives a DOI.

## What is here

| Path | Contents |
|---|---|
| `data/labels/` | Finite-difference derivative label samples, 240 files (20 geometries × 4 probe atoms × 3 Cartesian axes, δ = 0.001 Å). Each `.npz` stores the QC-gated midpoint Hamiltonian block `h_mid` and the derivative block `dh_dr` (per-Bohr, float32) for one displacement, in the packed graph-block layout of the training code. `qc.csv`, `audit.json`, `manifest.json` carry the per-sample QC gates; `RESCALE_NOTE.json` documents the 2026-09-13 unit-incident correction (per-Å → per-Bohr, factor 1.889716) applied to this cache. |
| `data/geometries/` | The complete 20-cell input set (one perfect-cell stem `Perf_chg0` + 19 displaced stems `Bulk01–19`, neutral cells) as `.npz` graphs plus `bulk_cloud_meta.json`. These are the only geometries used for training, validation, and test. |
| `data/tables/` | Machine-readable result tables behind the manuscript's figures and tables: band-edge response evaluations (`bulk_band_response/eval*.json`), static-error and gate summaries (`g0/`, `g4/`, `g4_mixed/`, `uni_static_admission.json`), and the bulk side of the FD-label audit (`label_audit/bulk.csv`; the audit's charged-defect rows belong to a separate study and are excluded). |
| `data/provenance/` | Canonical-run log (seeds and stage budgets) for the derivative-supervised training chain. |
| `config/` | `dh_bulk_manifest.yaml`: geometry list, 12/4/4 geometry folds, probe-atom indices, displacement step. |
| `scripts/diff_hamgnn/` | The reproduction chain: displacement-cloud generation (`make_bulk_displacement_cloud.py`), label building (`build_dh_label_cache.py`, `openmx_fd.py`, `assemble.py`), the three-arm λ-joint-loss fine-tuner (`derivative_training.py`, `joint_loss.py`, `dh_labels.py`), and the evaluators (`bulk_fold_eval.py`, `bulk_band_response_eval.py`, `evaluate_dh_response.py`, `audit_dh_labels.py`). |
| `figures/` | Generator and underlying curve data for Fig. 1 (`make_fig_results.py`, `make_fig_train_curves.py`, `fig_bulk_train_curves_data.csv`). |

## What is deliberately not here

- **HamGNN-family checkpoint and OpenMX basis/pseudopotential files** — third-party
  assets; see `data/THIRD_PARTY_NOT_SHIPPED.md` for their published sources.
- **Raw OpenMX `*.scfout` outputs** (~110 GB) — not required to reproduce any reported
  number; the QC-gated packed label samples are the record the manuscript consumes.

## Reproducing the headline numbers

1. Install PyTorch + the published HamGNN code base (see scripts' import comments) and
   fetch the base checkpoint listed in `data/THIRD_PARTY_NOT_SHIPPED.md`.
2. Train: `scripts/diff_hamgnn/derivative_training.py` with `--cache-root data/labels`
   and the fold/weight schedule of `config/dh_bulk_manifest.yaml` and the deposit log
   in `data/provenance/`.
3. Evaluate: `bulk_fold_eval.py` (Tables 1–2) and `bulk_band_response_eval.py`
   (band-edge response, Tables 3–4 and Fig. 1 arms).

## Licenses

- Software (all files under `scripts/`, `figures/`, `config/`): **MIT** — see `LICENSE.code`.
- Data (all files under `data/`): **CC-BY-4.0** — see `LICENSE.data`.

`MANIFEST.sha256` lists SHA-256 digests of every file in this deposit.
